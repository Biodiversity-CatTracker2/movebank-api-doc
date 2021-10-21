#!/usr/bin/env python
# coding: utf-8

# Author of original script: Philipp Meschenmoser, DBVIS, Uni Konstanz
# Python wrapper with functions using Movebank's REST API to view available studies, read data and accept license terms programmatically
# Acknowledgements to Anne K. Scharf and her great moveACC package, see https://gitlab.com/anneks/moveACC

import csv
import hashlib
import io
import json
import os
import requests
from datetime import datetime, timedelta

import keyring
import numpy as np
import pandas as pd


class MovebankAPI:
    def __init__(self, username, password, study_id=None):
        self.username = username
        self.password = password
        self.study_id = study_id

    def callMovebankAPI(self, params):
        """"
        params: Requests Movebank API with ((param1, value1), (param2, value2),).
        Return the API response as plain text.
        """
        response = requests.get(
            'https://www.movebank.org/movebank/service/direct-read',
            params=params,
            auth=(self.username, self.password))
        print("Request " + response.url)
        if response.status_code == 200:  # successful request
            if 'License Terms:' in str(response.content):
                # only the license terms are returned, hash and append them in a
                # subsequent request.
                # See also
                # https://github.com/movebank/movebank-api-doc/blob/master/movebank
                # api.md#read-and-accept-license-terms-using-curl
                print("Has license terms")
                hash = hashlib.md5(response.content).hexdigest()
                params = params + (('license-md5', hash), )
                # also attach previous cookie:
                response = requests.get(
                    'https://www.movebank.org/movebank/service/direct-read',
                    params=params,
                    cookies=response.cookies,
                    auth=(os.environ['mbus'], os.environ['mbpw']))
                if response.status_code == 403:  # incorrect hash
                    print("Incorrect hash")
                    return ''
            return response.content.decode('utf-8')
        print(str(response.content))
        return str(response.content)

    def getStudies(self):
        studies = self.callMovebankAPI(
            (('entity_type', 'study'), ('i_can_see_data', 'true'),
             ('there_are_data_which_i_cannot_see', 'false')))
        if len(studies) > 0:
            # parse raw text to dicts
            studies = csv.DictReader(io.StringIO(studies), delimiter=',')
            all_studies = [
                s for s in studies if s['i_can_see_data'] == 'true'
                and s['there_are_data_which_i_cannot_see'] == 'false'
            ]
            return all_studies

    @staticmethod
    def getStudiesBySensor(studies, sensorname='GPS'):
        return [s for s in studies if sensorname in s['sensor_type_ids']]

    def getIndividualsByStudy(self):
        individuals = self.callMovebankAPI(
            (('entity_type', 'individual'), ('study_id', self.study_id)))
        if len(individuals) > 0:
            return list(csv.DictReader(io.StringIO(individuals),
                                       delimiter=','))

    def getIndividualEvents(self,
                            individual_id,
                            sensor_type_id=653,
                            transform=False):
        """
        SENSORS
        ===============================================================================
        description,external_id,id,is_location_sensor,name
        "","bird-ring",397,true,"Bird Ring"
        "","gps",653,true,"GPS"
        "","radio-transmitter",673,true,"Radio Transmitter"
        "","argos-doppler-shift",82798,true,"Argos Doppler Shift"
        "","natural-mark",2365682,true,"Natural Mark"
        "","acceleration",2365683,false,"Acceleration"
        "","solar-geolocator",3886361,true,"Solar Geolocator"
        "","accessory-measurements",7842954,false,"Accessory Measurements"
        "","solar-geolocator-raw",9301403,false,"Solar Geolocator Raw"
        "","barometer",77740391,false,"Barometer"
        "","magnetometer",77740402,false,"Magnetometer"
        "","orientation",819073350,false,"Orientation"
        "","solar-geolocator-twilight",914097241,false,"Solar Geolocator Twilight"
        """
        params = (('entity_type', 'event'), ('study_id', self.study_id),
                  ('individual_id', individual_id),
                  ('sensor_type_id', sensor_type_id), ('attributes', 'all'))
        events_ = self.callMovebankAPI(params)
        if events_:
            events = list(csv.DictReader(io.StringIO(events_), delimiter=','))
            if sensor_type_id == 653 and transform:
                return self.transformRawGPS(events)
            elif sensor_type_id == 2365683 and transform:
                return self.transformRawACC(events)
            else:
                return events

    @staticmethod
    def transformRawGPS(gpsevents):
        # Returns a list of (ts, deployment_id, lat, long) tuples

        def transform(e):  # dimension reduction and data type conversion
            try:
                if len(e['location_lat']) > 0:
                    e['location_lat'] = float(e['location_lat'])
                if len(e['location_long']) > 0:
                    e['location_long'] = float(e['location_long'])
            except:
                print("Could not parse long/lat.")
            return e['timestamp'], e['deployment_id'], e['location_lat'], e[
                'location_long']

        return [transform(e) for e in gpsevents]

    @staticmethod
    def transformRawACC(accevents, unit='m/s2', sensitivity='high'):
        """
        Transforms raw tri-axial acceleration from X Y Z X Y X Y Z to
        [(ts_interpol, deployment, X', Y', Z'),...] X', Y', Z' are in m/s^2 or g.
        Assumes e-obs acceleration sensors.
        Acknowledgments to Anne K. Scharf and her great moveACC package,
        see https://gitlab.com/anneks/moveACC
        """

        ts_format = '%Y-%m-%d %H:%M:%S.%f'
        out = []

        if unit == 'g':
            unitfactor = 1
        else:
            unitfactor = 9.81

        tag_local_identifier = int(accevents[0]['tag_local_identifier'])
        slope = 0.001  # e-obs 1st generation, high sensitivity

        if tag_local_identifier <= 2241:
            if sensitivity == 'low':
                slope = 0.0027
        elif 2242 <= tag_local_identifier <= 4117:  # e-obs 2nd generation
            slope = 0.0022
        else:
            slope = 1 / 512

        for event in accevents:
            deploym = event['deployment_id']
            seconds = 1 / float(
                event['acceleration_sampling_frequency_per_axis'])
            parsedts = datetime.strptime(event['timestamp'],
                                         ts_format)  # start timestamp
            raw = list(map(int, event['accelerations_raw'].split()))

            #  derive in-between timestamps:
            ts = [
                parsedts + timedelta(seconds=seconds * x)
                for x in range(0, int(len(raw) / 3))
            ]

            #  transform XYZ list to list of (ts, deployment, x, y, z) tuples
            it = iter(raw)
            transformed = [(a.strftime(ts_format), deploym,
                            (b[0] - 2048) * slope * unitfactor,
                            (b[1] - 2048) * slope * unitfactor,
                            (b[2] - 2048) * slope * unitfactor)
                           for (a, b) in list(zip(ts, list(zip(it, it, it))))]
            out.append(transformed)
        return out

    @staticmethod
    def _pprint(list_):
        print(json.dumps(list_, indent=4))

    @staticmethod
    def to_pandas(list_, sensor_type=None, save_to=None):
        if sensor_type and sensor_type.lower() == 'acc':
            arr = np.array(list_)
            m, n, r = arr.shape
            out_arr = np.column_stack((np.repeat(np.arange(m),
                                                 n), arr.reshape(m * n, -1)))
            df = pd.DataFrame(
                out_arr,
                columns=[
                    'idx', 'timestamp', 'deployment_id', 'AccX', 'AccY', 'AccZ'
                ],
            )
            df.drop(columns=['idx'], inplace=True)
            df = df.astype({
                'timestamp': 'datetime64[ns]',
                'deployment_id': 'int32',
                'AccX': 'float32',
                'AccY': 'float32',
                'AccZ': 'float32'
            })
        elif sensor_type and sensor_type.lower() == 'gps':
            df = pd.DataFrame(list_,
                              columns=[
                                  'timestamp', 'deployment_id', 'location_lat',
                                  'location_long'
                              ])
            df = df.astype({
                'timestamp': 'datetime64[ns]',
                'deployment_id': 'int32',
                'location_lat': 'float32',
                'location_long': 'float32'
            })
        else:
            df = pd.DataFrame(list_)
        if save_to:
            df.to_csv(save_to, index=False)
        return df
