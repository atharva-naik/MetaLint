import numpy as np
import pandas as pd
import os

from feature_utils import *

"""
Outline:
Weekend/weekday marker
Holiday marker
Split time into 24 classes
Location could also be split into classes

Clean data - remove outliers:
check for erroneous data for all types:
Location data (off the island)
date_time 1800s for ex

remove 0 passenger data and > 6. Test set only has 1-6
remove longitude < -75 and > -72
remove latitude < 40 and > 42

Test set attributes:
[-72.990963, -72.986532, 41.696683, 41.709555, 6] final max values
[-74.263242, -74.252193, 40.568973, 40.573143, 1] final min values

Fix saving so don't have to clean dataset everytime. And can load from disk
"""

def clean_dataset(train_df):
#train_df = pd.read_csv('/media/shuza/HDD_Toshiba/Taxi_NYC/train.csv',nrows=100000)
    #train_df = pd.read_csv('/media/shuza/HDD_Toshiba/Taxi_NYC/train.csv')

    #max_list = [max(train_df.dropoff_longitude),max(train_df.pickup_longitude),max(train_df.dropoff_latitude),max(train_df.pickup_latitude),max(train_df.passenger_count)]
    #min_list = [min(train_df.dropoff_longitude),min(train_df.pickup_longitude),min(train_df.dropoff_latitude),min(train_df.pickup_latitude),min(train_df.passenger_count)]
    #print(max_list)
    #print(min_list)

    print('Old size: %d' % len(train_df))
    train_df = train_df[(train_df.dropoff_longitude > -75) & (train_df.dropoff_longitude < -72)]
    train_df = train_df[(train_df.pickup_longitude > -75) & (train_df.pickup_longitude < -72)]
    train_df = train_df[(train_df.dropoff_latitude > 40) & (train_df.dropoff_latitude < 42)]
    train_df = train_df[(train_df.pickup_latitude > 40) & (train_df.pickup_latitude < 42)]
    train_df = train_df[(train_df.passenger_count > 0) & (train_df.passenger_count < 7)]
    print('New size: %d' % len(train_df))

    #add_hour(train_df)
    #add_day(train_df)
    #add_perimeter_distance(train_df)

    max_list = [max(train_df.dropoff_longitude),max(train_df.pickup_longitude),max(train_df.dropoff_latitude),max(train_df.pickup_latitude),max(train_df.passenger_count)]
    min_list = [min(train_df.dropoff_longitude),min(train_df.pickup_longitude),min(train_df.dropoff_latitude),min(train_df.pickup_latitude),min(train_df.passenger_count)]
    print(max_list,'final max values')
    print(min_list,'final min values')

    print('Old size: %d' % len(train_df))
    train_df = train_df.dropna(how='any', axis = 'rows')
    print('New size: %d' % len(train_df))

    cleaned_training_set = train_df
    return cleaned_training_set


#check test set for outliers
def check_test_set():
    test_df = pd.read_csv('/media/shuza/HDD_Toshiba/Taxi_NYC/test.csv')
    max_list = [max(test_df.dropoff_longitude),max(test_df.pickup_longitude),max(test_df.dropoff_latitude),max(test_df.pickup_latitude),max(test_df.passenger_count)]
    min_list = [min(test_df.dropoff_longitude),min(test_df.pickup_longitude),min(test_df.dropoff_latitude),min(test_df.pickup_latitude),min(test_df.passenger_count)]
    print(max_list,'final max values')
    print(min_list,'final min values')
#paths cols = [['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count','hour','day','perimeter_distance']],
#folder = '/media/shuza/HDD_Toshiba/Taxi_NYC/Training_set'

#need to save in chucks due to size. and potentially in multiple files and then append
#cleaned_training_set.to_csv('cleaned_training_set.csv',chunksize=1000,mode='a',index = False)