import os
import re
import json
from datetime import datetime, timedelta
import time
import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d
import logging
logging.basicConfig(level=logging.DEBUG)

def parse_time_features(timestamp):
    '''
        Parse higher-dimensional time information from `timestamp`
    '''
    # Year, Month, Day of Month, Day of Week, Hour, Minutes, Seconds
    return [int(time.mktime(timestamp.timetuple())), timestamp.year, timestamp.month, timestamp.day, 
            timestamp.weekday(), timestamp.hour, timestamp.minute, timestamp.second]

def parse_timestamp(date):
    '''
        Parse timestamp and higher-dimensional time information from `email` containing corpus of emails
    '''
    # parse timestamp 
    timestamp = datetime.strptime(date[0:19], '%Y-%m-%dT%H:%M:%S')
    if date[19]=='+':
        timestamp-=timedelta(hours=int(date[20:22]), minutes = int(date[23:]))
    elif date[19]=='-':
        timestamp+=timedelta(hours=int(date[20:22]), minutes = int(date[23:]))
    return parse_time_features(timestamp)

def parse_email(email, filename, index, length):
    '''
        Parse timestamp, and higher-dimensional time information from `email` 
    '''
    logging.debug(f'Parsing email {index} of {length} from file: {filename}')
    return [parse_timestamp(email['date'])[0], filename]

def events_to_rates(event_times, filter_bandwidth=1, num_bins=60, min_time = None, max_time=None, density = True):
    """ convert list of event times into rate function with a discrete time bin_size of 1/rates_per_unit.
    Uses a guassian filter over the empirical rate (histogram count / bin_size) """

    if len(event_times) == 0:  # if event times is an empty list or array
        logging.debug("empty event_times list/array")
        return np.zeros(num_bins), np.zeros(num_bins)

    if not max_time:
        max_time = max(event_times)
    if not min_time:
        min_time = min(event_times)
    bins = np.linspace(min_time, max_time, num=num_bins + 1)
    rate_times = (bins[1:] + bins[:-1]) / 2

    bin_size = (max_time - min_time) / num_bins
    if density:
        counts = np.array(np.histogram(event_times, bins=bins)[0])
        sampled_rates = counts / sum(counts)
    else:
        counts = np.array(np.histogram(event_times, bins=bins)[0])
        sampled_rates = counts / bin_size
    rate_vals = gaussian_filter1d(sampled_rates, filter_bandwidth, mode="nearest")
    return rate_vals, rate_times

# walk through datapath and parse all jsonl files
datapath = 'historical_dry_run_data'
logging.debug(f'Parsing emails from {datapath} raw json files...\n')
all_emails = []
for path, _, files in os.walk(datapath):
    for file in files:
        if re.match(".*.jsonl$", file):
            fullpath = os.path.join(path, file)
            with open(fullpath) as data_file:
                emails = data_file.readlines()
                parsed_emails = [parse_email(json.loads(email), file, index, len(emails)) for index, email in enumerate(emails)]
                all_emails.extend(parsed_emails)

time_features = pd.DataFrame(all_emails, columns = ['Timestamp', 'file'])
#time_features = pd.DataFrame(all_emails, columns = ['Timestamp','Year', 'Month', 'Day of Month', 'Day of Week', 'Hour', 'Minute', 'Seconds', 'file'])

# convert timestamps to rates
logging.debug(f'Converting timestamps to rate functions...\n')
days = 5
hours = 24
num_bins = 60
start_date = "2019-03-04"
start_time = datetime.strptime(start_date, '%Y-%m-%d')
min_time = int(time.mktime(start_time.timetuple()))
series_size = 60 * 60 * 4
window = 60
max_time = min_time + days * hours * 3600

# create time series (1 series / hr --> 120 total series)
times = time_features['Timestamp'].values.astype(int)
series_values = []
series_times = []
series_labels = []
t = min_time
while ((t + series_size) < max_time):
    events = times[(t <= times) & (times < (t + series_size))]
    rate_vals, rate_times = events_to_rates(times, 
                                            num_bins = num_bins, 
                                             filter_bandwidth = 1, 
                                            density = False,
                                            min_time = t,
                                            max_time = t + series_size)
    series_values.append(rate_vals)
    series_times.append(rate_times)
    label = 0
    series_labels.append(label)
    logging.debug(f"Added time series with {len(events)} events and label {label}")
    t += window

logging.debug(f'A total of {len(series_labels)} time series were created')
labels = np.array(series_labels)

# generate high-dimensional time features from series_times
logging.debug(f'Generating high-dimensional time features from rate times...\n')
t_features = [[parse_time_features(datetime.fromtimestamp(t))[1:] for t in s] for s in series_times]
X_data = [t_feat.append(vals) for t_feat, vals in zip(t_features, series_values)]
X_data = np.vstack(t_features)

assert len(labels) == X_data.shape[0], f"The number of labels is {len(labels)}, but the number of records is {X_data.shape[0]}"

# splitting data into training and testing 
train_split = 0
logging.debug(f'Splitting data into training and testing sets...')
'''if not os.path.isfile(datapath + "/prepared/train_X.npy"):
    os.mkdir(datapath + "/prepared")
'''
np.save(datapath + "/prepared/test_X_4_hrs.npy", X_data)
np.save(datapath + "/prepared/test_y_4_hrs.npy", labels)

logging.debug(f'Test set: March 8 contains {len(labels) - train_split} series')
