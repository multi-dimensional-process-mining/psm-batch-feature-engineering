import pandas as pd
import numpy as np
import time
import os

######## USER INPUT ###########
log_name = 'bpic2020_DD'
use_partition = True
pred_distance = True
nr_partitions = 6
forecast_method = 'exp_smoothing'
alpha_w_min = 0.006
alpha_BI = 0.0001

log_file = "C:\\Users\\s111402\\OneDrive - TU Eindhoven\\1_Research\\Code" \
           "\\time-prediction-benchmark-master\\experiments\\logdata\\%s.csv" % log_name
log_out_path = "C:\\Users\\s111402\\OneDrive - TU Eindhoven\\1_Research\\Code\\time-prediction-benchmark-master\\" \
               "experiments\\logdata\\inter-case-logs"
forecast_batch_params_file = 'C:\\Users\\s111402\\OneDrive - TU Eindhoven\\1_Research\\Code\\' \
                             'time-prediction-benchmark-master\\inter_case\\2_inter_case_feature_creation\\' \
                             'TM_batch_params\\forecast_batch_params_%s_%s_%s_%s.csv' \
                             % (log_name, forecast_method, alpha_BI, alpha_w_min)
start_segment = 'Request Payment'
################################

if pred_distance:
    if use_partition:
        method_name = 'partition_distance_%s' % nr_partitions
    else:
        method_name = 'pred_time_distance'
else:
    method_name = 'actual_time_distance'

timestamp_col = "Complete Timestamp"
case_id_col = "Case ID"
activity_col = 'Activity'

# Measuring performance
nr_features = 0
nr_cases = 0
start_time = time.time()


def read_log(log_file):
    data = pd.read_csv(log_file, sep=";")
    data[timestamp_col] = pd.to_datetime(data[timestamp_col], yearfirst=True)
    data['distance'] = np.nan
    return data


def split_data(data, train_ratio):
    # split into train and test using temporal split
    grouped = data.groupby(case_id_col)
    start_timestamps = grouped[timestamp_col].min().reset_index()
    start_timestamps = start_timestamps.sort_values(timestamp_col, ascending=True, kind='mergesort')
    train_ids = list(start_timestamps[case_id_col])[:int(train_ratio * len(start_timestamps))]
    # print(train_ids)
    train = data[data[case_id_col].isin(train_ids)].sort_values(timestamp_col, ascending=True,
                                                                     kind='mergesort')
    test = data[~data[case_id_col].isin(train_ids)].sort_values(timestamp_col, ascending=True,
                                                                     kind='mergesort')
    # print(test.columns)
    return train, test


def add_feature(group, activity):
    global nr_cases, nr_features
    nr_cases = nr_cases + 1
    relevant_activity_idxs = np.where(group[activity_col] == activity)[0]
    if len(relevant_activity_idxs) > 0:
        nr_features = nr_features + 1
        idx = relevant_activity_idxs[0]
        current_ts = group.iloc[idx, group.columns.get_loc(timestamp_col)]
        if pred_distance:
            if use_partition:
                group.iloc[idx, group.columns.get_loc('distance')] = get_predicted_batch_partition(
                    current_ts=current_ts, partitions=nr_partitions)
            else:
                group.iloc[idx, group.columns.get_loc('distance')] = get_predicted_time_to_next_batch(
                    current_ts=current_ts).total_seconds() / 3600
        else:
            actual_time_distance = group.iloc[idx + 1, group.columns.get_loc(timestamp_col)] - \
                                   group.iloc[idx, group.columns.get_loc(timestamp_col)]
            group.iloc[idx, group.columns.get_loc('distance')] = actual_time_distance.total_seconds() / 3600
    return group


def get_predicted_batch_partition(current_ts, partitions):
    fraction = 1.0 / partitions
    previous_batch = batch_params[batch_params.BM < current_ts].tail(1).squeeze()
    current_batch = batch_params[batch_params.BM > current_ts].head(1).squeeze()
    time_since_prev_batch = current_ts - previous_batch.BM + previous_batch.w_min
    # f_projected_BI = current_batch.f_BI - current_batch.f_w_min + previous_batch.w_min
    f_projected_BI = current_batch.f_BI
    batch_partition = 1
    current_ts_relative_to_BI = time_since_prev_batch / f_projected_BI
    for fraction_nr in range((2 * partitions) - 1):
        start_partition = fraction * fraction_nr
        end_partition = fraction * (fraction_nr + 1)
        if start_partition <= current_ts_relative_to_BI < end_partition:
            if fraction_nr + 1 > partitions:
                batch_partition = 2 * partitions - fraction_nr
            else:
                batch_partition = partitions - fraction_nr
    return batch_partition


def get_actual_batch_partition(current_ts, partitions):
    fraction = 1.0 / partitions
    previous_batch = batch_params[batch_params.BM < current_ts].tail(1).squeeze()
    current_batch = batch_params[batch_params.BM > current_ts].head(1).squeeze()
    time_since_prev_batch = current_ts - previous_batch.BM + previous_batch.w_min
    projected_BI = current_batch.BI - current_batch.w_min + previous_batch.w_min
    batch_partition = 1
    current_ts_relative_to_BI = time_since_prev_batch / projected_BI
    for fraction_nr in range((2 * partitions) - 1):
        start_partition = fraction * fraction_nr
        end_partition = fraction * (fraction_nr + 1)
        if start_partition <= current_ts_relative_to_BI < end_partition:
            if fraction_nr + 1 > partitions:
                batch_partition = 2 * partitions - fraction_nr
            else:
                batch_partition = partitions - fraction_nr
    return batch_partition


def get_predicted_time_to_next_batch(current_ts):
    previous_batch = batch_params[batch_params.BM < current_ts].tail(1).squeeze()
    current_batch = batch_params[batch_params.BM > current_ts].head(1).squeeze()
    # print('TS: ', current_ts, ' ', type(current_ts))
    # print('w_min: ', previous_batch.w_min, ' ', type(previous_batch.w_min))
    # print('f_w_min: ', current_batch.f_w_min, ' ', type(current_batch.f_w_min))
    # print('f_BI: ', current_batch.f_BI, ' ', type(current_batch.f_BI))
    # print('-------------------------------------------')
    if current_ts <= previous_batch.BM + current_batch.f_BI - current_batch.f_w_min:
        time_to_next_BM = previous_batch.BM + current_batch.f_BI - current_ts
    else:
        time_to_next_BM = previous_batch.BM + 2*current_batch.f_BI - current_ts
        if current_ts > previous_batch.BM + 2*current_batch.f_BI - current_batch.f_w_min:
            time_to_next_BM = previous_batch.BM + 3*current_batch.f_BI - current_ts
    return time_to_next_BM


# IMPORT EVENT LOG
print('Importing event log...')
log = read_log(log_file)

# IMPORT ACTUAL AND FORECAST BATCH PARAMETERS
print('Importing actual and forecast batch parameters...')
batch_params = pd.read_csv(forecast_batch_params_file)
batch_params.BM = pd.to_datetime(batch_params.BM, yearfirst=True)
batch_params.BI = pd.to_timedelta(batch_params.BI)
batch_params.f_BI = pd.to_timedelta(batch_params.f_BI)
batch_params.f_w_min = pd.to_timedelta(batch_params.f_w_min)
batch_params.w_min = pd.to_timedelta(batch_params.w_min)

# ADD FEATURE TO LOG (for all cases that have "Request Payment" activity)
print('Add %s as feature to log...' % method_name)
log = log.sort_values([timestamp_col], ascending=True, kind='mergesort')
log = log.groupby(case_id_col).apply(add_feature, start_segment)

# DELETE ALL LAST EVENTS
# log = log[log.remtime != 0]

log_out_file = os.path.join(log_out_path, log_name, "%s_%s.csv" % (log_name, method_name))
log.to_csv(log_out_file, sep=';', header=True, index=False)

elapsed_time = time.time() - start_time
print("Total elapsed time: ", elapsed_time)
print("Average time per feature: ", elapsed_time/nr_features)
print("Average time per case: ", elapsed_time/nr_cases)
