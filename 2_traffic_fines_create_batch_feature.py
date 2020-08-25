import pandas as pd
import numpy as np
import time
import os

######## USER INPUT ###########
log_name = 'traffic_fines'
use_partition = True
pred_distance = True
pred_classification = True
nr_partitions = 10
forecast_method = 'exp_smoothing'
alpha_w_min = 0.3
alpha_BI = 0.3
log_in_file = "C:\\Users\\s111402\\OneDrive - TU Eindhoven\\1_Research\\Code" \
           "\\time-prediction-benchmark-master\\experiments\\logdata\\traffic_fines.csv"
log_out_path = "C:\\Users\\s111402\\OneDrive - TU Eindhoven\\1_Research\\Code\\time-prediction-benchmark-master\\" \
               "experiments\\logdata\\inter-case-logs"
forecast_batch_params_file = 'C:\\Users\\s111402\\OneDrive - TU Eindhoven\\1_Research\\Code\\' \
                             'time-prediction-benchmark-master\\inter_case\\2_inter_case_feature_creation\\' \
                             'TM_batch_params\\forecast_batch_params_%s_%s_%s_%s.csv' \
                             % (log_name, forecast_method, alpha_BI, alpha_w_min)
pred_batch_classification_file = 'C:\\Users\\s111402\\OneDrive - TU Eindhoven\\1_Research\\Code\\' \
                                 'time-prediction-benchmark-master\\inter_case\\2_inter_case_feature_creation\\' \
                                 'CM_S\\c_S_traffic_fines_Send for Credit Collection_xgb_prefix_agg_4.csv'
end_segment = 'Send for Credit Collection'
################################

if pred_classification:
    if pred_distance:
        if use_partition:
            method_name = 'pred_class_partition_distance_%s' % nr_partitions
        else:
            method_name = 'pred_class_pred_time_distance'
    else:
        method_name = 'pred_class_actual_time_distance'
else:
    if pred_distance:
        if use_partition:
            method_name = 'actual_class_partition_distance_%s' % nr_partitions
        else:
            method_name = 'actual_class_pred_time_distance'
    else:
        method_name = 'actual_class_actual_time_distance'

timestamp_col = "Complete Timestamp"
case_id_col = "Case ID"
activity_col = 'Activity'

# Measuring performance
nr_features = 0
nr_cases = 0
start_time = time.time()


def read_log(log_in_file):
    data = pd.read_csv(log_in_file, sep=";")
    data.drop(columns=['elapsed'], inplace=True)
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


def extract_event_nr(group):
    group = group.sort_values(timestamp_col, ascending=True, kind='mergesort')
    group["nr_events"] = range(1, len(group) + 1)
    return group


def add_feature_predicted_class(group):
    global nr_cases, nr_features
    nr_cases = nr_cases + 1
    group = group.sort_values(timestamp_col, ascending=True, kind='mergesort')
    if len(group) > 4:
        if group.iloc[3, group.columns.get_loc('predicted_binary')] == 1:
            nr_features = nr_features + 1
            current_ts = group.iloc[3, group.columns.get_loc(timestamp_col)]
            # print('Case ID: ', group.iloc[3, group.columns.get_loc(case_id_col)])
            if pred_distance:
                if use_partition:
                    group.iloc[3, group.columns.get_loc('distance')] = get_predicted_batch_partition(
                        current_ts=current_ts, partitions=nr_partitions)
                else:
                    group.iloc[3, group.columns.get_loc('distance')] = \
                        get_predicted_time_to_next_batch(current_ts=current_ts).total_seconds() / 60
            else:
                actual_time_distance = group.iloc[4, group.columns.get_loc(timestamp_col)] - \
                                       group.iloc[3, group.columns.get_loc(timestamp_col)]
                group.iloc[3, group.columns.get_loc('distance')] = actual_time_distance.total_seconds() / 60
    return group


def add_feature_actual_class(group, activity):
    global nr_cases, nr_features
    nr_cases = nr_cases + 1
    relevant_activity_idxs = np.where(group[activity_col] == activity)[0]
    if len(relevant_activity_idxs) > 0:
        nr_features = nr_features + 1
        idx = relevant_activity_idxs[0]
        current_ts = group.iloc[idx - 1, group.columns.get_loc(timestamp_col)]
        if pred_distance:
            if use_partition:
                group.iloc[idx - 1, group.columns.get_loc('distance')] = get_predicted_batch_partition(
                    current_ts=current_ts, partitions=nr_partitions)
            else:
                group.iloc[idx - 1, group.columns.get_loc('distance')] = get_predicted_time_to_next_batch(
                    current_ts=current_ts).total_seconds() / 60
        else:
            actual_time_distance = group.iloc[idx, group.columns.get_loc(timestamp_col)] - \
                                   group.iloc[idx - 1, group.columns.get_loc(timestamp_col)]
            group.iloc[idx - 1, group.columns.get_loc('distance')] = actual_time_distance.total_seconds() / 60
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
log = read_log(log_in_file)

# IMPORT ACTUAL AND FORECAST BATCH PARAMETERS
print('Importing actual and forecast batch parameters...')
batch_params = pd.read_csv(forecast_batch_params_file)
batch_params.BM = pd.to_datetime(batch_params.BM, yearfirst=True)
batch_params.BI = pd.to_timedelta(batch_params.BI)
batch_params.f_BI = pd.to_timedelta(batch_params.f_BI)
batch_params.f_w_min = pd.to_timedelta(batch_params.f_w_min)
batch_params.w_min = pd.to_timedelta(batch_params.w_min)

if pred_classification:
    # ADD FEATURE TO LOG (for cases predicted to be in a batch at Send for Credit Collection activity)
    print('Add %s as feature to log...' % method_name)
    log = log.groupby(case_id_col).apply(extract_event_nr)
    log.reset_index(drop=True, inplace=True)

    classification = pd.read_csv(pred_batch_classification_file)
    log = pd.merge(log, classification, on=['Case ID', 'nr_events'], how='left')
    log.fillna(0, inplace=True)

    log = log.sort_values([timestamp_col], ascending=True, kind='mergesort')
    log = log.groupby(case_id_col).apply(add_feature_predicted_class)
    log.drop(columns=['predicted_binary'], inplace=True)
    log.drop(columns=['nr_events'], inplace=True)
else:
    # ADD FEATURE TO LOG (for cases that will be in a batch, i.e. with Send for Credit Collection as next activity)
    print('Add %s as feature to log...' % method_name)
    log = log.groupby(case_id_col).apply(add_feature_actual_class, end_segment)

# # DELETE ALL LAST EVENTS
# log = log[log.remtime != 0]

log_out_file = os.path.join(log_out_path, log_name, "%s_%s.csv" % (log_name, method_name))
log.to_csv(log_out_file, sep=';', header=True, index=False)

elapsed_time = time.time() - start_time
print("Total elapsed time: ", elapsed_time)
print("Average time per feature: ", elapsed_time/nr_features)
print("Average time per case: ", elapsed_time/nr_cases)
