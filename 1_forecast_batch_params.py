import pandas as pd
import datetime
from datetime import timedelta
import time
import os

start_time = time.time()

######## USER INPUT ###########
log_name = 'traffic_fines'
# log_name = 'bpic2020_DD'
batch_params_file = 'C:\\Users\\s111402\\OneDrive - TU Eindhoven\\1_Research\\Code\\' \
                    'time-prediction-benchmark-master\\inter_case\\2_inter_case_feature_creation\\TM_batch_params\\' \
                    'batch_params_%s.csv' % log_name
forecast_batch_params_path = 'C:\\Users\\s111402\\OneDrive - TU Eindhoven\\1_Research\\Code\\' \
                             'time-prediction-benchmark-master\\inter_case\\2_inter_case_feature_creation\\' \
                             'TM_batch_params'
################################

forecasting_method = 'exp_smoothing'
batch_params_import = pd.read_csv(batch_params_file, usecols=['batch', 'BM', 'w_min'])
batch_params = batch_params_import.copy()
batch_params['BM'] = pd.to_datetime(batch_params['BM'], dayfirst=False, yearfirst=False)

BI = pd.Series([], name='BI', dtype=float)
f_BI = pd.Series([], name='f_BI')
f_w_min = pd.Series([], name='f_w_min')
batch_params = pd.concat([batch_params, BI, f_w_min, f_BI], axis=1, sort=False)

batch_params.w_min = pd.to_timedelta(batch_params.w_min, unit='h')

for index, row in batch_params.iterrows():
    if index < len(batch_params) - 1:
        batch_params.at[index, 'BI'] = pd.to_timedelta(batch_params.BM[index + 1] - batch_params.BM[index])


batch_params.loc[-1] = [0, batch_params.BM[0] - batch_params.BI[0], batch_params.w_min[0], batch_params.BI[0],
                        batch_params.w_min[0], batch_params.BI[0]]
batch_params.loc[-2] = [0, batch_params.BM[0] - 2 * batch_params.BI[0], batch_params.w_min[0], batch_params.BI[0],
                        batch_params.w_min[0], batch_params.BI[0]]
batch_params.index = batch_params.index + 2
batch_params = batch_params.sort_index()
batch_params.loc[len(batch_params)] = [len(batch_params) - 1, datetime.datetime(2020, 5, 17), pd.to_timedelta(0),
                                       pd.to_timedelta(1), batch_params.f_BI[len(batch_params) - 1],
                                       batch_params.f_w_min[len(batch_params) - 1]]

final_mae_BI = timedelta(days=100)
final_mae_w_min = timedelta(days=100)

list_alpha_w_min = []
list_alpha_BI = []
list_mae_w_min = []
list_mae_BI = []

for alpha_BI in range(1, 10, 1):
    alpha_BI = alpha_BI / 10000

    for index, row in batch_params.iterrows():
        if index > 1:
            batch_params.at[index, 'f_BI'] = alpha_BI * batch_params.BI[index - 1] + \
                                             (1 - alpha_BI) * batch_params.f_BI[index - 1]

    batch_params['abs_error_BI'] = abs(batch_params.BI - batch_params.f_BI)
    mae_BI = batch_params['abs_error_BI'].mean()
    print('MAE BI: ', mae_BI, ', alpha: ', alpha_BI)

    list_mae_BI.append(mae_BI)
    list_alpha_BI.append(alpha_BI)

    if mae_BI < final_mae_BI:
        final_mae_BI = mae_BI

for alpha_w_min in range(1, 10, 1):
    alpha_w_min = alpha_w_min / 1000

    for index, row in batch_params.iterrows():
        if index > 1:
            batch_params.at[index, 'f_w_min'] = alpha_w_min * batch_params.w_min[index - 1] + \
                                                (1 - alpha_w_min) * batch_params.f_w_min[index - 1]

    batch_params['abs_error_w_min'] = abs(batch_params.w_min - batch_params.f_w_min)
    mae_w_min = batch_params['abs_error_w_min'].mean()
    print('MAE w_min: ', mae_w_min, ', alpha: ', alpha_w_min)

    list_mae_w_min.append(mae_w_min)
    list_alpha_w_min.append(alpha_w_min)

    if mae_w_min < final_mae_w_min:
        final_mae_w_min = mae_w_min

final_alpha_w_min = list_alpha_w_min[list_mae_w_min.index(final_mae_w_min)]
final_alpha_BI = list_alpha_BI[list_mae_BI.index(final_mae_BI)]

print('Final MAE w_min: ', final_mae_w_min, ', alpha: ', final_alpha_w_min)
print('Final MAE BI: ', final_mae_BI, ', alpha: ', final_alpha_BI)

for index, row in batch_params.iterrows():
    if index > 1:
        batch_params.at[index, 'f_BI'] = final_alpha_BI * batch_params.BI[index - 1] + \
                                         (1 - final_alpha_BI) * batch_params.f_BI[index - 1]
        batch_params.at[index, 'f_w_min'] = final_alpha_w_min * batch_params.w_min[index - 1] + \
                                            (1 - final_alpha_w_min) * batch_params.f_w_min[index - 1]

batch_params.BI = batch_params.BI.shift(1)
batch_params.f_BI = batch_params.f_BI.shift(1)

forecast_batch_params_file = os.path.join(forecast_batch_params_path, 'forecast_batch_params_%s_%s_%s_%s.csv'
                                          % (log_name, forecasting_method, final_alpha_BI, final_alpha_w_min))

batch_params.to_csv(forecast_batch_params_file, header=True, index=False,
                    columns=["batch", "BM", "w_min", "BI", "f_w_min", "f_BI"])

elapsed_time = time.time() - start_time
print("Total elapsed time: ", elapsed_time)
