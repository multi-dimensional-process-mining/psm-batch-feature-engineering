# batch-feature-engineering

*Feature engineering for inter-case dynamics caused by batching.*

These scripts first take as input batch parameters from the [Batch Miner](https://github.com/multi-dimensional-process-mining/psm-batchmining) in csv format and output forecasts of these batch parameters.
They then take (1) the forecast batch parameters, (2) a predicted batch classification and (3) an event log - all in csv format - and output an event log with additional batch feature.

>**Note:** This directory only contains scripts for creating features for the BPIC'20 log and Road Traffic Fine Management log, but can be applied to other logs with some simple adjustments.

## Prerequisites
- Python 3.7 (compatibility with older versions of Python is not tested and therefore uncertain).
- Installation of pandas and numpy libraries

## User input
The following parameters need to be specified for parameter forecasting (TM_1_forecast_batch_params.py):
* log_name: Name of the event log
* batch_params_import_file: Path to csv file with batch parameters (needs to contain columns 'batch', 'BM' and 'w_min')
* forecast_batch_params_path: Path to desired location of forecast batch parameters

The following parameters need to be specified for feature creation (TM_2_%s_create_batch_feature.py % logname):
* log_name: Name of the event log
* use_partition: Set to True for partition-distance
* pred_distance: Set to True for using the predicted distance (default)
* pred_classification: Set to True for using the predicted batch classification (default)
* nr_partitions: Specify the number of batch partitions when using the partition-distance feature
* alpha_w_min: Specify the optimal smoothing factor for w_min (corresponding to forecast batch parameters)
* alpha_BI: Specify the optimal smoothing factor for BI (corresponding to forecast batch parameters)
* log_in_file: Path to original event log file in csv format
* log_out_path: Path to desired location of new event log with inter-case feature
* forecast_batch_params_file: Path to forecast batch parameters
* pred_batch_classification_file: Path to predicted batch classification (needs to contain columns 'Case ID', 'nr_events' and 'predicted_binary')
