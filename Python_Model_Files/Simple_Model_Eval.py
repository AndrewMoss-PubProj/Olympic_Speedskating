import pandas as pd
import numpy as np
import glob

error_frame = pd.DataFrame()
for file in glob.glob('../Athlete_Events/**/*.csv'):
    print(file)
    error_frame = pd.concat([error_frame, pd.read_csv(file)])
error_frame.to_csv('../input_frame.csv')
clean_input_frame = error_frame[['Athlete', 'Athlete Age (Days)', 'Time', 'EMA_.5_mean',
                                 'EMA_.5_std', 'EMA_.1_mean', 'EMA_.1_std']]

error_frame = error_frame[['sma_infinite_time_resid', 'sma_731_D_time_resid', 'sma_365D_time_resid',
                           'sma_180D_time_resid', 'EMA_.1_time_resid', 'EMA_.5_time_resid']].dropna()

#Evaluating mean absolute error and mean squared error for moving average models
mean_absolute_errors = abs(error_frame).mean(axis=0)
square_error_frame = np.square(error_frame).mean(axis=0)
mean_absolute_errors_std = abs(error_frame).std(axis=0)
square_error_frame_std = np.square(error_frame).std(axis=0)

eval_frame = pd.concat([mean_absolute_errors, mean_absolute_errors_std, square_error_frame, square_error_frame_std], axis=1)
eval_frame.columns = ['MAE', 'AE std', 'MSE', 'SqE std']
eval_frame.to_csv('../moving_average_eval_table.csv')

print()