import pandas as pd
import numpy as np

Athletes = pd.read_csv('../Athlete_Tables/Athletes_Tables.csv').drop(columns='Unnamed: 0')
Athletes = Athletes[(Athletes['Elite Level Sample Size'] >= 5) | (Athletes['2022 Olympics'] == 1)]
Results = pd.read_csv('../Athlete_Tables/Augmented_Results.csv').drop(columns='Unnamed: 0')
Results['Date'] = pd.to_datetime(Results['Date'])
Results.set_index('Date', inplace=True)
Results.sort_index(inplace=True)

for athlete in Athletes['Athlete'].values:
    print(athlete)

    athlete_results = Results[(Results['Athlete'] == athlete)]
    athlete_results['Days Since Last'] = athlete_results['Athlete Age (Days)'].diff()
    ## 1 year simple moving average
    athlete_results['sma_365D_time_mean'] = np.append(np.full(1, np.nan),
                                                           athlete_results['Time'].rolling(
                                                               '365D', min_periods=1).mean().values[0:-1])
    athlete_results['sma_365D_time_resid'] = athlete_results['Time'] - athlete_results['sma_365D_time_mean']
    ## 2 Year Simple moving average
    athlete_results['sma_731_D_time_mean'] = np.append(np.full(1, np.nan),
                                                           athlete_results['Time'].rolling(
                                                               '731D', min_periods=1).mean().values[0:-1])
    athlete_results['sma_731_D_time_resid'] = athlete_results['Time'] - athlete_results['sma_731_D_time_mean']

    athlete_results['sma_infinite_time_mean'] = np.append(np.full(1, np.nan),
                                                           athlete_results['Time'].rolling(
                                                               '100000D', min_periods=1).mean().values[0:-1])
    ## average of all competitions up until the point in time
    athlete_results['sma_infinite_time_resid'] = athlete_results['Time'] - athlete_results['sma_infinite_time_mean']

    ## 180 day simple moving average
    athlete_results['sma_180D_time_mean'] = np.append(np.full(1, np.nan),
                                                           athlete_results['Time'].rolling(
                                                               '180D', min_periods=1).mean().values[0:-1])
    athlete_results['sma_180D_time_resid'] = athlete_results['Time'] - athlete_results['sma_180D_time_mean']



    # exponential weighted moving average with alpha = .5
    athlete_results['EMA_.5_mean'] = np.append(np.full(1, np.nan),
                                                           athlete_results['Time'].ewm(alpha=.5).mean().values[0:-1])
    athlete_results['EMA_.5_std'] = np.append(np.full(1, np.nan),
                                                           athlete_results['Time'].ewm(alpha=.5).std().values[0:-1])
    athlete_results['EMA_.5_time_resid'] = athlete_results['Time'] - athlete_results['EMA_.5_mean']

    # exponential weighted moving average with alpha = .1
    athlete_results['EMA_.1_mean'] = np.append(np.full(1, np.nan),
                                                           athlete_results['Time'].ewm(alpha=.1).mean().values[0:-1])
    athlete_results['EMA_.1_std'] = np.append(np.full(1, np.nan),
                                                           athlete_results['Time'].ewm(alpha=.1).std().values[0:-1])
    athlete_results['EMA_.1_time_resid'] = athlete_results['Time'] - athlete_results['EMA_.1_mean']


    ## using predicted start times instead of finish times to see if that removes noise in model
    # this failed but happy I looked into it
    athlete_results['sma_infinite_start_time_mean'] = np.append(np.full(1, np.nan),
                                                           athlete_results['200m Split'].rolling(
                                                               '100000D', min_periods=1).mean().values[0:-1])

    athlete_results['sma_731D_start_time_mean'] = np.append(np.full(1, np.nan),
                                                           athlete_results['200m Split'].rolling(
                                                               '365D', min_periods=1).mean().values[0:-1])

    athlete_results['sma_365D_start_time_mean'] = np.append(np.full(1, np.nan),
                                                           athlete_results['200m Split'].rolling(
                                                               '365D', min_periods=1).mean().values[0:-1])

    athlete_results['sma_180D_start_time_mean'] = np.append(np.full(1, np.nan),
                                                           athlete_results['200m Split'].rolling(
                                                               '180D', min_periods=1).mean().values[0:-1])


    athlete_results['EMA_.1_start_mean'] = np.append(np.full(1, np.nan),
                                                           athlete_results['200m Split'].ewm(alpha=.1).mean().values[0:-1])
    athlete_results['EMA_.1_start_mean'] = np.append(np.full(1, np.nan),
                                                           athlete_results['200m Split'].ewm(alpha=.1).mean().values[0:-1])


    athlete_results.to_csv('../Athlete_Events/' + athlete_results['Gender'].iloc[0] + '/' + athlete.replace(' ', '_') + '.csv')
