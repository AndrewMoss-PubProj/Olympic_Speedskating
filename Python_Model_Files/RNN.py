import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle



model_frame = pd.read_csv('../input_frame.csv').drop(columns='Unnamed: 0')
Olympic_games_2022 = model_frame[((model_frame['Competition Name'] == 'Olympic Games') |
                                  (model_frame['Competition Name'] == 'Heerenveen')) &
                                 (pd.to_datetime(model_frame['Date']).dt.year == 2022)]
model_frame = model_frame.loc[~np.isin(model_frame.index, Olympic_games_2022.index),
                                ['Date', 'Athlete', 'Athlete Age (Days)', 'EMA_.1_mean', 'EMA_.1_start_mean', 'EMA_.5_mean',
                                'sma_731D_start_time_mean', 'sma_731_D_time_mean', 'Gender', 'Time']]
Olympic_games_2022 = Olympic_games_2022[Olympic_games_2022['Competition Name'] == 'Olympic Games']



## Women 1, Men 0
model_frame['Gender'] = pd.get_dummies(model_frame['Gender'], drop_first=True)
model_frame['Days_since_last'] = pd.to_datetime(model_frame['Date']).diff().dt.days.fillna(-1)

inputs = model_frame[['Days_since_last', 'Athlete', 'Athlete Age (Days)', 'Time', 'Gender']]
print()
