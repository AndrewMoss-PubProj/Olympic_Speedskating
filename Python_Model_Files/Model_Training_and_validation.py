import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor
from sklearn import model_selection
from sklearn import linear_model
import statsmodels.api as sm


## assigns roughly equal sized groups and makes sure all trials involving same athlete are in the same group
def assign_groups(model_frame, num_groups, seed):
    cutoff_point = len(model_frame) / num_groups
    cutoff_point_temp = cutoff_point
    group_frame = model_frame.groupby('Athlete').count()['Athlete Age (Days)'].\
        reset_index().sample(frac=1, random_state=seed).reset_index(drop=True)
    group_frame['group'] = -1
    group_number = 0
    for index, row in group_frame.iterrows():
        cutoff_point_temp = cutoff_point_temp - row['Athlete Age (Days)']
        if cutoff_point_temp > 0:
            group_frame.loc[index, 'group'] = group_number
        else:
            group_frame.loc[index, 'group'] = group_number
            group_number += 1
            cutoff_point_temp = cutoff_point
    model_frame = pd.merge(left=model_frame, right=group_frame.drop(columns='Athlete Age (Days)'), on='Athlete', how='left')
    return model_frame
model_frame = pd.read_csv('../input_frame.csv').drop(columns='Unnamed: 0')
Olympic_games_2022 = model_frame[((model_frame['Competition Name'] == 'Olympic Games') |
                                  (model_frame['Competition Name'] == 'Heerenveen')) &
                                 (pd.to_datetime(model_frame['Date']).dt.year == 2022)]
model_frame = model_frame.loc[~np.isin(model_frame.index, Olympic_games_2022.index),
                                ['Athlete', 'Athlete Age (Days)', 'EMA_.1_mean', 'EMA_.1_start_mean', 'EMA_.5_mean',
                                'sma_731D_start_time_mean', 'sma_731_D_time_mean', 'Gender', 'Time']].dropna()
Olympic_games_2022 = Olympic_games_2022[Olympic_games_2022['Competition Name'] == 'Olympic Games']



## Women 1, Men 0
model_frame['Gender'] = pd.get_dummies(model_frame['Gender'], drop_first=True)
group_Kfold = model_selection.GroupKFold(n_splits=5)
model_frame = assign_groups(model_frame, group_Kfold.get_n_splits(), 4)
model_XGB = XGBRegressor()
linear_model = linear_model.LinearRegression()
X = model_frame[['Athlete Age (Days)', 'EMA_.5_mean', 'Gender']]
Y = model_frame['Time']

## Training xgboost model with cross val to look at MSE and MAE numbers to compare to linear model and moving averages
scores = model_selection.cross_val_score(model_XGB, X, Y, scoring='neg_mean_absolute_error',
                                cv=group_Kfold.split(model_frame[['EMA_.5_mean', 'Gender']],
                                                     model_frame['Time'], model_frame['group']), n_jobs=-1)
print('MAE: %.3f (%.3f)' % (scores.mean(), scores.std()))

scores = model_selection.cross_val_score(model_XGB, X, Y, scoring='neg_mean_squared_error',
                                cv=group_Kfold.split(model_frame[['EMA_.5_mean', 'Gender']],
                                                     model_frame['Time'], model_frame['group']), n_jobs=-1)
print('MSE: %.3f (%.3f)' % (scores.mean(), scores.std()))
print()
## Training linear model with cross val to look at MSE and MAE numbers to compare to xgboost model and moving averages


scores_linear = model_selection.cross_val_score(linear_model, X, Y, scoring='neg_mean_absolute_error',
                                cv=group_Kfold.split(pd.concat([model_frame[['EMA_.5_mean', 'Gender']]
                                                                   , model_frame['Athlete Age (Days)'] *
                                                                model_frame['EMA_.5_mean']], axis=1),
                                                     model_frame['Time'], model_frame['group']), n_jobs=-1)
print('MAE: %.3f (%.3f)' % (scores_linear.mean(), scores_linear.std()))

scores_linear = model_selection.cross_val_score(linear_model, X, Y, scoring='neg_mean_squared_error',
                                cv=group_Kfold.split(pd.concat([model_frame[['EMA_.5_mean', 'Gender']]
                                                                   ,model_frame['Athlete Age (Days)'] *
                                                                model_frame['EMA_.5_mean']], axis=1),
                                                     model_frame['Time'], model_frame['group']), n_jobs=-1)
print('MSE: %.3f (%.3f)' % (scores_linear.mean(), scores_linear.std()))

x_with_interaction = pd.concat([model_frame[['EMA_.5_mean', 'Gender']]
                                                                   ,model_frame['Athlete Age (Days)'] *
                                                                model_frame['EMA_.5_mean']], axis=1)
x_with_interaction.rename(columns={0: 'Interaction_term'}, inplace=True)
linear_model.fit(x_with_interaction, Y)
pickle.dump(linear_model, open('../Olympics_linear_model.sav', 'wb'))

## easier to evaluate final model with statsmodels api
x_with_interaction = sm.add_constant(x_with_interaction)
ols = sm.OLS(Y.values, x_with_interaction)
ols_result = ols.fit()
print(ols_result.summary())
print(round(np.sqrt(ols_result.scale),4))

Olympic_games_2022['Interaction_term'] = Olympic_games_2022['Athlete Age (Days)'] * Olympic_games_2022['EMA_.5_mean']
Olympic_games_2022['Gender'] = np.where(Olympic_games_2022['Gender'] == 'Men', 0, 1)
Olympic_games_2022 = Olympic_games_2022[['Athlete', 'EMA_.5_mean', 'Gender', 'Interaction_term']].dropna()
Olympic_games_2022['Prediction'] = linear_model.predict(Olympic_games_2022[['EMA_.5_mean',
                                                                            'Gender', 'Interaction_term']])
Olympic_games_2022.to_csv('../Olympic_Predictions.csv')