import pandas as pd
import numpy as np

## Loading mean predictions and instantiating dataframe
Olympic_predictions = pd.read_csv('../Olympic_Predictions.csv')
Results_Frame = pd.DataFrame(columns=['Name', 'Gender'] + list(range(10000)))
Results_Frame['Name'] = Olympic_predictions['Athlete']
Results_Frame['Gender'] = Olympic_predictions['Gender']
Results_Frame.set_index(['Name'], inplace=True)

#running sim
for i in Results_Frame.index.values:
    print(i)
    mean = Olympic_predictions[Olympic_predictions['Athlete'] == i]['Prediction'].values[0]
    Results_Frame.loc[i, range(10000)] = np.random.normal(mean, 2.48, 10000).round(2)

Placement_Frame = Results_Frame.groupby('Gender').rank(ascending=True)

Placement_Frame[['Gold', 'Silver', 'Bronze']] = 0
Placement_Frame['Gender'] = Results_Frame['Gender']
for i in Results_Frame.index.values:
    print(i)
    try:
        # value counts will use the .5 for rank when there are ties, this formula makes sure everything adds up appropriately
        Placement_Frame.loc[i, 'Gold'] = (Placement_Frame.loc[i, range(10000)].value_counts()[1] + .5 * \
                                           Placement_Frame.loc[i, range(10000)].value_counts()[1.5]) / 10000
    except KeyError:
        try:
            Placement_Frame.loc[i, 'Gold'] = Placement_Frame.loc[i, range(10000)].value_counts()[1] / 10000
        except KeyError:
            Placement_Frame.loc[i, 'Gold'] = 0
    try:
        Placement_Frame.loc[i, 'Silver'] = (Placement_Frame.loc[i, range(10000)].value_counts()[2] + .5 * \
                                           Placement_Frame.loc[i, range(10000)].value_counts()[2.5]) / 10000
    except KeyError:
        try:
            Placement_Frame.loc[i, 'Silver'] = Placement_Frame.loc[i, range(10000)].value_counts()[2] / 10000
        except KeyError:
            Placement_Frame.loc[i, 'Silver'] = 0
    try:
        Placement_Frame.loc[i, 'Bronze'] = Placement_Frame.loc[i, range(10000)].value_counts()[3] / 10000
    except KeyError:
        Placement_Frame.loc[i, 'Bronze'] = 0


Placement_Frame = Placement_Frame[['Gender', 'Gold', 'Silver', 'Bronze']]
Placement_Frame.to_csv('../Placement_Frame.csv')
