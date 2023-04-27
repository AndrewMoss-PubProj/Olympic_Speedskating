import pandas as pd
import numpy as np
import os

Augmented_Results = pd.read_csv('../Athlete_Tables/Augmented_Results.csv')
Augmented_Results['Athlete Year Old'] = Augmented_Results['Athlete Age (Days)'].__floordiv__(365.25)

## Getting age curves for splits
Age_Groups_means = Augmented_Results.groupby('Athlete Year Old').mean()[['200m Split', '600m Split', '1000m Split']]
Age_Groups_lower_quart = Augmented_Results.groupby('Athlete Year Old').quantile(.25)[['200m Split', '600m Split', '1000m Split']]
Age_Groups_medians = Augmented_Results.groupby('Athlete Year Old').median()[['200m Split', '600m Split', '1000m Split']]
Age_Groups_upper_quart = Augmented_Results.groupby('Athlete Year Old').quantile(.75)[['200m Split', '600m Split', '1000m Split']]
Age_Groups_std = Augmented_Results.groupby('Athlete Year Old').std()[['200m Split', '600m Split', '1000m Split']]

Age_Groups_means[['200m Split_25', '600m Split_25', '1000m Split_25']] = Age_Groups_upper_quart
Age_Groups_means[['200m Split_50', '600m Split_50', '1000m Split_50']] = Age_Groups_medians
Age_Groups_means[['200m Split_75', '600m Split_75', '1000m Split_75']] = Age_Groups_lower_quart
Age_Groups_means[['200m Split_std', '600m Split_std', '1000m Split_std']] = Age_Groups_std




Age_Groups_means.to_csv('age_curve_table.csv')

