import pandas as pd
import numpy as np
import os

##Loading in data and joining splits data with elite level competitions
results = pd.read_csv('../Competition_Results.csv')
splits = pd.read_csv('../Athlete_Tables/Split_times.csv')
splits['key'] = splits['Competition ID'].astype(str) + '-' + splits['Athlete ID'].astype(str)
splits = splits[['200m Split', '600m Split', '1000m Split', 'Time Behind Winner', 'key']]
results['key'] = results['Competition ID'].astype(str) + '-' + results['Athlete ID'].astype(str)
results = pd.merge(left=results, right=splits, on='key', how='left')
results['Elite'] = np.where(results['Competition Level'] == 'Elite', 1, 0)
results['Junior'] = np.where(results['Competition Level'] == 'Juniors', 1, 0)
results['Date'] = pd.to_datetime(results['Date'])
results = results.replace('Cha Min Kyu', 'Cha Min-Kyu')



##Getting DNFs, DNSs, etc. I don't think this is statistcally now that I've done some digging, rather a random aspect of competing
results['DNF'] = np.where(results['Result'].isna(), 1, 0)
DNFs_by_Athlete = results.groupby('Athlete').sum()['DNF']

DNFs = results['Result'].isna()
finishes = results[~DNFs]
DNFS = results[DNFs]

skaters = results['Athlete'].unique()
results.groupby('Athlete').sum()

## Getting final times, only necessary for junior level data
finishes['Minutes'] = finishes['Result'].str.replace(':', '.').str.split('.').str[0].str[1].astype(int)
finishes['Seconds'] = finishes['Result'].str.replace(':', '.').str.split('.').str[1].astype(int)
finishes['Tenths'] = finishes['Result'].str.replace(':', '.').str.split('.').str[2].astype(int)
finishes['Time'] = finishes['Minutes'] * 60 + finishes['Seconds'] + finishes['Tenths'] * .1



##Getting Splits as percentage of overall time
finishes['Start Split %'] = np.where(finishes['200m Split'].isna(), np.nan, finishes['200m Split']/finishes['Time'])
finishes['Middle 400 Split %'] = np.where(finishes['600m Split'].isna(), np.nan,
                                          (finishes['600m Split'] - finishes['200m Split'])/finishes['Time'])
finishes['Final 400 Split %'] = np.where(finishes['600m Split'].isna(), np.nan,
                                         (finishes['Time'] - finishes['600m Split'])/finishes['Time'])
start_corrs = finishes[['Start Split %', 'Middle 400 Split %', 'Final 400 Split %',
                        'Time', 'Athlete Age (Days)']].corr()

print()
## Adding columns so I can broadcast together at the end
DNFS['Minutes'] = np.nan
DNFS['Seconds'] = np.nan
DNFS['Tenths'] = np.nan
DNFS['Time'] = np.nan

DNFS['Start Split %'] = np.nan
DNFS['Middle 400 Split %'] = np.nan
DNFS['Final 400 Split %'] = np.nan

## EDA getting averages, standard deviations, sample sizes to get a better idea of what kind of data we have for
# competitor in 2022 olympics
Athlete_Profiles = finishes[['Athlete', 'Time']].groupby('Athlete').mean()
Athlete_Profiles['Variance'] = finishes[['Athlete', 'Time']].groupby('Athlete').var()['Time']
Athlete_Profiles['Median'] = finishes[['Athlete', 'Time']].groupby('Athlete').median()['Time']
Athlete_Profiles['Finishes'] = finishes[['Athlete', 'Time']].groupby('Athlete').count()['Time']
Athlete_Profiles['sd'] = finishes[['Athlete', 'Time']].groupby('Athlete').std()['Time']
Athlete_Profiles = pd.merge(left=Athlete_Profiles.reset_index(), right=results[['Athlete', 'Athlete ID', 'Country',
                                                                                'Gender']],
                            on='Athlete')
Athlete_Profiles = pd.merge(left=Athlete_Profiles, right=DNFs_by_Athlete.reset_index(),
                            on='Athlete', how='left').drop_duplicates('Athlete')
Athlete_Profiles['Sample Size'] = Athlete_Profiles['Finishes'] + Athlete_Profiles['DNF']
Athlete_Profiles['DNF Rate'] = (Athlete_Profiles['DNF']/Athlete_Profiles['Sample Size'])
Athlete_Profiles['DNF sd'] = np.where(Athlete_Profiles['Sample Size'] == 1, np.nan,
                                      Athlete_Profiles['Sample Size'] * Athlete_Profiles['DNF Rate']
                                      * (1 - Athlete_Profiles['DNF Rate']))
## Split data by level
Athlete_Profiles['Elite Level Sample Size'] = finishes.groupby('Athlete').sum()['Elite'].values
Athlete_Profiles['Elite Level 200m mean'] = finishes.groupby('Athlete').mean()['200m Split'].values
Athlete_Profiles['Elite Level 600m mean'] = finishes.groupby('Athlete').mean()['600m Split'].values
Athlete_Profiles['Elite Level 1000m mean'] = finishes.groupby('Athlete').mean()['1000m Split'].values

Athlete_Profiles['Elite Level 200m median'] = finishes.groupby('Athlete').median()['200m Split'].values
Athlete_Profiles['Elite Level 600m median'] = finishes.groupby('Athlete').median()['600m Split'].values
Athlete_Profiles['Elite Level 1000m median'] = finishes.groupby('Athlete').median()['1000m Split'].values

Athlete_Profiles['Elite Level 200m std'] = finishes.groupby('Athlete').std()['200m Split'].values
Athlete_Profiles['Elite Level 600m std'] = finishes.groupby('Athlete').std()['600m Split'].values
Athlete_Profiles['Elite Level 1000m std'] = finishes.groupby('Athlete').std()['1000m Split'].values


Athlete_Profiles['Elite Level 200m %'] = finishes.groupby('Athlete').mean()['Start Split %'].values
Athlete_Profiles['Elite Level 600m %'] = finishes.groupby('Athlete').mean()['Middle 400 Split %'].values
Athlete_Profiles['Elite Level Finish %'] = finishes.groupby('Athlete').mean()['Final 400 Split %'].values




## Getting dataframe of athletes that competed in the olympics
Olympic_Athletes = results[(results['Competition Name'] == 'Olympic Games') & (results['Competition City'] == 'Beijing')]['Athlete']
Athlete_Profiles['2022 Olympics'] = np.where(np.isin(Athlete_Profiles['Athlete'], Olympic_Athletes), 1, 0)

Athlete_Profiles.round(3).to_csv('Athlete_Tables/Athletes_Tables.csv')
pd.concat([finishes, DNFS]).to_csv('../Athlete_Tables/Augmented_Results.csv')

