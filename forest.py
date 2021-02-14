# Written by: Nick Gerend, @dataoutsider
# Viz: "Winter Forest", enjoy!

import pandas as pd
import numpy as np
import os
from math import pi, exp, sqrt, log10, sin, cos
import random
import matplotlib.pyplot as plt

#region input
df_area = pd.read_csv(os.path.dirname(__file__) + '/NFD - Area planted by ownership and species group - EN FR.csv', engine = 'python')
df_seeds = pd.read_csv(os.path.dirname(__file__) + '/NFD - Number of seedlings planted by ownership, species group - EN FR.csv', engine = 'python')
df_area = df_area.loc[df_area['Area (hectares)'] > 0]
df_seeds = df_seeds.loc[df_seeds['Number of seedlings'] > 0]
df_combo = new_df = pd.merge(df_seeds, df_area,  how='left', left_on=['Jurisdiction','Species group', 'Year', 'Tenure (En)'], right_on = ['Jurisdiction','Species group', 'Year', 'Tenure (En)'])
df_combo = df_combo[['Year', 'Jurisdiction', 'Species group', 'Tenure (En)', 'Number of seedlings', 'Area (hectares)', 'Data qualifier', 'Data Qualifier']]
df_combo.rename(columns={'Data qualifier': 'Qualifier Seeds', 'Data Qualifier': 'Qualifier Area', 'Species group': 'Species', 'Tenure (En)': 'Tenure'}, inplace=True)
df_combo = df_combo.dropna()
print(df_combo)
#endregion

#region add data
df_combo['width_log'] = 0.
df_combo['height_log'] = 0.
df_combo['width'] = 0.
df_combo['height'] = 0.
df_combo['year_count'] = 0

combo_group = df_combo.groupby(['Jurisdiction','Species', 'Tenure'], as_index=False)
tree_i = 1
test = 0

group_list = []
for name, row in combo_group:
    
    width_log = log10(row['Area (hectares)'].sum())
    height_log = log10(row['Number of seedlings'].sum())
    width = row['Area (hectares)'].sum()
    height = row['Number of seedlings'].sum()
    year_count = row['Year'].count()
    
    for index, row_i in row.iterrows():
        row.at[index, 'width_log'] = width_log
        row.at[index, 'height_log'] = height_log
        row.at[index, 'width'] = width
        row.at[index, 'height'] = height
        row.at[index, 'year_count'] = year_count
        row.at[index, 'tree'] = tree_i

    group_list.append(row)
    tree_i += 1
    test += 1
    print(test)
#endregion

#region output
df_out_year = pd.concat(group_list, axis=0)
df_out = df_out_year.groupby(['tree', 'Jurisdiction', 'Species', 'Tenure', 'width_log', 'height_log', 'width', 'height', 'year_count']).size().reset_index()

df_out_year.columns = ['FDY_' + str(col) for col in df_out_year.columns]
df_out.columns = ['FD_' + str(col) for col in df_out.columns]

df_out.to_csv(os.path.dirname(__file__) + '/forest_data.csv', encoding='utf-8', index=False)
df_out_year.to_csv(os.path.dirname(__file__) + '/forest_data_year.csv', encoding='utf-8', index=False)
#endregion

print('finished')