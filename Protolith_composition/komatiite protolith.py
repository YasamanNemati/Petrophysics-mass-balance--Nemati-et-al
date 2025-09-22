# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 10:38:05 2022

Bootstrapping the fresh/slightly altered komatiite data that we gathered from
literature to be used as least altered samples for our mass balance calculations.
Then we save the geometric data with 95% confidence in a separate
excel file. (This 95% confidence is from Hood et al. 2019 paper)
For geometric mean, as some data had 0 values, I replaced them with 1e-06 in 
order to avoid having 0.

@author: Yasaman Nemati
"""

import pandas as pd
import numpy as np
from scipy.stats import gmean


df = pd.read_csv('compiled_komatiite_data_from_literature.csv')

df = df.drop(df.columns[0:5], axis=1)


"""Bootstrapping the komatiites"""

# Extract the numeric columns (geochemical data)
geochemical_data = df.select_dtypes(include=[np.number])

# Perform bootstrapping by resampling the geochemical data 5000 times
bootstrap_samples = geochemical_data.sample(n=5000, replace=True)

# Function to calculate geometric mean, excluding NaN values
def calculate_geom_mean(column_data):
    # Replace zero values with a small value (e.g., 1e-6) to avoid zeros affecting the geometric mean
    column_data = column_data.replace(0, 1e-6)
    # Remove NaN values before calculating the geometric mean
    return gmean(column_data.dropna())

# Calculate the geometric mean for each column, excluding NaN values
geom_mean = geochemical_data.apply(calculate_geom_mean)

# Create a new dataframe (dft) excluding the geometric mean row
dft = bootstrap_samples.copy()

# Now, process each column to drop the top and bottom 2.5% (i.e., 125 rows for 5000 rows)
dft = dft.iloc[126:-125]

# Calculate the geometric mean for the trimmed dataset
geom_mean_95ci = dft.apply(calculate_geom_mean)

# Append the results to the original dataframe
geom_mean_df = pd.DataFrame(geom_mean).transpose()
geom_mean_df['Type'] = 'Geometric Mean'

# Add 95% confidence interval (after trimming) to a new dataframe
geom_mean_95ci_df = pd.DataFrame(geom_mean_95ci).transpose()
geom_mean_95ci_df['Type'] = 'Geometric Mean with 95% Confidence Interval'

# Combine the original data with the two new rows
final_df = pd.concat([bootstrap_samples, geom_mean_df], ignore_index=True)

# Save the bootstrapped data along with geometric mean and confidence intervals to a new Excel file
with pd.ExcelWriter('bootstrapped_with_geom_mean_and_confidence.xlsx') as writer:
    final_df.to_excel(writer, index=False)

# Optionally, return the path to the saved file for user download
'bootstrapped_with_geom_mean_and_confidence.xlsx'

# Save the 95% confidence interval to a separate Excel file
with pd.ExcelWriter('geometric_mean_95ci.xlsx') as writer: 
    geom_mean_95ci_df.to_excel(writer, index=False) 

# Optionally, return the path to the saved file for user download
'bootstrapped_komatiite.xlsx'