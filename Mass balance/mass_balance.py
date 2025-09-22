# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 09:49:54 2024

This code calculates the mass balance for the geochem data of 4 of my boreholes.
The immobile element is Zr. I have used the bootstrapped and averages komatiite 
data from papers as the least altered data.

@author: Yasaman Nemati
"""

import pandas as pd

# Load the data
least_altered_data = pd.read_excel('Protolith_composition/bootstrapped_komatiite.xlsx')
altered_data = pd.read_excel('geochem-piche-study-area.xlsx')

# Drop rows where Zr is missing in the altered data
altered_data = altered_data.dropna(subset=['Zr'])

# keeping the data until this column as the komatiite data has data until this columns
altered_data = altered_data.loc[:, : 'Pb']

# Extract the Zr concentration from the least altered sample
Zr_least_altered = least_altered_data['Zr'].values[0]

# Extract the columns for location data (first 5 columns)
location_columns = altered_data.columns[:5]

# Extract the element columns (excluding the first 5 and 'Zr') from the altered data
element_columns = [col for col in altered_data.columns[5:] if col != 'Zr']

# Create an empty list to store the results
mass_balance_results = []

# Iterate through each row of the altered data
for index, row in altered_data.iterrows():
    # Start with the location columns in the result row
    result_row = row[:5].tolist()  # First 5 columns (location data)
    
    # Get Zr concentration for current altered sample
    Zr_altered = row['Zr']
    
    # Iterate through each element column and calculate its mass balance
    for element in element_columns:
        if pd.notna(row[element]) and pd.notna(Zr_altered):  # Only calculate if both element and Zr are present
            # Get the concentration of the current element in altered sample
            altered_concentration = row[element]
            
            # Get the concentration of the current element in least altered sample
            least_altered_concentration = least_altered_data[element].values[0]
            
            # Calculate the mass balance for the element
            # Formula: (Zr_least_altered/Zr_altered) * (altered_concentration/least_altered_concentration) - 1
            mass_balance_j = (Zr_least_altered / Zr_altered) * (altered_concentration / least_altered_concentration) - 1
            result_row.append(mass_balance_j)
        else:
            result_row.append(None)  # If either the element or Zr is missing, append None
            
    # Append the result row to the list of results
    mass_balance_results.append(result_row)

# Create a new DataFrame with the results
result_columns = location_columns.tolist() + [f'{element}' for element in element_columns]
result_df = pd.DataFrame(mass_balance_results, columns=result_columns)

# Save the results to a new Excel file
result_df.to_excel('mass_balance_results_v2.xlsx', index=False)
print("Mass balance calculation completed and saved to 'mass_balance_results.xlsx'")


