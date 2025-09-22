# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 10:33:17 2025

A code for the plots of the mass-balance paper

@author: Yasaman Nemati
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import matplotlib.patches as patches
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


d_im = pd.read_csv(r'Area-Piche-RE.csv')
d_mb = pd.read_csv('petrophysics_and_massbalance.csv', na_values=-999)



"""Immobile element"""
# =============================================================================
# fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# 
# # Define the y-variables and their corresponding subplot positions
# y_vars = ['Ti (%)', 'Zr (ppm)', 'Nb (ppm)', 'Y (ppm)']
# subplot_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
# subplot_labels = ['(A)', '(B)', '(C)', '(D)']
# y_limits = [(0, 1.5), (0, 250), (0, 25), (0, 50)]  # Adjust these limits as needed
# 
# for i, (y_var, pos, label, ylim) in enumerate(zip(y_vars, subplot_positions, subplot_labels, y_limits)):
#     ax = axes[pos]
#     
#     # Drop NaN values for current pair
#     d_plot = d_im.dropna(subset=['Hf (ppm)', y_var]).copy()
#     
#     # Create scatter plot
#     sns.scatterplot(data=d_plot, x='Hf (ppm)', y=y_var, s=40, ax=ax)
#     
#     # Calculate the linear regression and R²
#     slope, intercept, r_value, p_value, std_err = stats.linregress(d_plot['Hf (ppm)'], d_plot[y_var])
#     r_squared = r_value**2
#     
#     # Add the regression line to the plot
#     x_values = np.array(ax.get_xlim())
#     y_values = intercept + slope * x_values
#     ax.plot(x_values, y_values, '--', color='gray')
#     
#     # Set labels and formatting
#     ax.set_xlabel('Hf (ppm)', fontsize=16)
#     ax.set_ylabel(y_var, fontsize=16)
#     ax.tick_params(axis='both', which='major', labelsize=15)
#     ax.set_xlim(0, 7)
#     ax.set_ylim(ylim)
#     ax.grid(True, alpha=0.3)
#     
#     # Add R² value in a box positioned to avoid overlap with subplot label
#     ax.text(0.23, 0.97, f'R² = {r_squared:.3f}', transform=ax.transAxes, fontsize=15, 
#             verticalalignment='top', horizontalalignment='center',
#             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
#     
#     # Add subplot label in top left corner
#     ax.text(0.02, 0.97, label, transform=ax.transAxes, fontsize=16, fontweight='bold',
#             verticalalignment='top')
# 
# plt.tight_layout()
# 
# plt.savefig('immobile-Hf-Zr.pdf', dpi=300)
# plt.show()
# =============================================================================

"""Plotting the GR"""
# =============================================================================
# ### Plotting for boreholes 1 & 2
# # Create figure with two subplots vertically stacked
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
# 
# # Define the color mapping to match your previous correlation plots
# # Based on the correlation matrix, these appear to be the colors used
# lithology_colors = {
#     'M1CC': '#1f77b4',    # Blue (similar to the correlation plot)
#     'M1Cb': '#ff7f0e',    # Orange 
#     'M1CbFu': '#2ca02c',  # Green
#     'M1TC': '#d62728'     # Red
# }
# 
# # Function to create plot for a specific well
# def create_well_plot(well_name, ax, title_suffix=""):
#     # Create a copy of the dataframe to avoid SettingWithCopyWarning
#     d_GR = d_mb[d_mb['well'] == well_name].copy()
#     
#     # Convert K2O to numeric, handling any non-numeric values
#     d_GR['K2O'] = pd.to_numeric(d_GR['K2O'], errors='coerce')
#     
#     # Group and aggregate data
#     averaged_data = d_GR.groupby(['Litho', 'Litho2'], as_index=False).agg({
#         'K2O': lambda x: x.mean(skipna=True), 
#         'K': 'mean', 
#         'GR': 'mean', 
#         'length': 'mean'
#     })
#     
#     # Map colors based on Litho2 column
#     bar_colors = [lithology_colors.get(litho2, '#808080') for litho2 in averaged_data['Litho2']]
#     
#     # Create bar plot
#     x_positions = range(len(averaged_data))
#     bars = ax.bar(x_positions, averaged_data['GR'], color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
#     
#     # Set labels and formatting
#     ax.set_ylabel('GR (API)', fontsize=16)
#     ax.tick_params(axis='y', labelsize=14)
#     ax.tick_params(axis='x', labelsize=14)
#     ax.set_xticks(x_positions)
#     ax.set_xticklabels(averaged_data['Litho'], rotation=45, ha='right')
#     
#     # Add simple letter label in corner
#     ax.text(0.02, 0.98, f'({title_suffix})', 
#             transform=ax.transAxes, fontsize=14, fontweight='bold', 
#             verticalalignment='top')
#     
#     # Determine which plot we're on by comparing with the global ax2
#     if ax is ax2:
#         ax.set_xlabel('Lithology', fontsize=16)
#     else:
#         ax.set_xlabel('')  # Remove x-axis label for top plot
#     
#     # Create the second y-axis (twin)
#     ax_twin = ax.twinx()
#     
#     # Plot K2O and K on the second y-axis
#     k2o_line = ax_twin.plot(x_positions, averaged_data['K2O'], color='red', marker='o', 
#                            linewidth=2, markersize=8, label='K$_2$O (%)')
#     k_line = ax_twin.plot(x_positions, averaged_data['K']*200, color='blue', marker='s', 
#                          linewidth=2, markersize=8, label='K (%) × 200')
#     
#     ax_twin.set_ylabel('K$_2$O & K (%)', fontsize=16)
#     ax_twin.tick_params(axis='y', labelsize=14)
#     
#     # Add grid
#     ax.grid(True, alpha=0.3)
#     
#     return ax, ax_twin
# 
# # Create plots for both wells - using uppercase A and B
# ax1, ax1_twin = create_well_plot('1', ax1, 'A')
# ax2, ax2_twin = create_well_plot('2', ax2, 'B')
# 
# # Create custom legend handles
# legend_handles = []
# legend_labels = []
# 
# # Add lithology color legend
# for litho, color in lithology_colors.items():
#     if litho in d_mb['Litho2'].values:  # Only include lithologies present in data
#         legend_handles.append(plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.8, edgecolor='black'))
#         legend_labels.append(litho)
# 
# # Add line plot legends
# k2o_handle = plt.Line2D([0], [0], color='red', marker='o', linestyle='-', 
#                         linewidth=2, markersize=8, label='K$_2$O (%)')
# k_handle = plt.Line2D([0], [0], color='blue', marker='s', linestyle='-', 
#                      linewidth=2, markersize=8, label='K (%) × 200')
# 
# legend_handles.extend([k2o_handle, k_handle])
# legend_labels.extend(['K$_2$O (%)', 'K (%) × 200'])
# 
# # Place the legend outside the plot on the right side
# plt.figlegend(legend_handles, legend_labels, loc='center left', bbox_to_anchor=(0.80, 0.5), 
#               ncol=1, fontsize=13, title='GR', title_fontsize=14)
# 
# # Adjust layout to make room for legend
# plt.tight_layout(rect=[0, 0, 0.85, 1])
# 
# # Save the figure
# plt.savefig('K2O-K-GR-comparison.pdf', dpi=300)
# plt.show()
# =============================================================================


"""Statistical analysis for K-K2O"""
# =============================================================================
# d_mb['K2O'] = pd.to_numeric(d_mb['K2O'], errors='coerce')
# d_mb['GR'] = pd.to_numeric(d_mb['GR'], errors='coerce')
# d_mb['K'] = pd.to_numeric(d_mb['K'], errors='coerce')
# df_clean = d_mb.dropna(subset=['K2O', 'GR', 'K']).copy()
# df_clean = df_clean[df_clean['K2O'] >= 0].copy()
# 
# # Separate boreholes
# borehole_1 = df_clean[df_clean['well'] == '1'].copy()
# borehole_2 = df_clean[df_clean['well'] == '2'].copy()
# 
# # Calculate statistics
# corr_k_k2o_bh1 = borehole_1['K'].corr(borehole_1['K2O'])
# corr_k_k2o_bh2 = borehole_2['K'].corr(borehole_2['K2O'])
# 
# # Fit models
# X_bh1 = borehole_1[['K']]
# y_bh1 = borehole_1['K2O']
# model_bh1 = LinearRegression()
# model_bh1.fit(X_bh1, y_bh1)
# r2_bh1 = model_bh1.score(X_bh1, y_bh1)
# 
# X_bh2 = borehole_2[['K']]
# y_bh2 = borehole_2['K2O']
# model_bh2 = LinearRegression()
# model_bh2.fit(X_bh2, y_bh2)
# r2_bh2 = model_bh2.score(X_bh2, y_bh2)
# 
# # Error analysis for M1CbFu intervals
# typical_k2o_error = 0.05  # 5% relative error
# 
# # Create the publication figure with 3x2 layout
# fig, axes = plt.subplots(3, 2, figsize=(14, 16))
# 
# # Updated color scheme for lithologies to match correlation matrix plots
# colors = {
#     'M1CC': '#1f77b4',    # Blue (same as correlation plot)
#     'M1Cb': '#ff7f0e',    # Orange
#     'M1CbFu': '#2ca02c',  # Green
#     'M1TC': '#d62728'     # Red
# }
# 
# # Calculate correlation matrices first
# corr_matrix_bh1 = borehole_1[['GR', 'K', 'K2O']].corr()
# corr_matrix_bh2 = borehole_2[['GR', 'K', 'K2O']].corr()
# 
# # Determine common color scale limits for both correlation matrices
# all_corr_values = np.concatenate([
#     corr_matrix_bh1.values.flatten(),
#     corr_matrix_bh2.values.flatten()
# ])
# # Remove NaN values and diagonal (1.0) values for better scaling
# all_corr_values = all_corr_values[~np.isnan(all_corr_values)]
# all_corr_values = all_corr_values[all_corr_values != 1.0]
# vmin = np.min(all_corr_values)
# vmax = 1.0  # Set max to 1.0 to include the full range
# 
# # Create a custom colormap that starts with more saturated red
# import matplotlib.colors as mcolors
# # Use only the darker portion of the Reds colormap (from 0.3 to 1.0)
# reds_cmap = plt.cm.Reds
# colors_list = [reds_cmap(i) for i in np.linspace(0.3, 1.0, 256)]
# custom_red_cmap = mcolors.LinearSegmentedColormap.from_list('custom_reds', colors_list)
# 
# # Plot 1: Correlation Matrix - Borehole 1
# ax1 = axes[0, 0]
# mask = np.triu(np.ones_like(corr_matrix_bh1, dtype=bool))
# sns.heatmap(corr_matrix_bh1, mask=mask, annot=True, cmap=custom_red_cmap, center=None, 
#             square=True, fmt='.2f', cbar_kws={"shrink": .8}, ax=ax1, 
#             annot_kws={"size": 14}, vmin=vmin, vmax=vmax)
# ax1.text(0.02, 0.98, '(A)', transform=ax1.transAxes, fontsize=16, fontweight='bold', 
#          verticalalignment='top', horizontalalignment='left')
# ax1.tick_params(axis='both', which='major', labelsize=14)
# # Update tick labels with proper subscript notation
# labels = ax1.get_yticklabels()
# ax1.set_yticklabels(['GR', 'K', 'K₂O'], fontsize=14)
# 
# # Make colorbar text larger
# cbar1 = ax1.collections[0].colorbar
# cbar1.ax.tick_params(labelsize=12)
# 
# # Plot 2: Correlation Matrix - Borehole 2
# ax2 = axes[0, 1]
# mask = np.triu(np.ones_like(corr_matrix_bh2, dtype=bool))
# sns.heatmap(corr_matrix_bh2, mask=mask, annot=True, cmap=custom_red_cmap, center=None, 
#             square=True, fmt='.2f', cbar_kws={"shrink": .8}, ax=ax2, 
#             annot_kws={"size": 14}, vmin=vmin, vmax=vmax)
# ax2.text(0.02, 0.98, '(B)', transform=ax2.transAxes, fontsize=16, fontweight='bold', 
#          verticalalignment='top', horizontalalignment='left')
# ax2.tick_params(axis='both', which='major', labelsize=14)
# # Update tick labels with proper subscript notation
# ax2.set_yticklabels(['GR', 'K', 'K₂O'], fontsize=14)
# 
# # Make colorbar text larger
# cbar2 = ax2.collections[0].colorbar
# cbar2.ax.tick_params(labelsize=12)
# 
# # Plot 3: K vs K2O - Borehole 1
# ax3 = axes[1, 0]
# for litho in borehole_1['Litho2'].unique():
#     subset = borehole_1[borehole_1['Litho2'] == litho]
#     ax3.scatter(subset['K'], subset['K2O'], 
#                c=colors.get(litho, 'gray'), label=litho, alpha=0.7, s=50)
# 
# # Add regression line
# k_range = np.linspace(0, 0.9, 100)
# k2o_pred = model_bh1.predict(k_range.reshape(-1, 1))
# ax3.plot(k_range, k2o_pred, 'k--', linewidth=2, 
#          label=f'R² = {r2_bh1:.3f}')
# ax3.set_xlabel('K (%)', fontsize=16)  # Increased from 14
# ax3.set_ylabel('K₂O (%)', fontsize=16)  # Increased from 14
# ax3.text(0.02, 0.98, '(C)', transform=ax3.transAxes, fontsize=16, fontweight='bold', 
#          verticalalignment='top', horizontalalignment='left')
# ax3.legend(fontsize=14)  # Increased from 12
# ax3.grid(True, alpha=0.3)
# ax3.tick_params(axis='both', which='major', labelsize=14)  # Increased from 12
# # Set consistent axis limits
# ax3.set_xlim(0, 0.9)
# ax3.set_ylim(0, 180)
# 
# # Plot 4: K vs K2O - Borehole 2
# ax4 = axes[1, 1]
# for litho in borehole_2['Litho2'].unique():
#     subset = borehole_2[borehole_2['Litho2'] == litho]
#     ax4.scatter(subset['K'], subset['K2O'], 
#                c=colors.get(litho, 'gray'), label=litho, alpha=0.7, s=50)
# 
# # Add regression line
# k_range_bh2 = np.linspace(0, 0.9, 100)
# k2o_pred_bh2 = model_bh2.predict(k_range_bh2.reshape(-1, 1))
# ax4.plot(k_range_bh2, k2o_pred_bh2, 'k--', linewidth=2, 
#          label=f'R² = {r2_bh2:.3f}')
# ax4.set_xlabel('K (%)', fontsize=16)  # Increased from 14
# ax4.set_ylabel('K₂O (%)', fontsize=16)  # Increased from 14
# ax4.text(0.02, 0.98, '(D)', transform=ax4.transAxes, fontsize=16, fontweight='bold', 
#          verticalalignment='top', horizontalalignment='left')
# ax4.legend(fontsize=14)  # Increased from 12
# ax4.grid(True, alpha=0.3)
# ax4.tick_params(axis='both', which='major', labelsize=14)  # Increased from 12
# # Set consistent axis limits
# ax4.set_xlim(0, 0.9)
# ax4.set_ylim(0, 180)
# 
# # Plot 5: Combined comparison
# ax5 = axes[2, 0]
# 
# # Use new colors for plot E (different from lithology colors)
# borehole1_color = '#9467bd'  # Purple
# borehole2_color = '#8c564b'  # Brown
# 
# ax5.scatter(borehole_1['K'], borehole_1['K2O'], 
#            alpha=0.6, label=f'Borehole 1 (r={corr_k_k2o_bh1:.3f})', color=borehole1_color, s=40)
# ax5.scatter(borehole_2['K'], borehole_2['K2O'], 
#            alpha=0.6, label=f'Borehole 2 (r={corr_k_k2o_bh2:.3f})', color=borehole2_color, s=40)
# 
# # Add regression lines
# k_range_combined = np.linspace(0, 0.9, 100)
# k2o_pred_bh1_line = model_bh1.predict(k_range_combined.reshape(-1, 1))
# k2o_pred_bh2_line = model_bh2.predict(k_range_combined.reshape(-1, 1))
# ax5.plot(k_range_combined, k2o_pred_bh1_line, '--', linewidth=2, color=borehole1_color, alpha=0.8)
# ax5.plot(k_range_combined, k2o_pred_bh2_line, '--', linewidth=2, color=borehole2_color, alpha=0.8)
# 
# ax5.set_xlabel('K (%)', fontsize=16)  # Increased from 14
# ax5.set_ylabel('K₂O (%)', fontsize=16)  # Increased from 14
# ax5.text(0.02, 0.98, '(E)', transform=ax5.transAxes, fontsize=16, fontweight='bold', 
#          verticalalignment='top', horizontalalignment='left')
# # Move legend lower to avoid overlapping with (E) label
# ax5.legend(fontsize=14, loc='upper left', bbox_to_anchor=(0.01, 0.94))  # Increased font size and repositioned
# ax5.grid(True, alpha=0.3)
# ax5.tick_params(axis='both', which='major', labelsize=14)  # Increased from 12
# # Set consistent axis limits
# ax5.set_xlim(0, 0.9)
# ax5.set_ylim(0, 180)
# 
# # Turn off subplot 6 (bottom-right)
# axes[2, 1].axis('off')
# 
# plt.tight_layout()
# plt.savefig('K_K2O_analysis.pdf', dpi=300)
# plt.show()
# 
# print("BH1 intercept:", model_bh1.intercept_)
# print("BH2 intercept:", model_bh2.intercept_)
# =============================================================================
"""CaO and density"""
# =============================================================================
# # borehole 1 data
# d_au = d_mb[d_mb['well'] == '1'].copy()
# 
# # Clean the data: Drop rows with NaN and ensure numeric types
# d_au_clean = d_au.dropna(subset=['rho_ND', 'CaO']).copy()
# d_au_clean['rho_ND'] = pd.to_numeric(d_au_clean['rho_ND'], errors='coerce')
# d_au_clean['CaO'] = pd.to_numeric(d_au_clean['CaO'], errors='coerce')
# 
# # Create the figure and the first axis
# fig, ax1 = plt.subplots(figsize=(18, 6))
# 
# # Plot CaO as bar plot
# # Calculate bar heights and positions based on 'From' and 'To' columns
# bar_heights = d_au_clean['CaO']
# bar_bottoms = d_au_clean['From']
# bar_tops = d_au_clean['To']
# bar_widths = bar_tops - bar_bottoms
# 
# # Add background coloration for 'Inverse relationship'
# for index, row in d_au_clean.iterrows():
#     if row['inverse_rho_Ca'] == 'Inverse relationship':
#         # Create a light gray rectangle for rows with 'Inverse relationship'
#         rect = patches.Rectangle((row['From'], -3), 
#                                  row['To'] - row['From'], 
#                                  10, 
#                                  facecolor='lightgray', 
# #                                 alpha=0.3, 
#                                  zorder=0)
#         ax1.add_patch(rect)
# 
# 
# # Create bar plot on the first axis
# bars = ax1.bar(bar_bottoms + (bar_widths/2), bar_heights, 
#                width=bar_widths, 
#                align='center', 
#                color='lightblue',
#                alpha=0.7, 
#                edgecolor='black')
# 
# # Set labels and title for the first axis
# ax1.set_xlabel('Depth (m)', fontsize=16)
# ax1.set_ylabel('CaO (%)', color='blue', fontsize=16)
# 
# ax1.tick_params(axis='y', labelcolor='blue')
# 
# # Create twin axis
# ax2 = ax1.twinx()
# 
# # Separate the data into different segments
# segment1 = d_au_clean[d_au_clean['To'] <= 57.81]
# segment2 = d_au_clean[(d_au_clean['From'] >= 58.81) & (d_au_clean['To'] <= 84.98)]
# segment3 = d_au_clean[d_au_clean['From'] >= 90.98]
# 
# # Plot each segment separately
# x1 = segment1['From'] + (segment1['To'] - segment1['From'])/2
# x2 = segment2['From'] + (segment2['To'] - segment2['From'])/2
# x3 = segment3['From'] + (segment3['To'] - segment3['From'])/2
# 
# ax2.plot(x1, segment1['rho_ND'], color='red', linewidth=2, marker='o')
# ax2.plot(x2, segment2['rho_ND'], color='red', linewidth=2, marker='o')
# ax2.plot(x3, segment3['rho_ND'], color='red', linewidth=2, marker='o')
# 
# # Set label for the second axis
# ax2.set_ylabel(r'Density ($\mathrm{g/cm^3}$)', color='red', fontsize=16)
# ax2.tick_params(axis='y', labelcolor='red')
# ax1.tick_params(axis='both', which='major', labelsize=14)
# ax2.tick_params(axis='both', which='major', labelsize=14)
# 
# 
# plt.xlim(24.23, 245.81)
# ax1.set_ylim(-1, 7)
# ax2.set_ylim(2.62, 2.9)
# 
# ax1.grid(True, which='minor', linestyle=':', linewidth=0.5, color='gray', alpha=0.3)
# ax1.grid(True, which='major', linestyle='--', linewidth=0.7, color='gray', alpha=0.5)
# ax1.minorticks_on()
# 
# plt.tight_layout()
# plt.savefig("CaO-Density.pdf", dpi=300)
# plt.show()
# =============================================================================


"""Magsus image, attaching to each other"""
# =============================================================================
# from PIL import Image, ImageDraw, ImageFont
# import sys
# 
# def combine_images(image1_path, image2_path, output_path, space=200, top_space=200, label_font_size=180):
#     # Open the images
#     img1 = Image.open(image1_path).convert("RGBA")
#     img2 = Image.open(image2_path).convert("RGBA")
#     
#     # Ensure the images have the same width
#     width = max(img1.width, img2.width)
#     new_height = img1.height + img2.height + space + top_space
#     
#     # Create a new image with white background
#     new_img = Image.new("RGBA", (width, new_height), (255, 255, 255, 255))
#     
#     # Paste the first image with top space
#     new_img.paste(img1, (0, top_space))
#     
#     # Paste the second image below the first with space in between
#     new_img.paste(img2, (0, img1.height + space + top_space))
#     
#     # Draw labels in the white space
#     draw = ImageDraw.Draw(new_img)
#     try:
#         font = ImageFont.truetype("arialbd.ttf", label_font_size)
#     except IOError:
#         font = ImageFont.load_default()
#     
#     draw.text((5, 5), "(A)", fill="black", font=font)  # Label A in the top white space
#     draw.text((5, img1.height + space), "(B)", fill="black", font=font)  # Label B in the space before the second image
#     
#     # Save the output image
#     new_img.save(output_path)
#     print(f"Combined image saved as {output_path}")
# 
# 
# if __name__ == "__main__":
#     image2_path = "bh1-mgsus-ano.png"
#     image1_path = "bh1-mgsus-nor.png"
#     output_path = "magsus-combined.pdf"
#     
#     combine_images(image1_path, image2_path, output_path, space=200, top_space=200, label_font_size=120)
# =============================================================================













