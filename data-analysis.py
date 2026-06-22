import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import contextily as ctx
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import numpy as np
import os
import ssl
from pathlib import Path
import pickle

# Read in csv or grab the county imputed pickle
df_path = Path("dfs/df.pkl")
if df_path.is_file():
    with open('df.pkl', 'rb') as f:
        df = pickle.load(f) 
else:
    user_choice = input("You are loading in the raw dataset. Wish to proceed? ")
    if user_choice.lower()=='yes':
        df = pd.read_csv("pfas-data.csv") # Raw data load in

# Filtered/subset dfs
# PFAS only
pfas_strings = ['PF', 'FTCA', 'FTS', 'HFPA-DA', 'ADONA', 'ClP', 'FOS']
pfas_string = '|'.join(pfas_strings)
df_pfas = df.filter(regex=pfas_string)

# Enviro feats
df_env = df.drop(columns=df_pfas.columns.to_list())

def pfasMissing():
    print(df['Fac_Conf_type'].head)

    # Missing data heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        df_pfas.isna(),
        cmap="viridis",
        cbar=False,
        yticklabels=False
    )
    plt.title("Heatmap of Missing PFAS Data")
    plt.close()
    #plt.show()

def contaminationMap():
    # Load geospatial files
    industrial_gdf = gpd.read_file(
        "/Users/JasL/pfas-mrl-jasmine/shape-files/industrial/PFAS_Investigation_Sites.shp"
    )
    military_gdf = gpd.read_file(
        "/Users/JasL/pfas-mrl-jasmine/shape-files/military/tl_2022_us_mil.shp"
    )

    # Convert wells dataframe to GeoDataFrame in standard lat/lon coordinates
    wells_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(
            df["longitude"],
            df["latitude"]
        ),
        crs="EPSG:4326"
    )

    industrial_gdf.columns = industrial_gdf.columns.str.lower()
    military_gdf.columns = military_gdf.columns.str.lower()

    source_colors = {
        "Airport": "dodgerblue",
        "Bulk Terminal": "orange",
        "Chrome Plater": "green",
        "Refinery": "brown"
    }
    global sources # make a global variable
    sources = list(source_colors.keys())

    # Make same CRS --> 3857
    industrial_gdf = industrial_gdf.to_crs(epsg=3857)
    military_gdf = military_gdf.to_crs(epsg=3857)
    wells_gdf = wells_gdf.to_crs(epsg=3857)

    # Convert California bounding box from degrees (4326) to meters (3857)
    bounds_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([-125, -114], [32, 43]), 
        crs="EPSG:4326"
    ).to_crs(epsg=3857)

    xmin_bound, xmax_bound = bounds_gdf.geometry.x
    ymin_bound, ymax_bound = bounds_gdf.geometry.y

    # Crop layers to California bounds using meters
    industrial_gdf = industrial_gdf.cx[xmin_bound:xmax_bound, ymin_bound:ymax_bound]
    military_gdf = military_gdf.cx[xmin_bound:xmax_bound, ymin_bound:ymax_bound]
    wells_gdf = wells_gdf.cx[xmin_bound:xmax_bound, ymin_bound:ymax_bound]

    # Filter industrial sites only to target sources now that CRS mismatch won't drop rows
    industrial_gdf = industrial_gdf[industrial_gdf["site_type"].isin(sources)]

    # Debugging check to confirm your target rows survived the transformations
    print("Remaining unique site types to map:", industrial_gdf['site_type'].unique())
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot Military Areas
    military_gdf.plot(
        ax=ax,
        color="purple",
        alpha=0.15,
        edgecolor="purple",
        linewidth=0.5,
        label="Military Training Areas",
        zorder=1
    )

    # Plot Industrial Sites by Type
    for source_type, group in industrial_gdf.groupby("site_type"):
        group.plot(
            ax=ax,
            color=source_colors[source_type],
            markersize=60,
            edgecolor="black",
            linewidth=0.4,
            alpha=0.85,
            label=source_type,
            zorder=2
        )

    # Plot Well Locations
    wells_gdf.plot(
        ax=ax,
        color="red",
        markersize=12,
        edgecolor="white",
        linewidth=0.6,
        alpha=0.9,
        label="Well Locations",
        zorder=3
    )
    """

    """
    # Add web map background (Contextily handles 3857 naturally)
    if not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context
    ctx.add_basemap(
        ax,
        source=ctx.providers.CartoDB.Positron
    )

    # Dynamic viewport sizing using data bounds (now safely calculated in meters)
    xmin, ymin, xmax, ymax = wells_gdf.total_bounds
    padding = 30000  # 30 Kilometers

    ax.set_xlim(xmin - padding, xmax + padding)
    ax.set_ylim(ymin - padding, ymax + padding)
    ax.set_axis_off()

    plt.title(
        "California Well Locations and Potential PFAS Contamination Sources",
        fontsize=18
    )
    plt.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=11
    )

    plt.tight_layout()
    plt.close()

    #plt.show()
    """

def enviroMissing():
    # Environmental Data Missingness

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        df_env.isna(),
        cmap="viridis",
        cbar=False,
        yticklabels=False
    )
    plt.title("Heatmap of Missing Environmental Data") # Added title for clarity
    plt.close()

def corrMatrices():
    # Correlation matrix / heatmap
    pfas_corr_spear = df_pfas.corr(method='spearman') # nonlinear, monotonic relationships
    pfas_corr_pear = df_pfas.corr(method='pearson')

    plt.figure(figsize=(10, 8))

    # Create a customized heatmap
    sns.heatmap(
        pfas_corr_pear, 
        annot=True,          # Overlays the exact correlation numbers on the cells
        fmt=".2f",           # Rounds the displayed numbers to 2 decimal places
        cmap="coolwarm",     # Uses a red-to-blue divergent color scale
        vmin=-1,             # Fixes the minimum color scale anchor to -1
        vmax=1,              # Fixes the maximum color scale anchor to 1
        square=True,          # Forces the cells to be perfect squares
        annot_kws={"size": 4}  
    )
    plt.title("PFAS Pearson Correlation Matrix")
    plt.close()

    #plt.show()

    plt.figure(figsize=(10, 8))

    # Create a customized heatmap
    sns.heatmap(
        pfas_corr_spear, 
        annot=True,          # Overlays the exact correlation numbers on the cells
        fmt=".2f",           # Rounds the displayed numbers to 2 decimal places
        cmap="coolwarm",     # Uses a red-to-blue divergent color scale
        vmin=-1,             # Fixes the minimum color scale anchor to -1
        vmax=1,              # Fixes the maximum color scale anchor to 1
        square=True,          # Forces the cells to be perfect squares
        annot_kws={"size": 4}  
    )

    plt.title("PFAS Spearman Correlation Matrix")
    plt.close()
    #plt.show()

    # Correlation in env feats
    df_env_plt = df_env.select_dtypes(include='number')
    env_pear = df_env_plt.corr(method='pearson')
    env_spear = df_env_plt.corr(method='spearman')

def basinDense():
    per_basin_counts = df.groupby('gm_gis_dwr_basin').size()
    per_basin_counts.hist(bins=50, edgecolor='black')
    plt.title("Histogram of Data Per Hydrogeologic Basin")
    plt.xlabel("Number of Data Points in a Basin")
    plt.ylabel("Frequency of Basins")
    plt.show()

# Outliers and data distributions; boxplots and histograms --> also already have script for this...

# VIF
def VIF(df):
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    numeric_df = numeric_df.loc[:, numeric_df.std() > 0]

    # Remove perfectly collinear to avoid divide by zero / inf error
    while True:
        X = add_constant(numeric_df)
        
        vifs = []
        for i in range(X.shape[1]):
            try:
                vif = variance_inflation_factor(X.values, i)
            except ZeroDivisionError:
                vif = np.inf
            vifs.append(vif)
            
        vif_df = pd.DataFrame({"feature": X.columns, "VIF": vifs})
        
        # Exclude the constant column from being dropped
        vif_df = vif_df[vif_df["feature"] != "const"]
        
        # Find the maximum VIF value
        max_vif = vif_df["VIF"].max()
        
        # If the highest VIF is infinite (or above your threshold, e.g., 5 or 10), drop it
        if np.isinf(max_vif) or max_vif > 10:
            # Get the name of the feature with the highest VIF
            feature_to_drop = vif_df.loc[vif_df["VIF"].idxmax(), "feature"]
            print(f"Dropping '{feature_to_drop}' (VIF: {max_vif})")
            numeric_df = numeric_df.drop(columns=[feature_to_drop])
        else:
            # Stop the loop when all remaining features have acceptable VIFs
            break

    # Print final clean VIF dataframe
    print("\nFinal VIF Dataframe:")
    print(vif_df)

# PFAS Fingerprint Based Contamination Source Proximity (e.g., wastewater, manufacturing, etc)
... # dtype int64 need lookup table


# PFAS Sampling Density --> Identifying "sparse" regions
"""
- per county
- per basin
"""

county_dense = {} # County name: num samples
basin_dense = {} # Basin name: num samples

for county, samples in df.groupby('gm_gis_county'):
    county_dense[county] = len(samples)

for basin, samples in df.groupby('gm_gis_dwr_basin'):
    basin_dense[basin] = samples



# Regional PFAS Fingerprint Differences

# Building Co-Occurence Networks (Spearman corr) Based on Diff Enviro Conditions

# Reducing conditions
"""
Several literature show that nitrate conditions reduce
fluorotelomer precursors significantly

Half lives can differ by OOM
- EX: wastewater treatment plants abundant in in nitrite --> nitrate
Ammonia in human waste converted to nitrate
"""

"""
Temp, Precip, GW recharge --> accelerate metabolism of bacteria
- PFAS often under go biotransformation pathways via bacteria
"""

