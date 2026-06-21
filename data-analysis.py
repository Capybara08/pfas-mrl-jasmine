import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import contextily as ctx

df = pd.read_csv("pfas-data.csv")

pfas_strings = ['PF', 'FTCA', 'FTS', 'HFPA-DA', 'ADONA', 'ClP', 'FOS']
pfas_string = '|'.join(pfas_strings)
df_pfas = df.filter(regex=pfas_string)

# Missing data heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    df_pfas.isna(),
    cmap="viridis",
    cbar=False,
    yticklabels=False
)
plt.title("Heatmap of Missing PFAS Data")
#plt.show()

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

# Add web map background (Contextily handles 3857 naturally)
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

# Environmental Data Missingness
df_env = df.drop(columns=df_pfas.columns.to_list())
plt.figure(figsize=(12, 8))
sns.heatmap(
    df_env.isna(),
    cmap="viridis",
    cbar=False,
    yticklabels=False
)

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

per_basin_counts = df.groupby('gm_gis_dwr_basin').size()

# 2. Plot the histogram of those counts
per_basin_counts.hist()
plt.title("Histogram of Data Per Hydrogeologic Basin")
plt.xlabel("Number of Data Points in a Basin")
plt.ylabel("Frequency of Basins")
plt.show()
# Data distribution for same-well samples

# Data normalization

# Outliers and data distributions; boxplots and histograms

# Collinearity --> in this case, I believe we are leveraging collinearity using co-occurence

"""
Bc sm features, create a script that automatically decides the normalization strategy
"""

# Visualizing dimensionality of the dataset using UMAP, t-sne, and PCA