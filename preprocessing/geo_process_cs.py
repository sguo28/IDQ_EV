import sys
import os
import pandas as pd
import geopandas as gpd
import numpy as np
# this is grid based process
df = pd.read_csv('data/EV_facility/alt_fuel_stations.csv')
df = df[['Fuel Type Code','State','EV Level2 EVSE Num', 'EV DC Fast Count','Latitude','Longitude']].replace(np.nan, 0)
df = df[df['Fuel Type Code']=='ELEC']
df_ny = df[df['State']=='NY']

gdf_ny = gpd.GeoDataFrame(df_ny,geometry=gpd.points_from_xy(df_ny.Longitude,df_ny.Latitude))

polygons = gpd.GeoDataFrame.from_file('data/NYC_shapefiles/selected_hexagon.shp')

polygons = polygons.to_crs({'init':'epsg:4326'})

pts = gdf_ny.copy()
pts.crs = polygons.crs

pts_within = gpd.sjoin(pts,polygons,how="left", op="within")
pts_within = pts_within[pts_within['Id'].notnull()]
df_within = pd.DataFrame(pts_within)
#df_within.rename(columns={('Fuel Type Code', 'State', 'EV Level2 EVSE Num', 'EV DC Fast Count','Latitude', 'Longitude', 'geometry', 'index_right', 'Id', 'GRID_ID','lat', 'lon'):('Fuel Type Code', 'State', 'EV_Level2', 'EV_DC_Fast','Latitude', 'Longitude', 'geometry', 'index_right', 'Id', 'GRID_ID','lat', 'lon')},inplace = True)
df_within.columns=['Fuel Type Code', 'State', 'EV_Level2', 'EV_DC_Fast','Latitude', 'Longitude', 'geometry', 'index_right', 'Id', 'GRID_ID','lat', 'lon']
df_within.to_csv('data/processed_cs_v2.csv')

import pandas as pd
import geopandas as gpd
from scipy.spatial import KDTree
df=gpd.read_file('data/NYC_shapefiles/tagged_clustered_hex.shp')
hxt= KDTree(df[['lon','lat']])
pcs = pd.read_csv('data/processed_cs.csv')
coord = pcs[['Longitude','Latitude']].to_numpy()
_, hex_id = hxt.query(coord)

pcs['hex_id']=hex_id
pcs = pcs[['EV_Level2', 'EV_DC_Fast', 'Latitude', 'Longitude','hex_id']]
pcs.to_csv('data/process_cs_new.csv',index=False)

