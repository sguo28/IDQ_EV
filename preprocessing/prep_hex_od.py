import pandas as pd
import geopandas as gpd

from scipy.spatial import KDTree


# first get cooresponding HEX_ID of origin and destination
taxi_records = pd.read_csv('data/trip_records/hex_trips_2016-05.csv')
# taxi_gdf_o = gpd.GeoDataFrame(taxi_records,geometry=gpd.points_from_xy(taxi_records.origin_lon,taxi_records.origin_lat))

polygons = gpd.read_file('data/NYC_shapefiles/reachable_hexes.shp') # or insert the full-action-space hex shapefile and only select 'lon' != -1

# polygons = polygons.to_crs({'init':'epsg:4326'}) # lon,lat unit: feet to degree

polygons['CELL_ID']=polygons.index

hex_tree = KDTree(polygons[['lon','lat']]) # hex's centroid lon, lat
_,ohex_ids = hex_tree.query(taxi_records[['origin_lon','origin_lat']])
_,dhex_ids = hex_tree.query(taxi_records[['destination_lon','destination_lat']])
taxi_records['o_hex_id'] = ohex_ids
taxi_records['d_hex_id'] = dhex_ids

taxi_records['hour'] = (taxi_records['request_datetime'] - taxi_records['request_datetime'][0])//(60*60)%24

within_trip_od = taxi_records[['hour','o_hex_id','d_hex_id','trip_time']]


trip_count = within_trip_od.groupby(['hour','o_hex_id','d_hex_id'])['trip_time'].count().reset_index()
# trip_count = trip_count[['hour','o_hex_id','d_hex_id']]

trip_count.columns = ['h', 'o', 'd', 'n']
trip_count.to_csv('data/trip_od_hex.csv',index_label=False)


trip_duration = within_trip_od.groupby(['hour','o_hex_id','d_hex_id'])['trip_time'].mean().reset_index()

trip_duration.columns = ['h', 'o', 'd', 't']

trip_duration.to_csv('data/trip_time_od_hex.csv',index_label=False)



