import pandas as pd
import geopandas as gpd
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from simulator.services.osrm_engine import OSRMEngine
from preprocessing.preprocess_nyc_dataset import extract_bounding_box, BOUNDING_BOX
from shapely.geometry import Point
def create_snapped_trips(df, engine, batch_size=2000):
    mm_origins = []
    mm_destins = []
    for i in range(0, len(df), batch_size):
        print("n: {}".format(i))
        df_ = df.iloc[i : i + batch_size]
        origins = [(lon, lat) for lat, lon in zip(df_.origin_lat, df_.origin_lon)]
        mm_origins += [loc for loc, _ in engine.nearest_road(origins)]
        destins = [(lon, lat) for lat, lon in zip(df_.destination_lat, df_.destination_lon)]
        mm_destins += [loc for loc, _ in engine.nearest_road(destins)]

    df[['origin_lon', 'origin_lat']] = mm_origins
    df[['destination_lon', 'destination_lat']] = mm_destins
    df = extract_bounding_box(df, BOUNDING_BOX)
    return df

def create_snapped_hex(df,engine,batch_size=2000):
    nearest_coord = []
    batch_size=min(df.shape[0],batch_size)
    for i in range(0,df.shape[0],batch_size):
        df_i = df.iloc[i:i+batch_size]
        coord = [(lon,lat) for lat, lon in zip(df_i.lat,df_i.lon)]
        nearest_coord +=[loc for loc,_ in engine.nearest_road(coord)]
    
    tagged_lat = [];tagged_lon = []
    nearest_coord_pts = [Point(pt[0],pt[1]) for pt in nearest_coord]
    for poly, points,lat_lon in zip(df.geometry,nearest_coord_pts,nearest_coord):
        if points.within(poly):
            tagged_lat.append(lat_lon[1])
            tagged_lon.append(lat_lon[0])
        else:
            tagged_lat.append(-1)
            tagged_lon.append(-1)
    df['tagged_lat'],df['tagged_lon']=tagged_lat,tagged_lon     
    
    return df

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("input_file", help="input csv file path of ride requests to be map match")
    # parser.add_argument("output_file", help="output csv file path")
    # args = parser.parse_args()
    
    engine = OSRMEngine()

    # df = pd.read_csv(args.input_file, index_col='id')
    # print("load {} rows".format(len(df)))
    # df = create_snapped_trips(df, engine)
    # print("extract {} rows".format(len(df)))
    # df.to_csv(args.output_file)

    df = gpd.read_file('data/NYC_shapefiles/clustered_hex.shp')

    snapped_hex = create_snapped_hex(df,engine)
    print(snapped_hex.head)
    snapped_hex.to_file('data/NYC_shapefiles/tagged_clustered_hex.shp')
    selected_hex = snapped_hex['tagged_lon'!=-1]

    # print((snapped_hex[snapped_hex['tagged_coord']!=-1]).shape[0],snapped_hex.shape[0])



