# hex configuration
import os
db_dir = "../data/db.sqlite3"
DB_HOST_PATH = ''.join(['sqlite:///', db_dir])

OSRM_HOSTPORT = os.getenv("OSRM_HOSTPORT", "localhost:5000")

DEFAULT_LOG_DIR = "../logs/tmp"

DATA_DIR = "../data"

num_reachable_hex = 1347 # number of hex that has intersection with road network
charging_station_shp_path = '../data/NYC_shapefiles/processed_cs.shp'
hex_shp_path = '../data/NYC_shapefiles/clustered_hex.shp' # '../data/NYC_shapefiles/tagged_clustered_hex.shp'
trip_file='../data/trip_od_hex.csv' 
travel_time_file='../data/trip_time_od_hex.csv'
charging_station_data_path = '../data/processed_cs_new.csv'


# config settings for DQN 
learning_rate = 1e-3
gamma = 0.99
replay_buffer_size = 1e5
batch_size = 128
relocation_dim = 7 # 1+6
charging_dim = 5
action_space = [i for i in range(relocation_dim+charging_dim)] # 7 relocation hex candidates, 5 nearest charging stations
input_dim = 4 # num of state fature lon/lat to hex_id
output_dim = relocation_dim+charging_dim
epsilon = 0.01


#
OFF_DURATION=0
NUM_NEAREST_CS = 5
#weights for reward calculation
beta_earning =1 
beta_cost = 1
SOC_penalty =1

# GRID TO INITIAL VEHICLE LOCATION
CENTER_LATITUDE = 40.75
CENTER_LONGITUDE = -73.90
LAT_WIDTH = 18.0 / 60
LON_WIDTH = 18.0 / 60
MIN_LAT, MIN_LON = CENTER_LATITUDE - LAT_WIDTH / 2.0, CENTER_LONGITUDE - LON_WIDTH / 2.0
MAX_LAT, MAX_LON = CENTER_LATITUDE + LAT_WIDTH / 2.0, CENTER_LONGITUDE + LON_WIDTH / 2.0
BOUNDING_BOX = [[MIN_LAT-10, MIN_LON-10], [MAX_LAT+10, MAX_LON+10]] # [[MIN_LAT, MIN_LON], [MAX_LAT, MAX_LON]]
DELTA_LON = 21.0 / 3600
DELTA_LAT = 16.0 / 3600
MAP_WIDTH = int(LON_WIDTH / DELTA_LON) + 1
MAP_HEIGHT = int(LAT_WIDTH / DELTA_LAT) + 1
DESTINATION_PROFILE_TEMPORAL_AGGREGATION = 3 #hours
DESTINATION_PROFILE_SPATIAL_AGGREGATION = 5 #(x, y) coordinates

# TIME PARAMETERS
GLOBAL_STATE_UPDATE_CYCLE = 60 * 5
TIMESTEP = 60
ENTERING_TIME_BUFFER = 3600*4
