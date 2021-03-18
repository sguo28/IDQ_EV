# hex configuration
import os
db_dir = "../data/db.sqlite3"
DB_HOST_PATH = ''.join(['sqlite:///', db_dir])

OSRM_HOSTPORT = os.getenv("OSRM_HOSTPORT", "localhost:5000")

DEFAULT_LOG_DIR = "../logs/tmp"

DATA_DIR = "../data"
NUM_REACHABLE_HEX = 1347
num_reachable_hex = 1347 # number of hex that has intersection with road network
charging_station_shp_path = '../data/NYC_shapefiles/cs_snap_lonlat.shp' # '../data/NYC_shapefiles/processed_cs.shp'
hex_shp_path = '../data/NYC_shapefiles/tagged_clustered_hex.shp' # '../data/NYC_shapefiles/clustered_hex.shp' #
trip_file = '../data/trip_od_hex.csv'
travel_time_file='../data/trip_time_od_hex.csv'
charging_station_data_path = '../data/cs_snap_lonlat.csv' # '../data/processed_cs_new.csv'


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
epsilon = 0.00



#
OFF_DURATION= 20*60 #20min
NUM_NEAREST_CS = 5
#weights for reward calculation
BETA_EARNING =1 
BETA_COST = 1
SOC_PENALTY = 5

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
ENTERING_TIME_BUFFER = 60 # now 60 sec; 60*60*1 #  1 hour

REJECT_TIME =30
START_OFFSET = int(0) #simulation start datetime offset (days)")
SIM_DAYS = int(5)  #simulation days")
START_TIME = int(1464753600 + 3600 * 5)  #simulation start datetime (unixtime)")


# VEHICLE 
FULL_CHARGE_PRICE = 13 # 0.26 usd/kwh *50kwh

# UNIT CONVERT
MILE_PER_METER = 0.000621371
SEC_PER_MIN = 60

DIM_OF_RELOCATION = 7
DIM_OF_CHARGING = 5

SIM_ACCELERATOR = 5 # accelerate to 5x speed: charging and consuming.
SPEED = 5 # m/s
MAX_MOVE = 1