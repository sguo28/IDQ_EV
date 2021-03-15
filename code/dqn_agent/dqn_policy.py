import pickle
import os
from simulator.models.zone.hex_zone import hex_zone
import numpy as np
from collections import OrderedDict, defaultdict, deque
from numpy.core.fromnumeric import mean
from scipy.spatial.kdtree import KDTree
from simulator.settings import FLAGS
from config.settings import GLOBAL_STATE_UPDATE_CYCLE, MIN_DISPATCH_CYCLE
from dqn_agent.q_network import DeepQNetwork
from dummy_agent.dispatch_policy import DispatchPolicy
from .dqn_config import setting
from common.time_utils import get_local_datetime
from common import geoutils, mesh
from novelties import status_codes
from simulator.models.vehicle.vehicle_repository import VehicleRepository
import geopandas as gpd
class DQNDispatchPolicy(DispatchPolicy):
    '''
    feature_constructor: update demand supply patterns
    '''
    def __init__(self):
        super().__init__()
        # self.feature_constructor = FeatureConstructor()
        # self.q_cache = {}
        cs_poly = gpd.read_file('data/NYC_shapefiles/selected_cs.shp')
        self.cs_loc = cs_poly[["lon","lat"]]
        self.charging_station_KDTree = KDTree(self.cs_loc[["lon","lat"]])
        self.q_network = DeepQNetwork()
    # Overriding the parent function in dummy_agent.dispatch_policy
    def get_dispatch_decisions(self, tbd_vehicles,current_time):
        dispatch_commands = []
        for vehicle_id, vehicle_state in tbd_vehicles.iterrows():
            # Get best action for this vehicle and whether it will be offduty or not
            a_id, offduty = self.predict_best_action(vehicle_id, vehicle_state,current_time)
            if offduty:
                command = self.create_dispatch_dict(vehicle_id, offduty=True,action=a_id)
            else:
                # Get target destination and key to cache
                target, cache_key = self.convert_action_to_destination(vehicle_state, self.q_network.get_action_space()[a_id])
                # create dispatch dictionary with the given attribute
                if target is None:
                    continue
                if cache_key is None:
                    command = self.create_dispatch_dict(vehicle_id, target,action=a_id)
                else:
                    command = self.create_dispatch_dict(vehicle_id, cache_key=cache_key,action=a_id)
            dispatch_commands.append(command)
        return dispatch_commands

    # Return best action for this vehicle given its state, and returns whether it will be Offduty or not
    def predict_best_action(self, vehicle_id, vehicle_state,current_time):
        
        # if vehicle_state.idle_duration >= MIN_DISPATCH_CYCLE and FLAGS.offduty_probability > np.random.random():
        #     a, offduty = (0, 0), 1
        
        if self.q_network is None:
            action, offduty = 0, 0
        else:
            # print(vehicle_state.vehicle_id)
            state_rep = [current_time,vehicle_state.vehicle_id, vehicle_state.hex_id,vehicle_state.SOC] # vehicle_state.assigned_hex.get_nearest_cs(), get_cs_waiting_time
            
            aidx = self.q_network.get_action(state=state_rep)

            offduty = 0
        return aidx, offduty

    # Get the destination from dispatched vehicles
    def convert_action_to_destination(self, vehicle_state, action):
        cache_key = None
        target = None
        hex_repo = hex_zone_collection[vehicle_state.hex_id]
        try:
            ax,ay = [action_id] # ax, ay = action  # Action from action space matrix
            lon, lat = mesh.convert_xy_to_lonlat(x + ax, y + ay)
        except TypeError:
            dist_cs,id_cs = self.charging_station_KDTree.query([vehicle_state.lon,vehicle_state.lat],k=5)
            lon,lat = self.cs_loc.loc[id_cs[action],['lon','lat']].values
        if lon == vehicle_state.lon and lat == vehicle_state.lat:
            pass
        elif FLAGS.use_osrm and mesh.convert_xy_to_lonlat(x, y) == (lon, lat):
            cache_key = ((x, y), (ax, ay))  # Create cache key with location associated with action
        else:
            target = (lat, lon)

        return target, cache_key

'''    def convert_action_to_hex_id(self,action_index,vehicle_state):
        # action id corresponds to:
        # state.current_hex = [0]
        # state.neighbour_hexs = [1,2,3,4,5,6]
        # state.nearest_cs = [7,8,9,10,11]
        current_hex_id = vehicle_state.assgined_hex.get_hex_id() # get veh's assgined hex (object), get_hex_id is a function of the hex object
        target_hex_id = [current_hex_id,vehicle_state.assgined_hex.get_neighbour_hexs(current_hex_id)+vehicle_state.assgined_hex.get_nearest_cs(current_hex_id)][action_index]
        return target_hex_id'''



# Learner that uses experience memory and learned previous models
class DQNDispatchPolicyLearner(DQNDispatchPolicy):
    def __init__(self):
        super().__init__()

    def predict_best_action(self, vehicle_id, vehicle_state,current_time):
        a, offduty = super().predict_best_action(vehicle_id,vehicle_state,current_time)
        return a, offduty


    def dispatch(self, current_time, vehicles,f):
        # self.give_rewards(vehicles)
        dispatch_commands = super().dispatch(current_time, vehicles,f)
        # self.backup_supply_demand()
        reward_list =deque([0],maxlen=100)
        for vehicle_id, row in vehicles.iterrows():
            veh_state = VehicleRepository.get(vehicle_id)
            if len(veh_state.get_transitions())>0:
                state, action, next_state, reward,flag = veh_state.get_transitions()[-1]
                self.q_network.memory.push(state, action, next_state, reward,flag)
                reward_list.append(reward)
        
        # If size exceeded, run training
        self.q_network.train()
        start_time = FLAGS.start_time + int(60 * 60 * 24 * FLAGS.start_offset)
        
        f.writelines('{},{}\n'.format((current_time-start_time)//60,mean(reward_list)))
        print((current_time-start_time)//60,mean(reward_list))
            # print("iterations : {}, average_loss : {:.3f}, average_q_max : {:.3f}".format(
            #     self.q_network.n_steps, average_loss, average_q_max), flush=True)
            # self.q_network.write_summary(loss)
        return dispatch_commands