import numpy as np
from ray import state
from .models.vehicle.vehicle_repository import VehicleRepository
from .models.customer.customer_repository import CustomerRepository
from .models.charging_pile.charging_pile_repository import ChargingRepository
from .services.demand_generation_service import DemandGenerator
from .services.routing_service import RoutingEngine
from common import mesh
from logger import sim_logger
from dummy_agent.demand_loader import DemandLoader
from common.time_utils import get_local_datetime
from simulator.settings import FLAGS, NUM_NEAREST_CS
from logger import sim_logger
from logging import getLogger
from .models.vehicle.vehicle_state import VehicleState
from .models.vehicle.vehicle import Vehicle
from .models.zone.matching_zone import matching_zone
from .models.zone.hex_zone import hex_zone
from novelties import agent_codes, status_codes
from random import randrange
import geopandas as gpd
import time
from scipy.spatial import KDTree
from pathos.pools import ThreadPool  #this is for parallel
from config.hex_setting import NUM_REACHABLE_HEX,NUM_NEAREST_CS, MAP_HEIGHT, MAP_WIDTH,ENTERING_TIME_BUFFER
import copy
import pickle
import ray
from dqn_agent.dqn_agent import DeepQNetwork


class Simulator(object):
    def __init__(self, start_time, timestep):
        self.reset(start_time, timestep)
        self.last_vehicle_id =1
        self.vehicle_queue = []
        sim_logger.setup_logging(self)
        self.logger = getLogger(__name__)
        self.demand_generator = DemandGenerator()
        self.routing_engine = RoutingEngine.create_engine()
        self.route_cache = {}
        self.current_dummyV = 0
        self.current_dqnV = 0
        #containers as dictionaries
        self.match_zone_collection=[]
        self.hex_zone_collection={}
        self.vehicle_collection={}
        # DQN for getting actions and dumping transitions
        self.dqn_network = DeepQNetwork()
        self.transition_batch = []
        self.n_matches = 0
        #initialize ray
        ray.init()

    def reset(self, start_time=None, timestep=None):
        '''
        todo: init charging stations? [done]
        '''
        if start_time is not None:
            self.__t = start_time
            self.start_time=start_time
        if timestep is not None:
            self.__dt = timestep
        VehicleRepository.init()
        CustomerRepository.init()
        ChargingRepository.init()
        

    def process_trip(self,filename):
        '''
        todo: make the following variables (hours, nhex,nhex) as input or some global vars
        :param filenmame:
        :return:
        '''
        nhex = NUM_REACHABLE_HEX
        #process the line based file into a hour by
        data=np.zeros((24, nhex,nhex))
        with open(filename,'r') as f:
            next(f)
            for lines in f:
                line=lines.strip().split(',')
                h,o,d,t=line[1:] # hour, oridin, dest, trip_time/num of trip
                data[int(h),int(o),int(d)]=float(t)
        return data


    def init_zones(self,file_hex,file_charging, trip_file, travel_time_file, n_nearest=NUM_NEAREST_CS):
        '''
        todo: finalize the location of each file and some simulation setting in a config file
        :param file_hex:
        :param file_charging:
        :param trip_file:
        :param travel_time_file:
        :param n_nearest:
        :return:
        '''
        df=gpd.read_file(file_hex) # tagged_cluster_hex './data/NYC_shapefiles/reachable_hexes.shp'
        charging_stations=gpd.read_file(file_charging) # point geometry # 'data/NYC_shapefiles/processed_cs.shp'
        self.charging_kdtree=KDTree(charging_stations[['Longitude','Latitude']]) 
        self.hex_kdtree=KDTree(df[['lon','lat']])

        matchzones=np.unique(df['cluster_la'])

        hex_ids=df.index.to_numpy()
        print('Number of total hexagons:', len(hex_ids))

        hex_coords=df[['lon','lat']].to_numpy() # coord
        hex_to_match=df['cluster_la'].to_numpy() # corresponded match zone id

        demand= self.process_trip(trip_file) # 'data/trip_od_hex.csv' 'data/trip_time_od_hex.csv'
        travel_time=self.process_trip(travel_time_file) # 

        epoch_length= 60*24*7 #this is the total number of ticks set for simulation, change this value.'
        t_unit=60 #number of time steps per hour

        #we initiaze the set of hexagone zones first
        for h_idx,coords,match_id in zip(hex_ids,hex_coords,hex_to_match):
            neighbors = df[df.geometry.touches(df.geometry[h_idx])].index.tolist() # len from 0 to 6
            _,charging_idx=self.charging_kdtree.query(coords, k=n_nearest) # charging station id
            charging_coords=charging_stations.loc[charging_idx,['Longitude','Latitude']]
            self.hex_zone_collection[h_idx]=hex_zone(h_idx,coords,hex_coords,match_id,neighbors,charging_idx,charging_coords,demand[:,h_idx,:],travel_time[:,h_idx,:],t_unit, epoch_length)


        #ray init hex, try this
        hex_collects=[]
        for m_idx in matchzones:
            h_ids = df[df['cluster_la'] == m_idx].index.tolist()
            hex_collects.append([self.hex_zone_collection[hid] for hid in h_ids])

        #we initialize the matching zones through ray
        self.match_to_hex={} # a local map of hexagones to matching zones
        for idx,hexs in zip(matchzones,hex_collects):
            self.match_zone_collection.append(matching_zone.remote(idx,hexs)) #ray object on the remote side
            self.match_to_hex[idx]=hexs  #a local container
        print('ray initialize match zone compelte')

        # for i in range(1000):
        #     t1=time.time()
        #     ray.get([c.async_demand_gen.remote(i) for c in self.match_zone_collection])
        #     ll=ray.get(self.match_zone_collection[0].get_arrivals_length.remote())
        #     print('tick {} time for demand gen {} arrival length {}'.format(i,time.time()-t1,ll))


        #we then initilize the matching zones
        # for m_idx in matchzones:
        #     h_ids=df[df['cluster_la']==m_idx].index.tolist()
        #     self.matching_zone_collection[m_idx]=matching_zone(m_idx,[self.hex_zone_collection[hid] for hid in h_ids])

    def par_step(self): #we use parallel update to call the step function.
        '''
        Parallel run of the simulator that involves the following key steps:
        1. conduct the matching for each matching zone
        2. Update passenger status
        3. Update vehicle status
        4. Dispatch vehicles
        5. Generate new passengers
        :param tick:
        :return:
        '''
        #conduct matching first
        tick=self.__t-self.start_time
        t_start=time.time()

        ray.get([m.match.remote(tick) for m in self.match_zone_collection]) #force this to complete

        #update passenger status
        [m.update_passengers.remote() for m in self.match_zone_collection]

        #update vehicle status
        t1=time.time()
        self.download_vehicles()

        self.update_vehicles()
        '''
        todo: complete the vehicle enter function 
        '''
        self.enter_market()
        '''
        todo: first match, then dispatch
        '''
        
        self.attach_dispatch_actions(tick)
        
        self.push_vehicles()
        t_v_update=time.time()-t1

        #vehicle dispatch
        [m.dispatch.remote(tick) for m in self.match_zone_collection]

        #update the demand for each matching zone
        ray.get([c.async_demand_gen.remote(tick) for c in self.match_zone_collection])

        '''
        todo: add a download and push function for charging stations. 
        '''
        for cs in ChargingRepository.get_all(): # get all charging stations
            cs.step(self.__dt,self.__t) # update charging piles in stations, e.g., available to occupied

        self.store_transition_batch()

        self.__update_time()
        if self.__t % 3600 == 0:
            self.logger.info("Elapsed : {}".format(get_local_datetime(self.__t)))
        t_end=time.time()-t_start

        print('Iteration {} completed, total cpu time={}, time for vehicle status update={}'.format(tick/60,t_end,t_v_update))
        #finally identify new vehicles, and update location of existing vehicles
        #the results is a list of list of dictionaries.

    def download_vehicles(self):
        #copy remote information to local
        #ray return read-only items. we need to make a deep copy first

        # copy_veh=ujson.dumps(ray.get([m.get_vehicles_by_hex.remote() for m in self.match_zone_collection]))
        all_vehs=ray.get([m.get_vehicles_by_hex.remote() for m in self.match_zone_collection])
        # self.n_matches+=ray.get([m.get_num_of_matches().remote() for m in self.match_zone_collection])
        for mid,vehs in zip(self.match_to_hex.keys(),all_vehs):
            for hexs,veh in zip(self.match_to_hex[mid],vehs):
                hexs.vehicles=veh #make a hard copy

    def update_vehicles(self):
        '''
        loop through all hexagones and update the vehicle status
        :return:
        '''
        for hex in self.hex_zone_collection.values():
            [vehicle.step(self.__dt,self.__t,self.hex_zone_collection) for vehicle in hex.vehicles.values()]

    def summarize_metrics(self):
        '''
        get metrics of all DQN vehicles
        '''
        all_vehicles = [vehicle for hex in self.hex_zone_collection.values() for vehicle in hex.vehicles.values() if vehicle.state.agent_type == agent_codes.dqn_agent]
        print([veh.state.status == status_codes.V_OCCUPIED for veh in all_vehicles][:20])
        num_idle = sum([veh.state.status == status_codes.V_IDLE for veh in all_vehicles])
        num_serving = sum([veh.state.status == status_codes.V_OCCUPIED for veh in all_vehicles])
        num_charging = sum([veh.state.status == status_codes.V_CHARGING for veh in all_vehicles])
        num_cruising = sum([veh.state.status == status_codes.V_CRUISING for veh in all_vehicles])
        # num_matches = sum([m.get_num_of_matches() for m in self.match_zone_collection])
        total_num_arrivals = sum([hex.get_num_arrivals() for mid in self.match_to_hex.keys() for hex in self.match_to_hex[mid]])
        total_removed_passengers = sum([hex.get_removed_passengers() for mid in self.match_to_hex.keys() for hex in self.match_to_hex[mid]])
        return num_idle, num_serving, num_charging, num_cruising,self.n_matches, total_num_arrivals,total_removed_passengers

    def attach_dispatch_actions(self,tick):
        '''
        todo: export all veh, get their states, plug the batches in DQN, get the actions, pass back to vehicles
        '''
        dqn_agents = [vehicle for hex in self.hex_zone_collection.values() for vehicle in hex.vehicles.values() if vehicle.state.agent_type == agent_codes.dqn_agent]

        if len(dqn_agents)>0:
            state_batches = [veh.dump_states(tick) for veh in dqn_agents] # (tick, veh_id, hex_id, SOC)
            num_valid_relos = [len(self.hex_zone_collection[veh.get_hex_id()].neighbor_hex_id)+1 for veh in dqn_agents]
            
            action_ids = self.dqn_network.get_actions(state_batches,num_valid_relos)
            print(action_ids)
            for veh,action_id in zip(dqn_agents,action_ids):
                veh.state.dispatch_action_id = [action_id]

    def push_vehicles(self):
        '''
        send local information to remote
        :return:
        '''
        all_vehs=[[] for _ in range(len(self.match_to_hex.keys()))] #create default length
        for mid in self.match_to_hex.keys():
            all_vehs[mid]=[hex.vehicles for hex in self.match_to_hex[mid]]
        ray.get([m.set_vehicles_by_hex.remote(vehs) for m,vehs in zip(self.match_zone_collection,all_vehs)])

    def match_zone_step_wrapper(self,zone):
        '''
        This is a wrapper to be fed to the parallel pool in each iteration
        todo:[done]  add self.current_tick somewhere in the simulator... can we use self.__t?
        '''
        tick=self.__t-self.start_time
        t1=time.time()
        zone.step(tick) #call the step function for the matching zone
        return time.time()-t1

    def thread_update(self,zones):
        '''
        this will create a parallel pool for run the step function of each matching zone in parallel
        # the update is performed in place, so no return is required
        :param zones:
        :return:
        '''
        results=self.pool.amap(self.match_zone_step_wrapper, zones)
        results=results.get()
        print('Time of different zones:', results)

    def sequential_update(self,zones):
        '''
        Perform sequential update
        :param zones:
        :return:
        '''
        times=[]
        for zone in zones:
            t=self.match_zone_step_wrapper(zone)
            times.append(t)
        print('Sequential time for each zone:', times)

        tick = self.__t - self.start_time
        t1=time.time()
        r=ray.get([mz.demand_gen_async.remote(len(mz.hex_zones),mz.hex_zones[0],tick) for mz in zones])
        print('Ray demand gen time:', time.time()-t1)
    
    def init_vehicle(self):
        '''
        todo: change locations in hex setting, also let demand loader generate coord.
        hex_coords=df[['lon','lat']].to_list()
        '''
        initial_locations = [mesh.convert_xy_to_lonlat(x, y)[::-1] for x in range(MAP_WIDTH) for y in range(MAP_HEIGHT)]
        p = DemandLoader.load_demand_profile(self.__t) # a matrix of M x N
        p = p.flatten() / p.sum()

        _,idx=self.hex_kdtree.query(initial_locations)
        vehicle_hex_ids = [idx[i] for i in np.random.choice(len(initial_locations), size=FLAGS.vehicles, p=p)]

        n_vehicles = len(vehicle_hex_ids)
        vehicle_ids = range(self.last_vehicle_id, self.last_vehicle_id + n_vehicles)
        self.last_vehicle_id += n_vehicles
        entering_time = np.random.uniform(self.__t, self.__t + ENTERING_TIME_BUFFER, n_vehicles).tolist()
        q = sorted(zip(entering_time, vehicle_ids, vehicle_hex_ids))
        self.vehicle_queue = q

    def enter_market(self):
        while self.vehicle_queue:
            t_enter, vehicle_id, vehicle_hex_id = self.vehicle_queue[0]
            if self.__t >= t_enter:
                self.vehicle_queue.pop(0) # no longer queueing
                self.populate_vehicle(vehicle_id, vehicle_hex_id)
            else:
                break

    def populate_vehicle(self, vehicle_id, vehicle_hex_ids):
        agent_type = 0
        r = randrange(2)
        if r == 0 and self.current_dummyV < FLAGS.dummy_vehicles:
            agent_type = agent_codes.dummy_agent
            self.current_dummyV += 1
            
        # If r = 1 or num of dummy agent satisfied
        elif self.current_dqnV < FLAGS.dqn_vehicles:
            agent_type = agent_codes.dqn_agent
            self.current_dqnV += 1

        else:
            agent_type = agent_codes.dummy_agent
            self.current_dummyV += 1

        #correct location and match to the nearest hexagon
        # _,idx=self.hex_kdtree.query(location)
        location=(self.hex_zone_collection[vehicle_hex_ids].lon,self.hex_zone_collection[vehicle_hex_ids].lat) #update its coordinate with the centroid of the hexagon
        self.hex_zone_collection[vehicle_hex_ids].add_veh(Vehicle(VehicleState(vehicle_id,location,vehicle_hex_ids,agent_type))) #append this new available vehicle to the hexagon zone

    def __update_time(self):
        self.__t += self.__dt

    def store_transition_batch(self):
        self.transition_batch = []
        for mid in self.match_to_hex.keys():
            self.transition_batch.append([hex.get_h_transitions() for hex in self.match_to_hex[mid]])

    def dump_transition_batch(self):
        return self.transition_batch

    def get_current_time(self):
        return self.__t