import numpy as np
from .models.vehicle.vehicle_repository import VehicleRepository
from .models.customer.customer_repository import CustomerRepository
from .models.charging_pile.charging_pile_repository import ChargingRepository
from .services.demand_generation_service import DemandGenerator
from .services.routing_service import RoutingEngine
from common.time_utils import get_local_datetime
from config.settings import OFF_DURATION, PICKUP_DURATION
from simulator.settings import FLAGS, NUM_NEAREST_CS
from logger import sim_logger
from logging import getLogger
from .models.vehicle.vehicle_state import VehicleState
from .models.vehicle.vehicle import Vehicle
from .models.zone.matching_zone import matching_zone
from .models.zone.hex_zone import hex_zone
from novelties import agent_codes
from random import randrange
import geopandas as gpd
import time
from scipy.spatial import KDTree
from pathos.pools import ThreadPool  #this is for parallel
from config.hex_setting import  num_reachable_hex
import ray

class Simulator(object):
    def __init__(self, start_time, timestep):
        self.reset(start_time, timestep)
        sim_logger.setup_logging(self)
        self.logger = getLogger(__name__)
        self.demand_generator = DemandGenerator()
        self.routing_engine = RoutingEngine.create_engine()
        self.route_cache = {}
        self.current_dummyV = 0
        self.current_dqnV = 0 
        self.threads=12 #number of threads to use , change this value later
        self.pool = ThreadPool(self.threads) #start a thread pool as the simulator class is created
        #containers as dictionaries
        self.match_zone_collection=[]
        self.hex_zone_collection={}
        self.vehicle_collection={}
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
        nhex = num_reachable_hex
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
        print('ray initiaze match zone compelte')

        # for i in range(1000):
        #     t1=time.time()
        #     ray.get([c.async_demand_gen.remote(i) for c in counter])
        #     ll=ray.get(counter[0].get_arrivals_length.remote())
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

        [m.match.remote(tick) for m in self.match_zone_collection]

        #update passenger status
        [m.update_passengers.remote(tick) for m in self.match_zone_collection]

        #update vehicle status
        self.download_vehicles()
        self.update_vehicles()
        '''
        todo: complete the vehicle enter function 
        '''
        # self.vehicle_enter()
        self.push_vehicles()

        #vehicle dispatch
        [m.dispatch.remote(tick) for m in self.match_zone_collection]

        #update the demand for each matching zone
        ray.get([m.async_demand_gen.remote(tick) for m in self.match_zone_collection])

        for cs in ChargingRepository.get_all(): # get all charging stations
            cs.step(self.__dt,self.__t) # update charging piles in stations, e.g., available to occupied

        # t+=1
        self.__update_time()

        if self.__t % 3600 == 0:
            self.logger.info("Elapsed : {}".format(get_local_datetime(self.__t)))

        #finally identify new vehicles, and update location of existing vehicles
        #the results is a list of list of dictionaries.

    def download_vehicles(self):
        #copy remote information to local
        all_vehs=ray.get([m.get_vehicles_by_hex() for m in self.match_zone_collection])
        for mid,vehs in zip(self.match_to_hex.keys(),all_vehs):
            for hexs,veh in zip(self.match_to_hex[mid],vehs):
                hexs.vehicles=veh

    def update_vehicles(self):
        '''
        loop through all hexagones and update the vehicle status
        :return:
        '''
        for hex in self.hex_zone_collection.values():
            for vehicle in hex.vehicles.values():
                vehicle.step(self.__dt,self.__t,self.hex_zone_collection)
                if vehicle.exit_market():
                    print("Agent EXIT", vehicle.state.status)
                    score = ','.join(map(str, [self.get_current_time(), vehicle.get_id()] + vehicle.get_score()))
                    if vehicle.get_agent_type() == agent_codes.dqn_agent:
                        self.current_dqnV -= 1
                    else:
                        self.current_dummyV -= 1
                    sim_logger.log_score(score)
                    vehicle.delete() #remove the vehicle


    def push_vehicles(self):
        '''
        send local information to remote
        :return:
        '''
        all_vehs=[]
        for mid in self.match_to_hex.keys():
            all_vehs.append([hex.vehicles for hex in self.match_to_hex[mid]])
        ray.get([m.set_vehicles_by_hex(vehs) for m,vehs in zip(self.match_zone_collection,all_vehs)])

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

    def populate_vehicle(self, vehicle_id, location):
        type = 0
        r = randrange(2)
        if r == 0 and self.current_dummyV < FLAGS.dummy_vehicles:
            type = agent_codes.dummy_agent
            self.current_dummyV += 1
            
        # If r = 1 or num of dummy agent satisfied
        elif self.current_dqnV < FLAGS.dqn_vehicles:
            type = agent_codes.dqn_agent
            self.current_dqnV += 1

        else:
            type = agent_codes.dummy_agent
            self.current_dummyV += 1

        #correct location and match to the nearest hexagon
        _,idx=self.hex_kdtree.query(location)
        location=(self.hex_zone_collection[idx].lon,self.hex_zone_collection[idx].lat) #update its coordinate with the centroid of the hexagon
        self.hex_zone_collection[idx].add_veh(Vehicle(VehicleState(vehicle_id,location,idx,type))) #append this new available vehicle to the hexagon zone
        self.push_vehicles()

    def step(self):
        '''
        todo: write a code to track vehicle who are not currently available and update their status?
        todo: 1. Change vehicle and passenger container to dictionary in each hexagon areas
        todo: 2. in vehicle_step, add the function to call hexagons and update its location
        :return:
        '''
        for vehicle in VehicleRepository.get_all():
            vehicle.step(self.__dt,self.__t,self.hex_zone_collection)
            # vehicle.print_vehicle()
            if vehicle.exit_market():
                print("Agent EXIT",vehicle.state.status)
                score = ','.join(map(str, [self.get_current_time(), vehicle.get_id()] + vehicle.get_score()))
                if vehicle.get_agent_type() == agent_codes.dqn_agent:
                    self.current_dqnV -= 1
                else:
                    self.current_dummyV -= 1
                sim_logger.log_score(score)
                VehicleRepository.delete(vehicle.get_id())
        #update vehicle and passenger status

        t1=time.time()
        self.thread_update(list(self.matching_zone_collection.values()))
        print('all matching update: {}'.format(time.time() - t1))

        for cs in ChargingRepository.get_all(): # get all charging stations
            cs.step(self.__dt,self.__t) # update charging piles in stations, e.g., available to occupied

        # t+=1
        self.__update_time()

        if self.__t % 3600 == 0:
            self.logger.info("Elapsed : {}".format(get_local_datetime(self.__t)))

    def match_vehicles(self, m_commands,c_commands, dqn_agent, dummy_agent):
        '''
        interpret and implement matching commands/charging commands
        '''
        # print("M: ", commands)
        vehicle_list = []
        rejected_requests = []
        accepted_commands = {}
        reject_count = 0
        # Comamnd is a dictionary created in dummy_agent
        for command in m_commands:
            rejected_flag = 0
            # print(command["vehicle_id"], command["customer_id"])
            vehicle = VehicleRepository.get(command["vehicle_id"])
            if vehicle is None:
                self.logger.warning("Invalid Vehicle id")
                continue
            customer = CustomerRepository.get(command["customer_id"])
            if customer is None:
                self.logger.warning("Invalid Customer id")
                continue

            triptime = command["duration"]
            # vid = command["vehicle_id"]
            # print("Maching: Vehicle " + vehicle.to_string() + " ---> " + customer.to_string())

            # price_response = command["init_price"]
            
            # customer.accepted_price = command["init_price"]
            x,y = customer.get_origin_lonlat()
            vehicle.head_for_customer((y,x), triptime, customer.get_id(), command["distance"])
            customer.wait_for_vehicle(triptime)
            v = VehicleRepository.get(command["vehicle_id"])
            v.state.current_capacity += 1 #? why not minus 1
            accepted_commands = m_commands
        # print("M: ", commands)
        
        charging_commands = {}
        reject_count = 0
        # Comamnd is a dictionary created in dummy_agent
        for command in c_commands:
            rejected_flag = 0
            vehicle = VehicleRepository.get(command["vehicle_id"])
            if vehicle is None:
                self.logger.warning("Invalid Vehicle id")
                continue
            charging_station_repo = ChargingRepository.get_charging_station(command["customer_id"]) # charging_station ID
            if charging_station_repo is None:
                # print("CS IS NONE!!!")
                self.logger.warning("Invalid charging_station id")
                continue
            triptime = command["duration"]
            vid = command["vehicle_id"]
            # print("Maching: Vehicle " + vehicle.to_string() + " --> " + customer.to_string())
            # print(charging_pile,charging_pile.index.values[0])
            # price_response = command["init_price"] # cost to travel to charging station
            cs_loc = charging_station_repo.get_cs_location()
            vehicle.head_for_charging_station(cs_loc, triptime,charging_station_repo, command["distance"])
            # charging_station.add_cruising_veh(vehicle,triptime)
            # v = VehicleRepository.get(vid)
            # v.state.current_capacity += 1 #? why not minus 1
            charging_commands = c_commands

        return rejected_requests, accepted_commands, charging_commands


    def dispatch_vehicles(self, commands):
        # print("D: ", commands)
        od_pairs = []
        vehicles = []
        # Comamnd is a dictionary created in dummy_agent
        for command in commands:
            vehicle = VehicleRepository.get(command["vehicle_id"])
            if vehicle is None:
                self.logger.warning("Invalid Vehicle id")
                continue

            if "offduty" in command:
                off_duration = self.sample_off_duration()   #Rand time to rest
                vehicle.take_rest(off_duration)
            elif "cache_key" in command:
                l, a = command["cache_key"]
                route, triptime = self.routing_engine.get_route_cache(l, a)
                vehicle.cruise(route, triptime)
            else:
                vehicles.append(vehicle)
                od_pairs.append((vehicle.get_location(), command["destination"]))

        routes = self.routing_engine.route(od_pairs)

        for vehicle, (route, triptime) in zip(vehicles, routes):
            if triptime == 0:
                continue
            vehicle.cruise(route, triptime)

    def __update_time(self):
        self.__t += self.__dt

    def __populate_new_customers(self):
        new_customers = self.demand_generator.generate(self.__t, self.__dt)
        CustomerRepository.update_customers(new_customers)

    def sample_off_duration(self):
        return np.random.randint(OFF_DURATION / 2, OFF_DURATION * 3 / 2)

    def sample_pickup_duration(self):
        return np.random.exponential(PICKUP_DURATION)

    def get_current_time(self):
        t = self.__t
        return t

    def get_new_requests(self):
        return CustomerRepository.get_new_requests()

    def get_vehicles_state(self):
        return VehicleRepository.get_states()

    def get_charging_stations(self):
        return ChargingRepository.get_all()

    def get_vehicles(self):
        return VehicleRepository.get_all()

    def get_customers(self):
        return CustomerRepository.get_all()
