from simulator.services.routing_service import RoutingEngine
from config.hex_setting import OFF_DURATION,DIM_OF_RELOCATION
from novelties import agent_codes, status_codes
from simulator.models.customer.customer import Customer
from simulator.models.customer.request import request
from collections import defaultdict
import numpy as np
import contextlib

@contextlib.contextmanager
def local_seed(seed):
    # this defines a local random seed funciton, and let the simulator to resume previous random seed
    state = np.random.get_state()
    np.random.seed(seed)  # set seed
    try:
        yield
    finally:
        np.random.set_state(state)  # put the state back on


class hex_zone:
    def __init__(self,hex_id,coord,coord_list, match_zone,neighbors, charging_station_ids, charging_coords,od_split, trip_time, t_unit, epoch_length):
        '''
        hex_id: id of the hexagon zone in the shapefile
        coord: lon and lat values
        arrival_rate: number of arrivals per tick
        neighbors: adjacent hexagons' ids
        charging_station_ids: nearest 5 charging station ids
        charging_coords: list of coordinates of the 5 charging stations
        epoch_length: total ticks per epoch of simulation
        '''
        self.hex_id = hex_id
        self.match_zone_id=match_zone
        self.lon, self.lat = coord
        self.coord_list=coord_list #this is the list for all the lon lat coordinates of the hexagons
        od_split=np.reshape(od_split, (od_split.shape[0],od_split.shape[-1]))
        trip_time = np.reshape(trip_time, (trip_time.shape[0], trip_time.shape[-1]))  #remove one of the dimension
        self.arrival_rate=np.sum(od_split,axis=-1).flatten()/t_unit #now this becomes a  hour by 1 array,and we convert this to each tick of demand!
        
        # 1 by N matrix
        self.od_ratio=od_split
        self.trip_time=trip_time

        # the following two defines the actions
        self.neighbor_hex_id = neighbors # length may vary
        self.nearest_cs = charging_station_ids

        self.passengers = defaultdict()
        self.vehicles = defaultdict()
        self.served_num = 0
        self.removed_passengers = 0 
        self.veh_waiting_time = 0
        self.rands=[] #the set of random arrivals generated
        self.served_id=[]
        self.total_pass=0 #this also servers as passenger id

        self.t_unit=t_unit # number of ticks per hour
        self.epoch_length=epoch_length
        self.q_network = None
        self.routing_engine = RoutingEngine.create_engine()
        self.cs_loc  =charging_coords
        self.cumsum_narrivals = 0
        #initialize the demand for each hexagon zone
        self.init_demand()
    
    def get_h_transitions(self):
        non_dummy=[veh for veh in self.vehicles.values() if veh.state.agent_type!=agent_codes.dummy_agent]
        transitions = []
        for vehicle in non_dummy:
            if len(vehicle.get_transitions())>0:
                state, action, next_state, reward,flag = vehicle.get_transitions()[-1]
                transitions.append([state, action, next_state, reward,flag])
        return transitions

    def init_demand(self):
        '''
        todo: generate all the initial demand for each hour. Fix a local random generator to reduce randomness
        :param simulation_length:
        :return:
        '''
        #copy the arrival rate list multiple times!
        with local_seed(self.hex_id):
            self.arrivals = np.random.poisson(list(self.arrival_rate)*int(max(1,np.ceil(self.epoch_length/len(self.arrival_rate)/self.t_unit))),\
                                              size=(self.t_unit, len(self.arrival_rate)*int(max(1,np.ceil(self.epoch_length/len(self.arrival_rate)/self.t_unit)))))
            self.arrivals=self.arrivals.flatten('F') #flatten by columns-major
            self.arrivals=list(self.arrivals)


    def add_veh(self,veh): # vehicle is an object
        '''
        add and remove vehicles by its id
        id contained in veh.state
        :param veh:
        :return:
        '''
        self.vehicles[veh.state.vehicle_id]=veh

    def remove_veh(self,veh):
        self.vehicles.pop(veh.state.vehicle_id) #remove the vehicle from the list

    def demand_generation(self,tick): #the arrival of passenger demand
        '''
        todo 1: store 60 to config
        todo 2:[done] complete the demand generation part with travel time and od split ratio
        :param tick: current time
        :return:
        '''
        with local_seed(tick): #fix the random seed
            hour=tick//(self.t_unit*60)%24 #convert into the corresponding hours. Tick are seconds and is incremeted by 60 seconds in each iteration
            # print('hour {}  tick{}'.format(hour, tick))
            narrivals=self.arrivals.pop() #number of arrivals
            self.cumsum_narrivals += narrivals
            destination_rate = self.od_ratio[hour,:]
            if narrivals>0 and sum(destination_rate)>0:
                # print('Tick {} hour {} and tunit{}'.format(tick,hour,self.t_unit))
                destination_rate=destination_rate/sum(destination_rate) #normalize to sum =1
                #lets generate some random des
                destinations=np.random.choice(np.arange(destination_rate.shape[-1]),p=destination_rate,size=narrivals) #choose the destinations
                for i in range(narrivals):
                    # r={'id':self.total_pass,'origin_id':self.hex_id, 'origin_lat':self.lat, 'origin_lon':self.lon, \
                    #    'destination_id':destinations[i], 'destination_lat':self.coord_list[destinations[i]][1], 'destination_lon':self.coord_list[destinations[i]][0], \
                    #        'trip_time':self.trip_time[hour,destinations[i]],'request_time':tick}
                    #r=request(self.total_pass, self.hex_id, (self.lon,self.lat,), destinations[i], self.coord_list[destinations[i]],self.trip_time[hour,destinations[i]],tick)
                    self.passengers[(self.hex_id,self.total_pass)]=Customer(request(self.total_pass, self.hex_id, (self.lon,self.lat,), destinations[i], self.coord_list[destinations[i]],self.trip_time[hour,destinations[i]],tick)) #hex_id and pass_id create a unique passenger identifier
                    self.total_pass+=1

        return

    def demand_generation_async(self,tick): #the arrival of passenger demand
        '''
        this function used for parallel purposes
        :param tick: current time
        :return:
        '''
        newpass=defaultdict()
        with local_seed(tick): #fix the random seed
            hour=tick//(self.t_unit*60)%24 #convert into the corresponding hours. Tick are seconds and is incremeted by 60 seconds in each iteration
            # print('hour {}  tick{}'.format(hour, tick))
            narrivals=self.arrivals.pop() #number of arrivals
            destination_rate = self.od_ratio[hour,:]
            if narrivals>0 and sum(destination_rate)>0:
                # print('Tick {} hour {} and tunit{}'.format(tick,hour,self.t_unit))
                destination_rate=destination_rate/sum(destination_rate) #normalize to sum =1
                #lets generate some random des
                destinations=np.random.choice(np.arange(destination_rate.shape[-1]),p=destination_rate,size=narrivals) #choose the destinations
                #request has the information of : id, origin_lon, origin_lat, destination_lon,destination_lat, trip_time
                for i in range(narrivals):
                    # r={'id':self.total_pass,'origin_id':self.hex_id, 'origin_lat':self.lat, 'origin_lon':self.lon, \
                    #    'destination_id':destinations[i], 'destination_lat':self.coord_list[destinations[i]][1], 'destination_lon':self.coord_list[destinations[i]][0], \
                    #        'trip_time':self.trip_time[hour,destinations[i]],'request_time':tick}
                    #r=request(self.total_pass, self.hex_id, (self.lon,self.lat,), destinations[i], self.coord_list[destinations[i]],self.trip_time[hour,destinations[i]],tick)
                    newpass[(self.hex_id,self.total_pass)]=Customer(request(self.total_pass, self.hex_id, (self.lon,self.lat,), destinations[i], self.coord_list[destinations[i]],self.trip_time[hour,destinations[i]],tick)) #hex_id and pass_id create a unique passenger identifier
                    self.total_pass+=1

        return newpass



    def remove_pass(self,pids): #remove passengers
        '''
        Remove passengers by key_id
        :param p:
        :return:
        '''
        [self.passengers.pop(pid) for pid in pids]


    def update_passengers(self):
        '''
        code for updating the passenger status / or remove them if picked up
        '''
        '''
        todo: change the following customer picked up code. right now is hard coded: Remove when picked up
        '''
        remove_ids=[]
        for pid in self.passengers.keys():
            if self.passengers[pid].status>1 or self.passengers[pid].waiting_time>=self.passengers[pid].max_tolerate_delay: #remove passengers after 10 ticks.
                remove_ids.append(pid)
            else:
                self.passengers[pid].waiting_time+=self.t_unit #update waiting time
        self.removed_passengers+=len(remove_ids)
        self.remove_pass(remove_ids)


    def vehicle_dispatch(self,tick):
        '''
        Dispatch the vehicles. This step follows from matching step
        :param tick:
        :return:
        '''
        if len(self.vehicles.keys()) == 0:
            #no vehicle to dispatch
            return
        tbd_vehicles = {key:vehicle for key,vehicle in self.vehicles.items() if vehicle.state.status == status_codes.V_TOBEDISPATCHED}
        self.dispatch(tbd_vehicles,tick)

    def dispatch(self,vehicles,current_time):
        '''
        todo: fulfill OFF_DUTY cycle: specify when to trigger OFF_Duty status 
        :vehicles: is dict with key and values
        '''
        for vehicle in vehicles.values():
            # for test
            # vehicle.state.dispatch_action_id = 0
            action_id = vehicle.state.dispatch_action_id; offduty = 0 # actions are attached before implementing dispatch
            if offduty:
                off_duration = np.random.randint(OFF_DURATION /2, OFF_DURATION * 3/2) #
                # self.sample_off_duration()   #Rand time to rest
                vehicle.take_rest(off_duration)
            else:
                # Get target destination and key to cache
                target, charge_flag, target_hex_id = self.convert_action_to_destination(vehicle, action_id)
                (route, trip_time) = self.routing_engine.route_hex(vehicle.get_location(),target)
                if charge_flag == 0:
                    vehicle.cruise(route,trip_time,target_hex_id,action_id,current_time) # check if stay still
                else:
                    cid = self.nearest_cs[action_id-DIM_OF_RELOCATION]
                    # cs_repo = ChargingRepository.get_charging_station(cid)
                    vehicle.head_for_charging_station(trip_time,cid,target,route) # self.cs_loc[cid],cid, route)

    def convert_action_to_destination(self, vehicle, action_id):
        '''
        vehicle: objects
        action_id: action ids from 0-11, pre-derived from DQN 
        '''
        target = None
        valid_relocation_space = [vehicle.state.hex_id]+self.neighbor_hex_id
        try:
            target_hex_id = valid_relocation_space[action_id]
            lon,lat = self.coord_list[target_hex_id]
            charge_flag = 0
        except IndexError:
            target_hex_id = None # dispatch to charging station do not need it.
            lon,lat = self.cs_loc[self.nearest_cs[action_id-DIM_OF_RELOCATION]] 
            charge_flag = 1
        target = (lon,lat)

        return target,charge_flag, target_hex_id
        
    def get_num_arrivals(self):
        return self.cumsum_narrivals
    def get_removed_passengers(self):
        return self.removed_passengers

 