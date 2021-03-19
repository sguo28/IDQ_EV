from collections import deque
from novelties import status_codes
from common import geoutils
from .vehicle_state import VehicleState
from .vehicle_behavior import Occupied, Cruising, Idle, Assigned, OffDuty, Waytocharge, Waitpile, Charging, Tobedispatched
from logger import sim_logger
from logging import getLogger
import numpy as np
from simulator.settings import SUPERCHARGING_PRICE ,SUPERCHARGING_TIME,FLAGS,PENALTY_CHARGING_TIME
from common.geoutils import great_circle_distance
from config.hex_setting import SOC_PENALTY,MILE_PER_METER, SIM_ACCELERATOR , BETA_COST, BETA_EARNING

class Vehicle(object):
    behavior_models = {
        status_codes.V_IDLE: Idle(),
        status_codes.V_CRUISING: Cruising(),
        status_codes.V_OCCUPIED: Occupied(),
        status_codes.V_ASSIGNED: Assigned(),
        status_codes.V_OFF_DUTY: OffDuty(),
        status_codes.V_WAYTOCHARGE: Waytocharge(),
        status_codes.V_CHARGING: Charging(),
        status_codes.V_WAITPILE: Waitpile(),
        status_codes.V_TOBEDISPATCHED: Tobedispatched()
    }

    def __init__(self, vehicle_state):
        if not isinstance(vehicle_state, VehicleState):
            raise ValueError
        self.state = vehicle_state
        self.customer=None # the passenger matched and to be picked up
        self.__behavior = self.behavior_models[vehicle_state.status]
        self.__customers = []       # A vehicle can have a list of cusotmers
        self.__customers_ids = []
        self.__charging_station = []
        self.__route_plan = []
        self.earnings = 0
        self.working_time = 0
        # self.first_dispatched = 0
        self.pickup_time = 0
        self.q_action_dict = {}
        self.duration = [0]*len(self.behavior_models)     # Duration for each state
        self.charging_wait = 0
        self.rb_state=[0,0,0,0]
        self.rb_action= 0
        self.rb_next_state=[0,0,0,0]
        self.rb_reward=0
        self.recent_transitions = deque(maxlen=10)
        self.flag = 0

    # update the information of vehicles, e.g. change the current hex zone location, this is not paralleled
    def update_info(self,timestep, hex,routes):
        self.working_time += timestep

        if self.state.need_route==True:
            self.state.route=routes[(self.state.hex_id,self.state.current_hex)]['route']
            self.state.time_to_destination=routes[(self.state.hex_id,self.state.current_hex)]['duration']
            self.state.travel_dist=routes[(self.state.hex_id,self.state.current_hex)]['distance']
            self.state.need_route=False
            self.state.need_interpolate=True #ask the remote to interpolate the coords


        # if self.state.current_hex!=self.state.hex_id:
        if self.state.current_hex!=self.state.hex_id:
            #update each vehicle if the new location (current_hex) is different from its current one (hex_id)
            hex.remove_veh(self)
            hex.add_veh(self)
            self.state.hex_id=self.state.current_hex

    def step(self,timestep,tick):
        '''
        step function for vehicle
        :param timestep:
        :param tick:
        :return:
        '''
        if self.state.need_interpolate==True: #make interpolation of the ticks, distance and coordinates
            self.location_interp(timestep)
            self.state.need_interpolate=False

        try:
            self.__behavior.step(self,timestep=timestep, tick=tick)
        except:
            logger = getLogger(__name__)
            logger.error(self.state.to_msg())
            raise
        if self.__behavior.available:
            self.state.idle_duration += timestep
        else:
            self.state.idle_duration = 0

        if self.state.status == status_codes.V_IDLE:
            self.duration[status_codes.V_IDLE] += timestep


    def dump_states(self,tick):
        state_rep = [tick,self.state.vehicle_id,self.state.hex_id,self.state.status==status_codes.V_OFF_DUTY,self.state.SOC]
        return state_rep

    def compute_speed(self, route, triptime):
        lons, lats = zip(*route)
        distance = geoutils.great_circle_distance(lons[:-1], lats[:-1], lons[1:], lats[1:])     # Distance in meters
        speed = sum(distance) / triptime
        self.state.travel_dist += sum(distance)
        self.state.SOC -= sum(distance)*MILE_PER_METER/self.get_mile_of_range() # meter to mile
        return speed

    def compute_fuel_consumption(self): # we calculate charging cost now
        return float(self.state.travel_dist * self.state.full_charge_price / (self.state.mile_of_range))
    
    def compute_charging_cost(self,trip_distance): # 30 min for 80% range
        return float(trip_distance*MILE_PER_METER*SIM_ACCELERATOR  * (SUPERCHARGING_PRICE * SUPERCHARGING_TIME/(self.state.mile_of_range)))

    def compute_profit(self):
        cost = self.compute_fuel_consumption() # there was "()"
        return self.earnings - cost

    def get_transitions(self):
        return self.recent_transitions

    def get_mile_of_range(self):
        return self.state.set_range()
        
    def get_SOC(self):
        return self.state.SOC

    def get_target_SOC(self):
        return self.state.target_SOC

    def send_to_dispatching_pool(self,action_id):
        self.state.dispatch_action_id = action_id
        self.__change_to_tobedispatched()

    def location_interp(self,t_unit=60):
        '''
        Interpolate the per tick travel distance and location based on the corresponding time unit
        Input: self.state.route: a list of coordinates
        self.state.time_to_destination: a list segment travel time
        self.state.travel_dist: a list of segment travel distance
        :return:
        '''
        total_tt=sum(self.state.time_to_destination)
        cum_time=np.cumsum(self.state.time_to_destination)
        cum_dist=np.cumsum(self.state.travel_dist)
        time_ticks=[i*t_unit for i in range(1,total_tt//60+1)] #the time steps to query from per simulation tick
        if total_tt%t_unit>0:
            time_ticks.append(total_tt) #add the final step
        per_tick_dist=np.interp(time_ticks,cum_time,cum_dist)
        per_tick_dist=[per_tick_dist[0]]+np.diff(per_tick_dist).tolist()

        lons=[self.state.route[0][0][0]]+[coord[1][0] for coord in self.state.route]
        lats=[self.state.route[0][0][1]]+[coord[1][1] for coord in self.state.route]

        cum_time=cum_time.tolist()
        per_tick_lon=np.interp(time_ticks,[0]+cum_time,lons)
        per_tick_lat=np.interp(time_ticks,[0]+cum_time,lats)
        per_tick_lon=per_tick_lon.tolist()
        per_tick_lat = per_tick_lat.tolist()

        per_tick_coords=[[lon,lat] for lon,lat in zip(per_tick_lon,per_tick_lat)]

        self.state.per_tick_coords=per_tick_coords
        self.state.per_tick_dist=per_tick_dist


    def cruise(self, route, triptime,action, tick):
        '''
        current hex is inplaced to destination's hex id
        use hex_id (true current hex_id) to store in state representative
        '''
        assert self.__behavior.available
        self.rb_state = [tick,self.state.vehicle_id,self.state.hex_id,self.state.SOC]
        self.rb_action = action
        if triptime == 0:
            self.park(tick)
            return
        speed = self.compute_speed(route, triptime)
        self.__reset_plan()
        self.__set_route(route, speed)
        self.__set_destination(route[-1], triptime)
        self.__change_to_cruising()
        self.__log()

    def head_for_customer(self, destination, triptime, customer_id, distance):
        '''
        :destination: lon, lat
        '''
        assert self.__behavior.available
        # if triptime >0:
        #     self.compute_speed(route,triptime)
        self.state.SOC -= distance*MILE_PER_METER*SIM_ACCELERATOR /self.get_mile_of_range()
        self.state.travel_dist += distance
        self.__reset_plan()
        self.__set_destination(destination, triptime)
        self.state.assigned_customer_id = customer_id
        self.__customers_ids.append(customer_id)
        self.__change_to_assigned()
        self.__log()

    def head_for_charging_station(self, route, triptime, cs_id, cs_coord):
        assert self.__behavior.available
        if triptime >0:
            self.compute_speed(route,triptime)
        self.__reset_plan()
        self.__set_destination(cs_coord, triptime)
        self.state.assigned_charging_station_id = cs_id
        self.__charging_station.append(cs_id)
        self.__change_to_waytocharge()
        self.__log()

    def take_rest(self, duration):
        assert self.__behavior.available
        self.__reset_plan()
        self.state.idle_duration = 0
        self.__set_destination(self.get_location(), duration)
        self.__change_to_off_duty()
        self.__log()

    def pickup(self):
        assert self.get_location() == self.customer.get_origin_lonlat()
        self.state.current_hex = self.customer.get_destination()
        self.customer.ride_on()
        self.__customers.append(self.customer)
        customer_id = self.customer.get_id()
        self.__reset_plan() # For now we don't consider routes of occupied trip
        self.state.assigned_customer_id = customer_id
        triptime = self.customer.get_trip_duration()
        self.__set_destination(self.customer.get_destination_lonlat(), triptime)
        self.__set_pickup_time(triptime)
        self.__change_to_occupied()
        self.customer=None
        self.__log()

    def dropoff(self,tick):
        assert len(self.__customers) > 0
        customer = self.__customers.pop(0)
        customer.get_off()
        self.customer_payment = customer.make_payment(1, self.state.driver_base_per_trip)
        
        self.earnings += self.customer_payment
        x1,y1 = customer.get_origin_lonlat()
        x2,y2 = customer.get_destination_lonlat()
        
        # great circle distance need lat lon
        trip_distance = great_circle_distance(x1,y1 ,x2,y2)
        self.state.travel_dist += trip_distance
        self.state.SOC -= trip_distance*MILE_PER_METER*SIM_ACCELERATOR /self.get_mile_of_range() # meter to mile
        self.rb_next_state = [tick,self.get_id(),self.state.hex_id,self.get_SOC()]
        self.rb_reward = BETA_EARNING* self.customer_payment - BETA_COST*self.compute_charging_cost(trip_distance) - SOC_PENALTY*(1-self.get_SOC())
        self.flag = int(self.state.status == status_codes.V_OFF_DUTY)

        if self.get_SOC() <0:
            self.rb_reward -= 50 # additional penalty for running out battery
            self.state.SOC = self.state.target_SOC
            self.__set_destination(self.get_location(), PENALTY_CHARGING_TIME) # 45 min for emergency charging
            self.__change_to_cruising()
            self.recent_transitions.append((self.rb_state,self.rb_action,self.rb_next_state,self.rb_reward,self.flag))
            # self.dqn_network.dump_transitions(self.recent_transitions[-1])
            return
        self.recent_transitions.append((self.rb_state,self.rb_action,self.rb_next_state,self.rb_reward,self.flag))
        # self.dump_replay_buffer()
        # self.dqn_network.dump_transitions(self.recent_transitions[-1])
        self.state.current_capacity = 0
        self.__customers_ids = []
        self.__change_to_idle()
        self.__reset_plan()
        self.__log()
        # return customer

    def park(self,tick):
        self.rb_next_state = [tick,self.get_id(),self.state.hex_id,self.get_SOC()]
        self.rb_reward = 0
        self.flag = int(self.state.status == status_codes.V_OFF_DUTY)
        self.recent_transitions.append((self.rb_state,self.rb_action,self.rb_next_state,self.rb_reward,self.flag))
        self.__reset_plan()
        self.__change_to_idle()
        self.__log()

    def start_waitpile(self):
        self.__change_to_waitpile()
        self.__log()

    
    def start_charge(self,coord):
        self.__reset_plan()
        self.state.lon, self.state.lat = coord
        self.__change_to_charging()
        self.__log()
    
    def end_charge(self,tick,coord):
        self.state.current_capacity = 0 # current passenger on vehicle
        self.__customers_ids = []
        self.__change_to_idle()
        self.__reset_plan()
        self.state.lon, self.state.lat = coord
        self.rb_next_state = [tick,self.get_id(),self.state.hex_id,self.get_SOC()]
        self.rb_reward = -SOC_PENALTY* (1-self.get_SOC())
        self.flag = int(self.state.status == status_codes.V_OFF_DUTY)
        self.recent_transitions.append((self.rb_state,self.rb_action,self.rb_next_state,self.rb_reward,self.flag))
        self.__log()
    
    def update_location(self, location, route):
        self.state.lon, self.state.lat = location
        self.__route_plan = route

    def update_customers(self, customer):
        # customer.ride_on()
        self.__customers.append(customer)

    def update_time_to_destination(self, timestep):
        dt = min(timestep, self.state.time_to_destination)
        self.duration[self.state.status] += dt
        self.state.time_to_destination -= dt
        if self.state.time_to_destination <= 0:
            self.state.time_to_destination = 0
            self.state.lat = self.state.destination_lat
            self.state.lon = self.state.destination_lon
            return True
        else:
            return False

    # some getter methods
    def get_id(self):
        vehicle_id = self.state.vehicle_id
        return vehicle_id

    def get_hex_id(self):
        return self.state.hex_id
        
    def get_customers_ids(self):
        return self.__customers_ids

    def get_location(self):
        location = self.state.lon, self.state.lat 
        return location

    def get_destination(self):
        destination = self.state.destination_lon, self.state.destination_lat 
        return destination

    def get_speed(self):
        speed = self.state.speed
        return speed

    def get_agent_type(self):
        return self.state.agent_type

    def get_price_rates(self):
        return [self.state.price_per_travel_m, self.state.price_per_wait_min]

    def reachedCapacity(self):
        if self.state.current_capacity == self.state.max_capacity:
            return True
        else:
            return False

    def get_assigned_customer_id(self):
        customer_id = self.state.assigned_customer_id
        return customer_id
    
    def get_assigned_cs_id(self):
        return self.state.assigned_charging_station_id

    def to_string(self):
        s = str(getattr(self.state, 'id')) + " Capacity: " + str(self.state.current_capacity)
        return s

    def print_vehicle(self):
        print("\n Vehicle Info")
        for attr in self.state.__slots__:
            print(attr, " ", getattr(self.state, attr))

        print("IDS::", self.__customers_ids)
    
        # print(self.state)
        print(self.__behavior)
        for cus in self.__customers:
            cus.print_customer()
        # print(self.__route_plan)
        print("earnings", self.earnings)
        print("working_time", self.working_time)
        print("current_capacity", self.state.current_capacity)
        # print(self.duration)

    def get_route(self):
        return self.__route_plan[:]

    def get_status(self):
        return self.state.status

    def get_total_dist(self):
        return self.state.travel_dist

    def get_idle_duration(self):
        dur = self.working_time - self.duration[status_codes.V_OCCUPIED] - self.duration[status_codes.V_ASSIGNED]
        # print(self.duration)
        return dur

    def get_pickup_time(self):
        return self.pickup_time

    def get_state(self):
        state = []
        for attr in self.state.__slots__:
            state.append(getattr(self.state, attr))
        return state

    def get_score(self):
        score = [self.working_time, self.earnings] + self.duration
        return score

    def get_num_cust(self):
        return self.state.current_capacity

    def get_vehicle(self, id):
        if id == self.state.vehicle_id:
            return self

    def exit_market(self):
        return False

    def __reset_plan(self):
        self.state.reset_plan()
        self.__route_plan = []

    def __set_route(self, route, speed):
        # assert self.get_location() == route[0]
        self.__route_plan = route
        self.state.speed = speed

    def __set_destination(self, destination, triptime):
        self.state.destination_lon, self.state.destination_lat = destination
        self.state.time_to_destination = triptime

    def __set_pickup_time(self, triptime):
        self.pickup_time = triptime

    def __change_to_idle(self):
        self.__change_behavior_model(status_codes.V_IDLE)

    def __change_to_cruising(self):
        self.__change_behavior_model(status_codes.V_CRUISING)

    def __change_to_assigned(self):
        self.__change_behavior_model(status_codes.V_ASSIGNED)

    def __change_to_occupied(self):
        self.__change_behavior_model(status_codes.V_OCCUPIED)

    def __change_to_off_duty(self):
        self.__change_behavior_model(status_codes.V_OFF_DUTY)

    def __change_to_waytocharge(self):
        self.__change_behavior_model(status_codes.V_WAYTOCHARGE)
        
    def __change_to_charging(self):
        self.__change_behavior_model(status_codes.V_CHARGING)

    def __change_to_waitpile(self):
        self.__change_behavior_model(status_codes.V_WAITPILE)

    def __change_to_tobedispatched(self):
        self.__change_behavior_model(status_codes.V_TOBEDISPATCHED)

    def __change_behavior_model(self, status):
        self.__behavior = self.behavior_models[status]
        self.state.status = status
        
    def __log(self):
        # self.charging_dict.to_csv("output_charging.csv")
        if FLAGS.log_vehicle:
            sim_logger.log_vehicle_event(self.state.to_msg())
