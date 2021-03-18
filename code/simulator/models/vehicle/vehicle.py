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
        # print(self.state.type, self.state.max_capacity)
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

    # state changing methods
    def step(self,timestep, tick, hex):
        self.working_time += timestep

        if self.state.status == status_codes.V_IDLE:
            self.duration[status_codes.V_IDLE] += timestep


        if self.__behavior.available:
            self.state.idle_duration += timestep
        else:
            self.state.idle_duration = 0

        try:
            self.__behavior.step(self,timestep=timestep, tick=tick)
        except:
            logger = getLogger(__name__)
            logger.error(self.state.to_msg())
            raise
        # if self.state.current_hex!=self.state.hex_id:
        if self.state.current_hex!=self.state.hex_id:
            #update each vehicle if the new location (current_hex) is different from its current one (hex_id)
            hex.remove_veh(self)
            hex.add_veh(self)
            self.state.hex_id=self.state.current_hex

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


    def cruise(self, route, triptime,hex_id= 0,a = 0, tick=0):
        '''
        stay_still: if stay_still == 1: skip virtual routing, convert to park(idle) as well as record a full transition
        '''
        assert self.__behavior.available
        self.state.current_hex = hex_id
        self.rb_state = [tick,self.get_id(),self.get_hex_id(),self.get_SOC()]
        self.rb_action = a
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
        self.state.SOC -= distance*MILE_PER_METER*SIM_ACCELERATOR /self.get_mile_of_range()
        self.state.travel_dist += distance
        self.__reset_plan()
        self.__set_destination(destination, triptime)
        self.state.assigned_customer_id = customer_id
        self.__customers_ids.append(customer_id)
        self.change_to_assigned()
        self.__log()

    def head_for_charging_station(self, triptime, cs_id, cs_coord, route):
        '''
        :destination:
        todo: change varibale name: station to station_id
        '''
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
        self.state.current_hex = self.customer.get_origin()
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
        lenC = len(self.__customers)
        self.state.current_hex = self.__customers[lenC-1].get_destination()
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
        self.rb_next_state = [tick,self.get_id(),self.state.current_hex,self.get_SOC()]
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
        self.rb_next_state = [tick,self.get_id(),self.state.current_hex,self.get_SOC()]
        self.rb_reward = 0
        self.flag = int(self.state.status == status_codes.V_OFF_DUTY)
        self.recent_transitions.append((self.rb_state,self.rb_action,self.rb_next_state,self.rb_reward,self.flag))
        # self.rb_state = self.rb_next_state
        self.__reset_plan()
        self.__change_to_idle()
        self.__log()

    def start_waitpile(self):
        self.__change_to_waitpile()
        self.__log()

    
    def start_charge(self,coord,hex_id):
        self.__reset_plan()
        self.state.lon, self.state.lat = coord
        self.state.current_hex = hex_id
        self.__change_to_charging()
        self.__log()
    
    def end_charge(self,tick,coord,hex_id):
        self.state.current_capacity = 0 # current passenger on vehicle
        self.__customers_ids = []
        self.__change_to_idle()
        self.__reset_plan()
        self.state.lon, self.state.lat = coord
        self.state.current_hex = hex_id
        self.rb_next_state = [tick,self.get_id(),self.state.current_hex,self.get_SOC()]
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

    def change_to_assigned(self):
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
