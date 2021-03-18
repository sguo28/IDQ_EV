from simulator.models.customer.customer_repository import CustomerRepository
from simulator.models.vehicle.vehicle_repository import VehicleRepository
import numpy as np
import time
from novelties import status_codes
from collections import defaultdict
# from config.settings import MAP_WIDTH, MAP_HEIGHT
from simulator.services.routing_service import RoutingEngine
import pandas as pd
from config.hex_setting import num_reachable_hex,REJECT_TIME, SEC_PER_MIN
import ray

@ray.remote
class matching_zone(object):
    def __init__(self,m_id,hex_zones):
        '''
        m_id: macthing zone id
        hex_zones: the list of hex zone objects
        '''
        self.matching_zone_id = m_id
        self.hex_zones = hex_zones
        self.reject_wait_time = REJECT_TIME*SEC_PER_MIN # sec
        self.routing_engine = RoutingEngine.create_engine()
        self.local_hex_collection={hex.hex_id:hex for hex in hex_zones} #createa a local hex
        self.num_matches = 0

    def get_local_collection(self):
        return self.local_hex_collection

    # def dump_m_transitions(self):
    #     m_transitions = []
    #     m_transitions.append([h.get_h_transitions()[0] for h in self.hex_zones])
    #     return m_transitions[0]
    
    def step(self,tick):
        '''
        Perform the matching here.
        :param tick:
        :return:
        '''
        # get all vehicles and passengers first
        all_pass=self.get_all_passenger()
        all_veh=self.get_all_veh()
        self.matching_algorithms(tick,all_pass,all_veh)
        # for h in self.hex_zones:
        #     h.step(tick)

    def match(self,tick):
        '''
        Perform the matching here.
        :param tick:
        :return:
        '''
        # get all vehicles and passengers first
        all_pass=self.get_all_passenger()
        all_veh=self.get_all_veh()
        self.matching_algorithms(tick,all_pass,all_veh)

    def dispatch(self,tick):
        '''
        Call dispatch for each hex zones
        :param tick:
        :return:
        '''
        [h.vehicle_dispatch(tick) for h in self.hex_zones]

    def update_passengers(self):
        '''
        Call update passenger in each hex zones
        :return:
        '''
        [h.update_passengers() for h in self.hex_zones]

    def async_demand_gen(self,tick):
        #do the demand generation for all hex zones in the matching zone
        [h.demand_generation(tick) for h in self.hex_zones]
        return True

    def get_vehicles_by_hex(self):
        #return the dictionary of vehicles by different hex areas in the matching zone
        #return as a dict
        veh_dict=[hex.vehicles for hex in self.hex_zones]
        return veh_dict

    def get_vehicles_by_hex_list(self):
        #return the dictionary of vehicles by different hex areas in the matching zone
        #return as a dict
        veh_dict=[hex.vehicles.values() for hex in self.hex_zones]
        return veh_dict

    def set_vehicles_by_hex(self,new_veh):
        #reset the new collection of vehicles for each hex areas in the matching zone
        #make sure the order in new_veh is the same as the hex zone orders in each matching zone
        for i in range(len(new_veh)):
            self.hex_zones[i].vehicles=new_veh[i]


    def get_arrivals_length(self):
        return len(self.hex_zones[0].arrivals)


    def get_all_veh(self):
        '''
        :return: all vehicles in the hex areas inside the matching zone
        '''
        all_vehs=defaultdict()
        for hex_zone in self.hex_zones:
            all_vehs.update(hex_zone.vehicles)

        return all_vehs

    def get_all_passenger(self):
        '''
        :return: all available passengers in the list
        todo: consider sorting the passengers based on their time of arrival?
        todo: add passenger status as matched and unmatched [done]
        todo: do we need to get available passengers?
        '''
        available_pass=defaultdict()
        for hex_zone in self.hex_zones:
            local_availables={key:value for (key,value) in hex_zone.passengers.items() if value.matched==False}
            available_pass.update(local_availables)
        return available_pass

    def get_served_num(self):
        return sum([h.served_num for h in self.hex_zones])
    
    def get_veh_waiting_time(self):
        '''
        todo: this function makes no sense [# this is the waiting time for a charging pile]
        :return:
        '''
        return sum([h.veh_waiting_time for h in self.hex_zones])

    def matching_algorithms(self,tick, passengers,vehicles):
        '''
        todo: complete the matching algorithm here
        passengers: the set of available Customer objects
        vehicles: the set of vehicle objects
        match available vehicles with passengers
        Change the status for passengers and vehicles
        :return: 
        no return here. We will change the mark for each passengers and drivers as they are matched
        '''
        match_commands = []
        if len(passengers.keys()) > 0 and len(vehicles.keys())>0:
            match_commands = self.match_requests(vehicles.values(),passengers.values())
        self.update_vehicles(vehicles, match_commands) # output is vehicle, but no use here. 
        self.num_matches += len(match_commands)
        for command in match_commands:
            vehicle = vehicles[command["vehicle_id"]]
            vehicle.state.current_capacity += 1
            if vehicle is None:
                # self.logger.warning("Invalid Vehicle id")
                continue
            customer = passengers[command["customer_id"]]
            customer.matched=True
            if customer is None:
                # self.logger.warning("Invalid Customer id")
                continue
            triptime = command["duration"]
            print('matched')
            vehicle.head_for_customer(customer.get_origin(), triptime, customer.get_id(), command["distance"],tick,command['ori_hex_id'],command['dest_hex_id'])
            customer.wait_for_vehicle(triptime)

    def update_vehicles(self, vehicles, match_commands):
        '''
        Make matched vehicles change status to matched.
        In-place update for vehicle objects in the set
        '''
        vehicle_ids = [command["vehicle_id"] for command in match_commands]
        for ids in vehicle_ids:
            vehicles[ids].state.status = status_codes.V_ASSIGNED

    ##### match requests to vehicles ######
    def match_requests(self, vehicles, passengers):
        match_list = []
        n_vehicles = len(vehicles)
        if n_vehicles == 0:
            return match_list
        v_latlon = [[veh.state.lat,veh.state.lon] for veh in vehicles]# vehicles[["lat", "lon","hex_id"]]
        v_hex_id = [veh.state.hex_id for veh in vehicles]
        V = defaultdict(list)
        vid2coord = {}
        for vid, row in enumerate(v_latlon):
            coord = (row[1], row[0]) # x, y
            vid2coord[vid] = coord
            V[coord].append(vid)
        r_latlon = [[customer.request.origin_lat, customer.request.origin_lon] for customer in passengers]
        r_hex_id = [customer.request.origin_id for customer in passengers]
        R = defaultdict(list)
        for rid, row in enumerate(r_latlon):
            coord =(row[1], row[0]) # self.get_coord(row.olon, row.olat)
            R[coord].append(rid)
        # V and R are two statuses: vehicle and request per zone.
        for coord in range(num_reachable_hex):
            if not R[coord]:
                continue

            target_rids = R[coord]
            candidate_vids = V[coord]
            if len(candidate_vids) == 0:
                continue
            
            T, dist = self.eta_matrix(v_latlon, r_latlon)
            
            assignments = self.assign_nearest_vehicle(v_hex_id[candidate_vids],r_hex_id[target_rids],T.T, dist.T)
            for vid, rid, tt, d in assignments:
                match_list.append(self.create_matching_dict(vid, rid, tt, d,v_hex_id[vid] ,r_hex_id[rid]))
                V[vid2coord[vid]].remove(vid)
            print(match_list["distance"])
        return match_list


    # Craeting matching dictionary assciated with each vehicle ID
    def create_matching_dict(self, vehicle_id, customer_id, duration, distance,ori_hex_id,dest_hex_id):
        match_dict = {}
        match_dict["vehicle_id"] = vehicle_id
        match_dict["customer_id"] = customer_id
        match_dict["duration"] = duration
        match_dict["distance"] = distance
        match_dict['ori_hex_id'] = ori_hex_id
        match_dict['dest_hex_id'] = dest_hex_id
        return match_dict

    # Returns list of assignments
    def assign_nearest_vehicle(self, ori_hex_ids, dest_hex_ids, T, dist):
        assignments = []
        for di, did in enumerate(dest_hex_ids):
            if len(assignments) >= len(ori_hex_ids):
                break

            # Reuturns the min distance
            # oi = T[di].argmin()
            oi = dist[di].argmin()
            tt = T[di, oi] # - t_queue
            dd = dist[di, oi]
            # print("Chosen t: ", tt)
            # print("Chosen D: ", dd)
            if tt > self.reject_wait_time:
                continue
            oid = ori_hex_ids[oi]

            assignments.append((oid, did, tt, dd))
            T[:, oi] = float('inf')
        return assignments

    def eta_matrix(self, origins_array, destins_array):
        try:
            destins = [(lat, lon) for lat, lon in destins_array.values]
        except AttributeError:
            destins = [(loc[0],loc[1]) for loc in destins_array]
        # destins = [(lat, lon) for lat, lon in destins_array.values]
        origins = [(lat, lon) for lat, lon in origins_array.values]
        # origin_set = list(set(origins))
        origin_set = list(origins)
        latlon2oi = {latlon: oi for oi, latlon in enumerate(origin_set)}
        T, d = np.array(self.routing_engine.eta_many_to_many(origin_set, destins), dtype=np.float32)
        
        T[np.isnan(T)] = float('inf')
        d[np.isnan(d)] = float('inf')
        T = T[[latlon2oi[latlon] for latlon in origins]]
        # print("T: ", T)
        # print("D: ", d.shape)
        return [T, d]
        
    def get_num_of_matches(self):
        return self.num_matches