import numpy as np
import time

from pandas.core.indexes.api import all_indexes_same
from novelties import status_codes
from collections import defaultdict
from simulator.services.routing_service import RoutingEngine
from config.hex_setting import REJECT_TIME, SEC_PER_MIN, SPEED
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

    def get_info(self):
        print('Match zone id: {}, number of hexs:{}'.format(self.matching_zone_id,len(self.hex_zones)))

    
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
        '''
        return a list
        '''
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
        nvs=0
        for i in range(len(new_veh)):
            self.hex_zones[i].vehicles=new_veh[i]
            nvs+=len(new_veh[i].keys())
        # print('total vehicles deployed to zone {} is {}'.format(self.matching_zone_id,nvs))


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

    def match(self):
        '''
        Perform the matching here.
        :param tick:
        :return:
        '''
        # get all vehicles and passengers first
        all_pass = self.get_all_passenger()
        all_veh = self.get_all_veh()
        self.matching_algorithms(all_pass,all_veh)


    def matching_algorithms(self, passengers,vehicles):
        '''
        todo: complete the matching algorithm here
        passengers: the set of available Customer objects
        vehicles: the set of vehicle objects
        match available vehicles with passengers
        Change the status for passengers and vehicles
        :return: 
        no return here. We will change the mark for each passengers and drivers as they are matched
        '''
        #get only available vehicles        
        print('Current matching zone={}, Total matched passengers={}, Number of passengers={}, Number of drivers={}'.format(self.matching_zone_id, self.num_matches,len(passengers.keys()),len(vehicles.keys())))
        if len(passengers.keys()) > 0 and len(vehicles.keys())>0:
            vehicles={key:value for key, value in vehicles.items() if value.state.status==status_codes.V_IDLE}
            if len(vehicles.keys())>0:
                self.num_matches += self.match_requests(vehicles,passengers)

        

    ##### match requests to vehicles ######
    def match_requests(self, vehicles, passengers):
        v_lonlat = [[veh.state.lon,veh.state.lat] for veh in vehicles.values()] 
        v_hex_id = [veh.state.hex_id for veh in vehicles.values()]
        # print('the vehicle hex ids are:',v_hex_id)
        r_lonlat = [[customer.request.origin_lon,customer.request.origin_lat] for customer in passengers.values()]
        r_hex_id = [customer.request.origin_id for customer in passengers.values()]
        print(v_lonlat[0],r_lonlat[0])
        dist = self.eta_matrix(v_lonlat, r_lonlat)
        print('dist_matrix for matching zone {} is {}'.format(self.matching_zone_id,dist[0,0]))
        assignments = self.assign_nearest_vehicle(v_hex_id,r_hex_id, dist.T)
        
        for (v_hex_id, r_hex_id, triptime,distance),vehicle,customer in zip(assignments,vehicles.values(),passengers.values()):
            if vehicle is None or customer is None:
                continue
            customer.matched=True
            vehicle.customer=customer #add matched passenger to the current one
            vehicle.head_for_customer(customer.get_origin_lonlat(), triptime, customer.get_id(), distance)
            print('Vehicle {} are dispatched'.format(vehicle.state.vehicle_id))
            # customer.wait_for_vehicle(triptime)
            
            vehicle.state.status = status_codes.V_ASSIGNED # update vehicle status

        return len(assignments) # record nums of getting matched

    def eta_matrix(self, origins_coord, dest_coord):
        origin_set = list(origins_coord)
        dest_set = list(dest_coord)
        d = np.array(self.routing_engine.eta_many_to_many(origin_set, dest_set), dtype=np.float32)
        d[np.isnan(d)] = float('inf')
        return  d

    # Returns list of assignments
    def assign_nearest_vehicle(self, ori_hex_ids, dest_hex_ids, dist):
        assignments = []
        for did,d in enumerate(dest_hex_ids):
            if len(ori_hex_ids)==0:
                break
            # Reuturns the min distance
            # oi = T[di].argmin()
            oid = dist[did].argmin()
            dd = dist[did, oid]
            tt = dd/SPEED
            # print("Chosen t: ", tt)
            # print("Chosen D: ", dd)
            if tt > self.reject_wait_time:
                continue
            o = ori_hex_ids[oid]

            assignments.append((o, d, tt, dd)) # o-d's hex_id, trip time, and distance
        return assignments

    def get_num_of_matches(self):
        return self.num_matches