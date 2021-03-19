import numpy as np
from novelties import status_codes
from collections import defaultdict
from simulator.services.routing_service import RoutingEngine
from config.hex_setting import REJECT_TIME, SEC_PER_MIN, SPEED
import ray


@ray.remote
class matching_zone(object):
    def __init__(self, m_id, hex_zones, routes):
        """
        m_id: matching zone id
        hex_zones: the list of hex zone objects
        """
        self.matching_zone_id = m_id
        self.hex_zones = hex_zones
        self.reject_wait_time = REJECT_TIME * SEC_PER_MIN  # sec
        self.routing_engine = RoutingEngine.create_engine()
        self.local_hex_collection = {hex.hex_id: hex for hex in hex_zones}  # create a a local hex
        self.num_matches = 0
        self.routes = routes

    def get_local_collection(self):
        return self.local_hex_collection

    def get_info(self):
        print('Match zone id: {}, number of hexs:{}'.format(self.matching_zone_id, len(self.hex_zones)))

    def dispatch(self, tick):
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

    def update_vehicles(self,timestep,timetick):
        '''
        call step function for each vehicles
        :return:
        '''
        for hex in self.hex_zones:
            for veh in hex.vehicles.values():
                veh.step(timestep,timetick)

    def async_demand_gen(self, tick):
        # do the demand generation for all hex zones in the matching zone
        [h.demand_generation(tick) for h in self.hex_zones]
        return True

    def get_vehicles_by_hex(self):
        '''
        return: list of vehicle_dict per hex
        '''
        veh_dict = [hex.vehicles for hex in self.hex_zones]
        return veh_dict

    def get_vehicles_by_hex_list(self):
        """
        :return:
        """
        veh_dict = [hex.vehicles.values() for hex in self.hex_zones]
        return veh_dict

    def set_vehicles_by_hex(self, new_veh):
        # reset the new collection of vehicles for each hex areas in the matching zone
        # make sure the order in new_veh is the same as the hex zone orders in each matching zone
        nvs = 0
        for i in range(len(new_veh)):
            self.hex_zones[i].vehicles = new_veh[i]
            nvs += len(new_veh[i].keys())
        # print('total vehicles deployed to zone {} is {}'.format(self.matching_zone_id,nvs))

    def get_arrivals_length(self):
        return len(self.hex_zones[0].arrivals)

    def get_all_veh(self):
        '''
        :return: all vehicles in the hex areas inside the matching zone
        '''
        all_vehs = defaultdict()
        for hex_zone in self.hex_zones:
            all_vehs.update(hex_zone.vehicles)

        return all_vehs

    def get_all_passenger(self):
        '''
        :return: all available passengers in the list
        todo: consider sorting the passengers based on their time of arrival?
        '''
        available_pass = defaultdict()
        for hex_zone in self.hex_zones:
            local_availables = {key: value for (key, value) in hex_zone.passengers.items() if value.matched == False}
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
        self.matching_algorithms(all_pass, all_veh)

    def matching_algorithms(self, passengers, vehicles):
        '''
        todo: complete the matching algorithm here
        passengers: the set of available Customer objects
        vehicles: the set of vehicle objects
        match available vehicles with passengers
        Change the status for passengers and vehicles
        :return: 
        no return here. We will change the mark for each passengers and drivers as they are matched
        '''
        # get only available vehicles
        print(
            'Current matching zone={}, Total matched passengers={}, Number of passengers={}, Number of drivers={}'.format(
                self.matching_zone_id, self.num_matches, len(passengers.keys()), len(vehicles.keys())))
        if len(passengers.keys()) > 0 and len(vehicles.keys()) > 0:
            vehicles = {key: value for key, value in vehicles.items() if value.state.status == status_codes.V_IDLE}
            if len(vehicles.keys()) > 0:
                self.num_matches += self.match_requests(vehicles, passengers)

    ##### match requests to vehicles ######
    def match_requests(self, vehicles, passengers):
        """

        :param vehicles:
        :param passengers:
        :return:
        """
        v_hex_id = [veh.state.hex_id for veh in vehicles.values()]
        # print('the vehicle hex ids are:',v_hex_id)
        r_hex_id = [customer.request.origin_id for customer in passengers.values()]
        # dist = self.eta_matrix(v_lonlat, r_lonlat)
        od_tensor = self.get_OD_tensor(v_hex_id,r_hex_id)
        assignments = self.assign_nearest_vehicle(v_hex_id, r_hex_id, od_tensor.T)

        for (v_hex_id, r_hex_id, triptime, distance), vehicle, customer in zip(assignments, vehicles.values(),
                                                                               passengers.values()):
            if vehicle is None or customer is None:
                continue
            customer.matched = True
            vehicle.customer = customer  # add matched passenger to the current on
            vehicle.state.need_route = True
            vehicle.state.current_hex = customer.get_id()
            vehicle.head_for_customer(customer.get_origin_lonlat(), triptime, customer.get_id(), distance)

        return len(assignments)  # record nums of getting matched

    def get_OD_tensor(self,origin_id,dest_id):
        """

        :param origin_id: origin hex_id
        :param dest_id: destination hex id
        :return: matrix O x D by 3 (route,time,distance)
        """
        tensor_od = np.zeros(len(origin_id),len(dest_id))

        for o in origin_id:
            for d in dest_id:
                tensor_od[o,d]=self.routes[(o,d)]

        tensor_od[np.isnan(tensor_od)] = 0.0
        return tensor_od

    def eta_matrix(self, origins_coord, dest_coord):
        origin_set = list(origins_coord)
        dest_set = list(dest_coord)
        # d = np.array(self.routing_engine.get_distance_matrix(origin_set,dest_set), dtype=np.float32)
        d = np.array(self.routing_engine.eta_many_to_many(origin_set, dest_set), dtype=np.float32)
        d[np.isnan(d)] = float('inf')
        return d

    # Returns list of assignments
    def assign_nearest_vehicle(self, ori_hex_ids, dest_hex_ids, od_tensor):
        assignments = []
        time = od_tensor[:,:,0]
        dist = od_tensor[:,:,1]
        for did, d in enumerate(dest_hex_ids):
            if len(ori_hex_ids) == 0:
                break
            # Reuturns the min distance
            # oi = T[di].argmin()
            oid = dist[did].argmin()
            dd = dist[did, oid]
            tt = time[did,oid]
            # print("Chosen t: ", tt)
            # print("Chosen D: ", dd)
            if tt > self.reject_wait_time:
                continue
            o = ori_hex_ids[oid]

            assignments.append((o, d, tt, dd))  # o-d's hex_id, trip time, and distance

        return assignments

    def get_num_of_matches(self):
        return self.num_matches
