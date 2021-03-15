import simulator.models.customer.customer_repository
from common import geoutils
from simulator.settings import FLAGS

class VehicleBehavior(object):
    available = True

    def step(self, vehicle, timestep,tick):
        pass


class Waytocharge(VehicleBehavior):
    available = False
    # Updated remaining time to destination
    def step(self, vehicle, timestep,tick):
        arrived = vehicle.update_time_to_destination(timestep)
        if arrived: # arrive at charing station
            charging_station = vehicle.get_assigned_cs()
            try:
                # print(vehicle.get_id(),"####WAIT TO CHARGE####")
                charging_station.add_arrival_veh(vehicle)
                vehicle.start_waitpile()
            except:
                AttributeError
            # vehicle.start_charge(charging_pile)
            # consider non_linear charging speed and add residual time as "time to destination(CS)"

class Charging(VehicleBehavior):
    available = False
    # Updated remaining time to destination, if arrived customer gets off
    pass

    # def step(self, vehicle, timestep):
    #     charged = vehicle.update_time_to_destination(timestep)
    #     if charged: # if finished charging
    #         # print("####END CHARGE####")
    #         vehicle.end_charge() # end charge and be IDLE status

class Waitpile(VehicleBehavior):

    available = False
    pass

class Idle(VehicleBehavior):

    pass


class Cruising(VehicleBehavior):
    # Updated remaining time to destination, if arrived states changes to parking
    def step(self, vehicle, timestep,tick):
        arrived = vehicle.update_time_to_destination(timestep)
        if arrived:
            vehicle.park(tick) # arrived and be idle.
            return
        self.drive(vehicle, timestep)


    def drive(self, vehicle, timestep):
        route = vehicle.get_route()      # Sequence of (lon, lat)
        speed = vehicle.get_speed()
        dist_left = timestep * speed    # Remaining Distance
        rlats, rlons = zip(*([vehicle.get_location()] + route)) # New vehicle location after driving this route
        step_dist = geoutils.great_circle_distance(rlats[:-1], rlons[:-1], rlats[1:], rlons[1:])    # Get distcnace in meters
        for i, d in enumerate(step_dist): # update location per time step
            if dist_left < d:
                bearing = geoutils.bearing(rlats[i], rlons[i], rlats[i + 1], rlons[i + 1])      # Calculate angle of motion
                next_location = geoutils.end_location(rlats[i], rlons[i], dist_left, bearing)   # Calculate nxt location
                vehicle.update_location(next_location, route[i + 1:])           # Updating location based on route's nxt (lon, lat)
                return
            dist_left -= d

        if len(route) > 0:
            vehicle.update_location(route[-1], [])  # Go the last step


class Occupied(VehicleBehavior):
    available = False
    # Updated remaining time to destination, if arrived customer gets off
    def step(self, vehicle, timestep,tick):
        arrived = vehicle.update_time_to_destination(timestep)
        if arrived:
            vehicle.dropoff(tick)

class Assigned(VehicleBehavior):
    available = False
    # Updated remaining time to destination, if arrived, update customer ID and picks him up
    def step(self, vehicle, timestep,tick):
        arrived = vehicle.update_time_to_destination(timestep)
        if arrived:
            customer = simulator.models.customer.customer_repository.CustomerRepository.get(
            vehicle.get_assigned_customer_id())
            vehicle.pickup(customer)

class OffDuty(VehicleBehavior):
    available = False
    # Updated remaining time to destination, if returned state changes to parking
    def step(self, vehicle, timestep,tick):
        returned = vehicle.update_time_to_destination(timestep)
        if returned:
            vehicle.park()
