from numpy.core.fromnumeric import mean
from novelties import status_codes
from collections import deque
from config.hex_setting import SIM_ACCELERATOR

class chargingpile:
    '''
    todo: check charging time unit
    '''
    def __init__(self,type,location,hex_id):
        self.occupied=False
        self.time_to_finish=0
        self.assigned_vehicle=None  #vehicle agent
        self.type = type
        self.location = location
        self.hex_id = hex_id
        self.served_num = 0
        # self.chargingpile_state = ChargingPileState(type = self.type, location = self.location)
        # self.target_SOC = self.assigned_vehicle.get_mile_of_range()
        # self.rate= self.chargingpile_state.set_charging_rate()
        if self.type == status_codes.SP_DCFC:
            self.rate = SIM_ACCELERATOR * 240/(60*60) # mile per sec
        else:
            self.rate = SIM_ACCELERATOR * 25/(60*60) # mile per sec

    def assign_vehicle(self,veh):
        self.occupied=True
        self.time_to_finish=(veh.get_target_SOC()-veh.get_SOC())*veh.get_mile_of_range()/self.rate
        self.assigned_vehicle=veh 
        self.assigned_vehicle.start_charge(self.get_cp_location(),self.get_cp_hex_id())

    def get_cp_location(self):
        return self.location

    def get_cp_hex_id(self):
        return self.hex_id

    def step(self,time_step,tick):
        if self.time_to_finish>0:
            self.time_to_finish-=time_step
            if self.time_to_finish<=0:
                #charging has been completed!
                self.occupied=False
                self.time_to_finish=0
                self.assigned_vehicle.state.SOC = self.assigned_vehicle.state.target_SOC  #fully charge
                self.assigned_vehicle.end_charge(tick,self.get_cp_location(),self.get_cp_hex_id())
                # some more code on updating vehicle status after completing charging
                # self.assigned_vehicle.charge_flag = False 
                # print("####VEH {} with STATUS {} END CHARGE#### AT LOC {}".format(self.assigned_vehicle.state.vehicle_id,self.assigned_vehicle.state.status==status_codes.V_IDLE ,self.get_cp_hex_id()))
                self.assigned_vehicle=None
                self.served_num +=1
                # self.queue
                # self.assigned_vehicle_id=None #again, you probably do not need it


class charging_station:
    def __init__(self,n_l2=1,n_dcfast=1,lat=None,lon=None,hex_id=None):
        self.location = float(lon), float(lat)
        #initial the charging piles for the charging station
        self.piles=[chargingpile(type=status_codes.SP_LEVEL2,location=self.location, hex_id = hex_id) for _ in range(n_l2)] + \
            [chargingpile(type=status_codes.SP_DCFC,location=self.location, hex_id = hex_id) for _ in range(n_dcfast)]
        # self.available_piles=self.piles
        self.waiting_time=[]
        self.charging_time=[]
        self.queue=deque() #waiting queue for vehicle
        self.virtual_queue =[]
        self.time_to_cs = []
        self.hex_id = hex_id

    def get_cs_location(self):
        return self.location

    def get_cs_hex_id(self):
        return self.hex_id
    def get_available(self):
        #set the list of available charging piles
        self.available_piles=[p for p in self.piles if p.occupied==False]

    def step(self,time_step,tick):
        '''
        First update each pile, then find available charging piles, then match, then update queue
        :return:
        '''
        #update the status
        [p.step(time_step,tick) for p in self.piles] #update the status of each charging pile
        self.get_available() #update available piles

        #update waiting time of each vehicle in the queue
        #assign waiting vehicles to each pile
        # must have both vehicle and pile available to proceed

        while len(self.queue)>0 and len(self.available_piles)>0:
            veh=self.queue.popleft(); pile=self.available_piles.pop()
            pile.assign_vehicle(veh)

            self.waiting_time.append(veh.charging_wait) #total waiting time of the vehicle
            self.charging_time.append(pile.time_to_finish) #total charging time

        # for unmatched vehicles, update the waiting time of vehicles in the charging queue
        for v in self.queue:
            v.charging_wait+=1

    def add_arrival_veh(self,veh):
        '''
        :param veh: vehicle object (class)
        '''
        self.queue.append(veh)
        veh.charging_wait=0 #no wait at the beginning

    def get_queue_length(self):
        return len(self.queue)

    def get_average_waiting_time(self):
        if len(self.waiting_time)<1:
            return 0.0
        else:
            return mean(self.waiting_time[-20:])

    def get_served_num(self):
        return sum([cp.served_num for cp in self.piles])

#### in your funciton, you can define your charging repository as a list of length N, N = total number of stations

