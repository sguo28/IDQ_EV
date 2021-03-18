from config.hex_setting import hex_shp_path, charging_station_shp_path, NUM_NEAREST_CS, trip_file, travel_time_file, \
    TIMESTEP, START_OFFSET, SIM_DAYS, START_TIME
from common.time_utils import get_local_datetime
from simulator.simulator import Simulator
from dqn_agent.dqn_agent import DeepQNetwork

# ---------------MAIN FILE---------------

if __name__ == '__main__':

    if SIM_DAYS > 0:
        start_time = START_TIME + int(60 * 60 * 24 * START_OFFSET)  # modify the start day.
        print("Simulate Episode Start Datetime: {}".format(get_local_datetime(start_time)))
        end_time = start_time + int(60 * 60 * 24 * SIM_DAYS)
        print("Simulate Episode End Datetime : {}".format(get_local_datetime(end_time)))

        simulator = Simulator(start_time, TIMESTEP)
        simulator.init(hex_shp_path, charging_station_shp_path, trip_file, travel_time_file, NUM_NEAREST_CS)

        q_network = DeepQNetwork()
        n_steps = int(3600 * 24 / TIMESTEP)  # 60 per minute
        with open('logs/parsed_results_new.csv', 'w') as f:
            f.writelines(
                '{},{},{},{},{},{},{},{},{},{},{}\n'.format("time", "num_idle", "num_serving", "num_charging", "num_cruising",
                                                   "num_matches", "pass_arrivals", "removed_pass","num_assigned","num_waitpile", "num_tobedisptached"))
            for day in range(SIM_DAYS):
                print("############################ SUMMARY ################################")
                for i in range(n_steps):
                    tick = simulator.get_current_time()
                    simulator.par_step()

                    num_idle, num_serving, num_charging, num_cruising, num_matches, total_num_arrivals, total_removed_passengers,num_assigned,num_waitpile, num_tobedisptached = simulator.summarize_metrics()

                    f.writelines(
                        '{},{},{},{},{},{},{},{},{},{},{}\n'.format(tick, num_idle, num_serving, num_charging, num_cruising,
                                                           num_matches, total_num_arrivals, total_removed_passengers,
                                                           num_assigned, num_waitpile, num_tobedisptached))

                    # state,action,next_state,reward,flag = Sim_experiment.simulator.dump_transition_batch()
                    # print(Sim_experiment.simulator.dump_transition_batch())
                    # Sim_experiment.q_network.add_transition(Sim_experiment.simulator.dump_transition_batch())
