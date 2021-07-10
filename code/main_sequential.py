from config.hex_setting import hex_shp_path, charging_station_shp_path, NUM_NEAREST_CS, trip_file, travel_time_file, \
    TIMESTEP, START_OFFSET, SIM_DAYS, START_TIME, TRAINING_CYCLE, UPDATE_PARAMETER
from common.time_utils import get_local_datetime
from simulator.simulator_sequential import Simulator
from dqn_agent.dqn_agent import DeepQNetworkAgent

# ---------------MAIN FILE---------------

if __name__ == '__main__':

    if SIM_DAYS > 0:
        start_time = START_TIME + int(60 * 60 * 24 * START_OFFSET)  # modify the start day.
        print("Simulate Episode Start Datetime: {}".format(get_local_datetime(start_time)))
        end_time = start_time + int(60 * 60 * 24 * SIM_DAYS)
        print("Simulate Episode End Datetime : {}".format(get_local_datetime(end_time)))

        simulator = Simulator(start_time, TIMESTEP)
        simulator.init(hex_shp_path, charging_station_shp_path, trip_file, travel_time_file, NUM_NEAREST_CS)

        dqn_agent = DeepQNetworkAgent()
        n_steps = int(3600 * 24 / TIMESTEP)  # 60 per minute
        with open('logs/parsed_results_new.csv', 'w') as f, open('logs/target_charging_stations.csv','w') as f1:
            f.writelines(
                '{},{},{},{},{},{},{},{},{},{},{},{}\n'.format("time", "num_idle", "num_serving", "num_charging",
                                                               "num_cruising", "num_assigned", "num_waitpile",
                                                               "num_tobedisptached", "num_offduty",
                                                               "num_matches", "pass_arrivals", "removed_pass"))
            for day in range(SIM_DAYS):
                print("############################ SUMMARY ################################")
                for i in range(n_steps):
                    tick = simulator.get_current_time()
                    simulator.par_step()

                    num_idle, num_serving, num_charging, num_cruising, num_matches, total_num_arrivals, total_removed_passengers, num_assigned, num_waitpile, num_tobedisptached, num_offduty = simulator.summarize_metrics()

                    f.writelines(
                        '{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(tick, num_idle, num_serving, num_charging,
                                                                       num_cruising, num_assigned, num_waitpile,
                                                                       num_tobedisptached, num_offduty,
                                                                       num_matches, total_num_arrivals,
                                                                       total_removed_passengers,
                                                                       ))
                    print(
                        'Number of matched: {}, number of idling:{}, number of cruising:{}， number of charging:{}'.format(
                            num_matches, num_idle, num_cruising, num_charging))

                    if simulator.dump_transition_to_dqn() is not None:
                        states, actions, next_states, rewards, flags = simulator.dump_transition_to_dqn()
                        [dqn_agent.add_transition(state, action, next_state, reward, flag) for
                         state, action, next_state, reward, flag in zip(states, actions, next_states, rewards, flags)]
                    # if tick % TRAINING_CYCLE == 0:
                    #     dqn_agent.train()
                    # if tick % UPDATE_PARAMETER == 0:
                    #     dqn_agent.copy_parameter()
                    for charging_station_id in simulator.get_charging_station_ids():
                        f1.writelines('{},{}\n'.format(tick,charging_station_id))
