import argparse
import time
import numpy as np
import pandas as pd
from common.time_utils import get_local_datetime
from config.setting import HEX_SHP_PATH, CS_SHP_PATH, NUM_NEAREST_CS, TRIP_FILE, TRAVEL_TIME_FILE, \
    TIMESTEP, START_OFFSET, SIM_DAYS, N_EPISODE, START_TIME, TRAINING_CYCLE, UPDATE_CYCLE, STORE_TRANSITION_CYCLE, CNN_RESUME
from dqn_option_agent.dqn_option_agent import DeepQNetworkOptionAgent
from dqn_agent.dqn_agent_with_cnn.cnn_dqn_agent import DeepQNetworkAgent
from simulator.simulator_option import Simulator
# from logs.parse_results_cnn import auto_save_metric_plots

# ---------------MAIN FILE---------------

if __name__ == '__main__':
    """
    todo: dont forget to remove the epsilon=1.0 and SOC threshold = -10. o.w. all action is randomly selected.
    """
    arg = argparse.ArgumentParser("Start running")

    arg.add_argument("--islocal", "-l", default=1, type=bool, help="choose local matching instead of global matching")
    arg.add_argument("--isoption", "-o", default=1, type=bool, help="choose covering option or not")
    arg.add_argument("--ischarging", "-c", default=0, type=bool, help="choose charging option or not")
    arg.add_argument("--num_option", "-option", default=3, type=int, help="number of options to append")
    args = arg.parse_args()
    if SIM_DAYS > 0:
        start_time = START_TIME + int(60 * 60 * 24 * START_OFFSET)  # start_time = 0
        print("Simulate Episode Start Datetime: {}".format(get_local_datetime(start_time)))
        end_time = start_time + int(60 * 60 * 24 * SIM_DAYS)
        print("Simulate Episode End Datetime : {}".format(get_local_datetime(end_time)))
        islocal = "l" if args.islocal else "nl"
        isoption = "o" if args.isoption else "no"
        ischarging = "c" if args.ischarging else "nc"
        simulator = Simulator(start_time, TIMESTEP,args.isoption,args.islocal,args.ischarging)
        simulator.init(HEX_SHP_PATH, CS_SHP_PATH, TRIP_FILE, TRAVEL_TIME_FILE, NUM_NEAREST_CS)
        dqn_agent = DeepQNetworkOptionAgent(simulator.hex_diffusions, args.num_option, args.isoption, args.islocal, args.ischarging) if args.isoption else \
            DeepQNetworkAgent(simulator.hex_diffusions, args.isoption, args.islocal, args.ischarging)
        n_steps = int(3600 * 24 / TIMESTEP)  # number of time ticks per day
        with open('logs/parsed_results_%s_%s_%s.csv'%(isoption,islocal,ischarging), 'w') as f, open('logs/target_charging_stations_%s_%s_%s.csv'%(isoption,islocal,ischarging), 'w') as g, open('logs/training_hist_%s_%s_%s.csv'%(isoption,islocal,ischarging), 'w') as h, open('logs/demand_supply_gap_%s_%s_%s.csv'%(isoption,islocal,ischarging), 'w') as l1, open('logs/cruising_od_%s_%s_%s.csv'%(isoption,islocal,ischarging), 'w') as m1, open('logs/matching_od_%s_%s_%s.csv'%(isoption,islocal,ischarging), 'w') as n1:

            f.writelines(
                '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format("time", "num_idle", "num_serving",
                                                                        "num_charging",
                                                                        "num_cruising", "num_assigned",
                                                                        "num_waitpile",
                                                                        "num_tobedisptached", "num_offduty",
                                                                        "num_matches", "pass_arrivals",
                                                                        "longwait_pass",
                                                                        "served_pass", "removed_pass",
                                                                        "consumed_SOC_per_cycle",
                                                                        "average_cumulated_earning"))
            g.writelines('{},{},{}\n'.format("tick", "cs_id","destination_cs_id"))
            h.writelines('{},{},{},{},{},{}\n'.format("step", "loss", "reward", "learning_rate","sample_reward","sample_SOC"))
            l1.writelines('{},{},{}\n'.format("step", "hex_zone_id", "demand_supply_gap"))
            m1.writelines('{},{},{}\n'.format("step","origin_hex","destination_hex"))
            n1.writelines('{},{},{}\n'.format("step", "origin_hex", "destination_hex"))
            for episode in range(N_EPISODE):
                #reinitialize the status of the simulator
                simulator.reset(start_time=episode*(end_time-start_time),timestep=TIMESTEP)
                for day in range(SIM_DAYS):
                    print("############################ DAY {} SUMMARY ################################".format(day))
                    for i in range(n_steps):
                        tick = simulator.get_current_time()
                        start_tick = time.time()
                        simulator.step()
                        t1 = time.time() - start_tick

                        t_start=time.time()
                        local_state_batches, num_valid_relos, assigned_option_ids = simulator.get_local_states()
                        # t2 = time.time() - start_tick
                        global_state = simulator.get_global_state()
                        # t3 = time.time() - start_tick
                        # if tick >0 and np.sum(global_state) == 0: # check if just reset
                        #     global_state = global_state_slice
                        # t_act=time.time()
                        if len(local_state_batches) > 0:
                            converted_action_set, assigned_options = dqn_agent.get_actions(local_state_batches, num_valid_relos,assigned_option_ids, global_state)
                            # print('max_converted_action_id:{}'.format(np.max(converted_action_set)))
                            simulator.attach_actions_to_vehs(converted_action_set, assigned_options)
                        # print('Time for get actions:{:.3f}'.format(time.time()-t_act))
                        # t4 = time.time() - start_tick
                        simulator.update()  # update time, get metrics.
                        # t5 = time.time() - start_tick

                        simulator.summarize_metrics(f, l1, g, m1, n1)

                        # dump transitions to DQN module
                        if tick % STORE_TRANSITION_CYCLE == 0:
                            simulator.store_transitions_from_veh()
                            states, actions, next_states, rewards, terminate_flags, time_steps, valid_action_nums_ = simulator.dump_transition_to_dqn()
                            if states is not None:
                                [dqn_agent.add_transition(states, actions, next_states, rewards, terminate_flag, time_steps,  valid_action_num_) for
                                 states, actions, next_states, rewards, terminate_flag, time_steps, valid_action_num_ in zip(states, actions, next_states, rewards, terminate_flags, time_steps, valid_action_nums_)]
                                print('For episode {}, tick {}, average reward is {}'.format(episode, tick/60,np.mean(rewards)))
                            gstates=simulator.dump_global()
                            dqn_agent.add_global_state_dict(gstates)  # a 4-dim np array
                            # now reset transition and global state
                            simulator.reset_storage()

                        t6 = time.time() - start_tick
                        t_start = time.time()
                        if tick % TRAINING_CYCLE == 0:
                            dqn_agent.train(h) #do training for 10 times.
                            dqn_agent.soft_target_update(1e-3)

                        # if tick % UPDATE_CYCLE == 0:
                        #     dqn_agent.copy_parameter()
                        #     print('Now copying target network parameters.........')
                        # #update target network weight


                        t7 = time.time() - t_start
                        print('Iteration {} completed, duration: {:.3f}, total duration: {:.3f} and training: {:.3f}'.format(tick / 60, t1,t6,t7))
                        # print('Iteration {} completed, Durations: store={:.3f}; step={:.3f}; get states ={:.3f}; attach state = {:.3f}; update time = {:.3f}; dump to replaybuffer = {:.3f}; training = {:.3f}'.format(tick / 60, t0,t1,t3,t4,t5, t6,t7))

                        # if tick % (60*60*24*5) == 0 and tick != 0: # plot per 5 days while rolling over on per day.
                        #     metric_df = pd.read_csv('logs/parsed_results_%s_%s_%s.csv'%(isoption,islocal,ischarging),error_bad_lines=False)
                        #     train_df = pd.read_csv('logs/training_hist_%s_%s_%s.csv'%(isoption,islocal,ischarging),error_bad_lines=False)
                        #     auto_save_metric_plots(metric_df, train_df, tick //(60*60*24*5),N_EPISODE)

                #last step transaction dump
                tick=simulator.get_current_time()
                simulator.store_global_states()
                simulator.last_step_transactions(tick)
                states, actions, next_states, rewards, terminate_flags, time_steps, valid_action_nums_ = simulator.dump_transition_to_dqn()
                if states is not None:
                        [dqn_agent.add_transition(states, actions, next_states, rewards, terminate_flag, time_steps,
                                                  valid_action_num_) for
                         states, actions, next_states, rewards, terminate_flag, time_steps, valid_action_num_ in
                         zip(states, actions, next_states, rewards, terminate_flags, time_steps, valid_action_nums_)]
                        print('For tick {}, average reward is {}'.format(tick / 60, np.mean(rewards)))
                gstates = simulator.dump_global()
                dqn_agent.add_global_state_dict(gstates)  # a 4-dim np array

