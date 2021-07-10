import argparse
import time
import numpy as np
import pandas as pd
from common.time_utils import get_local_datetime
from config.hex_setting import HEX_SHP_PATH, CS_SHP_PATH, NUM_NEAREST_CS, TRIP_FILE, TRAVEL_TIME_FILE, \
    TIMESTEP, START_OFFSET, SIM_DAYS, N_EPISODE, START_TIME, TRAINING_CYCLE, UPDATE_CYCLE, STORE_TRANSITION_CYCLE, CNN_RESUME, OPTION_DIM,LEARNING_RATE
# from dqn_option_agent.dqn_option_agent import DeepQNetworkOptionAgent
from dqn_agent.dqn_agent_with_cnn.cnn_dqn_agent import DeepQNetworkAgent
from simulator.simulator_cnn import Simulator

# ---------------MAIN FILE---------------

if __name__ == '__main__':
    arg = argparse.ArgumentParser("Start running")

    arg.add_argument("--islocal", "-l", default=1, type=bool, help="choose local matching instead of global matching")
    arg.add_argument("--isoption", "-o", default=0, type=bool, help="choose covering option or not")
    arg.add_argument("--ischarging", "-c", default=0, type=bool, help="choose charging option or not")
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
        dqn_agent = DeepQNetworkAgent(simulator.hex_diffusions,OPTION_DIM, args.isoption,args.islocal,args.ischarging)
        dqn_agent.neighbor_id=simulator.all_neighbors # 1347 by 7 matrix
        # DeepQNetworkOptionAgent(simulator.hex_diffusions,isoption,LEARNING_RATE,ischarging)
        n_steps = int(3600 * 24 / TIMESTEP)  # number of time ticks per day
        with open('logs/parsed_results_{}_{}_{}.csv'.format(OPTION_DIM,LEARNING_RATE,ischarging), 'w') as f, open('logs/target_charging_stations_{}_{}_{}.csv'.format(OPTION_DIM,LEARNING_RATE,ischarging), 'w') as g, open('logs/training_hist_{}_{}_{}.csv'.format(OPTION_DIM,LEARNING_RATE,ischarging), 'w') as h, open('logs/demand_supply_gap_{}_{}_{}.csv'.format(OPTION_DIM,LEARNING_RATE,ischarging), 'w') as l1, open('logs/cruising_od_{}_{}_{}.csv'.format(OPTION_DIM,LEARNING_RATE,ischarging), 'w') as m1, open('logs/matching_od_{}_{}_{}.csv'.format(OPTION_DIM,LEARNING_RATE,ischarging), 'w') as n1:

            f.writelines(
                '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format("time", "num_idle", "num_serving",
                                                                        "num_charging",
                                                                        "num_cruising", "num_assigned",
                                                                        "num_waitpile",
                                                                        "num_tobedisptached", "average_idle_time",
                                                                        "num_matches", "average_idle_dist",
                                                                        "longwait_pass",
                                                                        "served_pass", "removed_pass",
                                                                        "consumed_SOC_per_cycle",
                                                                        "total_system_revenue"))
            g.writelines('{},{},{}\n'.format("tick", "cs_id","destination_cs_id"))
            h.writelines('{},{},{},{},{},{}\n'.format("step", "loss", "reward", "learning_rate","sample_reward","sample_SOC"))
            l1.writelines('{},{},{}\n'.format("step", "hex_zone_id", "demand_supply_gap"))
            m1.writelines('{},{},{}\n'.format("step","origin_hex","destination_hex"))
            n1.writelines('{},{},{}\n'.format("step", "origin_hex", "destination_hex"))
            for episode in range(N_EPISODE):
                #reinitialize the status of the simulator
                simulator.reset(start_time=episode*(end_time-start_time),timestep=TIMESTEP,seed=int(episode%3))
                dqn_agent.reset_f_storage()
                for day in range(SIM_DAYS):
                    print("############################ DAY {} SUMMARY ################################".format(day))
                    for i in range(n_steps):
                        tick = simulator.get_current_time()
                        start_tick = time.time()
                        global_state = simulator.get_global_state()
                        local_state_batches, num_valid_relos,assigned_option_ids = simulator.get_local_states()
                        nidle=[veh for hx in simulator.hex_zone_collection.values() for veh in hx.vehicles.values() if veh.state.status==0] #idle vehicles

                        #set f
                        if OPTION_DIM>0:
                            dqn_agent.get_local_f(global_state,tick)



                        if len(local_state_batches) > 0:
                            # dump terminal transitions for those vehicle
                            if OPTION_DIM>0:
                                terminal_flag = dqn_agent.is_local_terminal(local_state_batches, global_state)
                                terminal_veh=0
                                for flag, veh in zip(terminal_flag,nidle):
                                    if flag > 0:
                                        if veh.assigned_option>=0:
                                            veh.save_terminal(tick); terminal_veh+=1

                            action_selected,action_to_execute,assigned_opts,contd_opts=dqn_agent.get_actions(local_state_batches, num_valid_relos, global_state,assigned_option_ids)

                            if OPTION_DIM==0:
                                simulator.attach_actions_to_vehs(action_selected,action_to_execute)
                            else:
                                simulator.attach_actions_to_vehs(action_selected,action_to_execute,assigned_opts,contd_opts)
                                if len(action_selected)>0:
                                    opt_select = sum([1 for a in action_selected if a == 0]) / len(action_selected)
                                    stay_select = sum([1 for a in action_selected if a == 1]) / len(action_selected)
                                    contd_prop = sum([1 for a in contd_opts if a == True]) / len(contd_opts + 0.00001)
                                    print(
                                        'Propotion of option selected={:.2f}, percent of continued option={:.2f},propotion of stay selected={:.2f}'.format(
                                            opt_select, contd_prop, stay_select))
                                    dqn_agent.writer.add_scalar('main_dqn/option_percent', opt_select, dqn_agent.train_step)
                                    dqn_agent.writer.add_scalar('main_dqn/stay_percent', stay_select,
                                                                dqn_agent.train_step)

                        simulator.step()
                        t1 = time.time() - start_tick

                        # t2 = time.time() - start_tick

                        # t3 = time.time() - start_tick
                        # if tick >0 and np.sum(global_state) == 0: # check if just reset
                        #     global_state = global_state_slice

                        # t4 = time.time() - start_tick
                        simulator.update()  # update time, get metrics.
                        # t5 = time.time() - start_tick
                        (num_idle, num_serving, num_charging, num_cruising, n_matches, total_num_arrivals,
                         total_removed_passengers, num_assigned, num_waitpile, num_tobedisptached, num_offduty,
                         average_reduced_SOC, total_num_longwait_pass, total_num_served_pass, average_cumulated_earning) = simulator.summarize_metrics(l1, g, m1, n1)


                        f.writelines(
                            '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(tick, num_idle, num_serving,
                                                                                    num_charging,
                                                                                    num_cruising, num_assigned,
                                                                                    num_waitpile,
                                                                                    num_tobedisptached, num_offduty,
                                                                                    n_matches,
                                                                                    total_num_arrivals,
                                                                                    total_num_longwait_pass,
                                                                                    total_num_served_pass,
                                                                                    total_removed_passengers,
                                                                                    average_reduced_SOC,
                                                                                    average_cumulated_earning))
                        # dump transitions to DQN module
                        if tick % STORE_TRANSITION_CYCLE == 0:
                            simulator.store_transitions_from_veh()
                            simulator.store_f_action_from_veh()
                            simulator.store_prime_action_from_veh()
                            states, actions, next_states, rewards, terminate_flags, time_steps, valid_action_nums_ = simulator.dump_transition_to_dqn()
                            if states is not None:
                                [dqn_agent.add_transition(states, actions, next_states, rewards, terminate_flag, time_steps,  valid_action_num_) for
                                 states, actions, next_states, rewards, terminate_flag, time_steps, valid_action_num_ in zip(states, actions, next_states, rewards, terminate_flags, time_steps, valid_action_nums_)]
                                print('For episode {}, tick {}, average reward is {},replay buffer={}'.format(episode, tick/60,np.mean(rewards),len(dqn_agent.memory.memory)))
                            gstates=simulator.dump_global()
                            print('now adding global state...size=',len(gstates.keys()))
                            dqn_agent.add_global_state_dict(gstates)  # a 4-dim np array

                            t1=time.time()
                            states, next_states = simulator.dump_f_transitions()
                            if states is not None:
                                [dqn_agent.add_f_transition([state, next_state]) for
                                 state, next_state in
                                 zip(states, next_states)]
                            print('Add f_transition cost={}'.format(time.time()-t1))

                            t1=time.time()
                            states, next_states = simulator.dump_fo_transitions()
                            if states is not None:
                                [dqn_agent.add_fo_transition([state, next_state]) for
                                 state, next_state in
                                 zip(states, next_states)]
                            print('Add fo_transition cost={}'.format(time.time()-t1))
                            t1=time.time()


                            states, actions, next_states, trip_flags, time_steps, valid_action_nums_ = simulator.dump_prime_action_to_dqn()
                            if states is not None:
                                [dqn_agent.add_H_transition(state, action, next_state, flag, time, valid_loc) for
                                 state, action, next_state, flag, time, valid_loc in
                                 zip(states, actions, next_states, trip_flags, time_steps, valid_action_nums_)]
                            print('Add H_transition cost={}'.format(time.time() - t1))
                            # now reset transition and global state
                            simulator.reset_storage()

                        t6 = time.time() - start_tick
                        t_start = time.time()
                        if tick % TRAINING_CYCLE == 0:
                            dqn_agent.train(h) #do training for 10 times.
                            # dqn_agent.soft_target_update(1e-3)

                        if tick % UPDATE_CYCLE ==0:
                            dqn_agent.copy_parameter()
                            print('Now copying target network parameters.........')

                        t7 = time.time() - t_start
                        # print('Iteration {} completed, duration: {:.3f} and training: {:.3f}'.format(tick / 60, t1,t7))
                        # print('Iteration {} completed, Durations: store={:.3f}; step={:.3f}; get states ={:.3f}; attach state = {:.3f}; update time = {:.3f}; dump to replaybuffer = {:.3f}; training = {:.3f}'.format(tick / 60, t0,t1,t3,t4,t5, t6,t7))

                        # if tick % (60*60*24*5) == 0 and tick != 0: # plot per 5 days while rolling over on per day.
                        #     metric_df = pd.read_csv('logs/parsed_results_{}_{}_{}.csv'.format(OPTION_DIM,LEARNING_RATE,ischarging),error_bad_lines=False)
                        #     train_df = pd.read_csv('logs/training_hist_{}_{}_{}.csv'%(OPTION_DIM,LEARNING_RATE,ischarging),error_bad_lines=False)
                        #     auto_save_metric_plots(metric_df, train_df, tick //(60*60*24*5),N_EPISODE)

                if OPTION_DIM>0:
                    print('-------------------------------Now updating the target f network!!!--------------------')
                    dqn_agent.summarize_median()
                    # dqn_agent.record_f_threshold()
                    # dqn_agent.record_fo_threshold()
                    # dqn_agent.f_memory.reset()
                    # dqn_agent.fo_memory.reset()
                    # dqn_agent.reset_h()
                    dqn_agent.train_f()
                    # for hs in range(1000):
                    #     dqn_agent.train_h()
                    #     if hs%50==0:
                    #         dqn_agent.copy_H_parameter()
                    # # dqn_agent.soft_f_update(1) #update the f network with the new values
                    # dqn_agent.copy_H_parameter()

                    # dqn_agent.train_fo()

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

                states, next_states = simulator.dump_f_transitions()
                if states is not None:
                    [dqn_agent.add_f_transition([state, next_state]) for
                     state, next_state in
                     zip(states, next_states)]
                print('Add f_transition cost={}'.format(time.time() - t1))
                t1 = time.time()
                states, actions, next_states, trip_flags, time_steps, valid_action_nums_ = simulator.dump_prime_action_to_dqn()
                if states is not None:
                    [dqn_agent.add_H_transition(state, action, next_state, flag, time, valid_loc) for
                     state, action, next_state, flag, time, valid_loc in
                     zip(states, actions, next_states, trip_flags, time_steps, valid_action_nums_)]
                print('Add H_transition cost={}'.format(time.time() - t1))
                # now reset transition and global state


                gstates = simulator.dump_global()
                dqn_agent.add_global_state_dict(gstates)  # a 4-dim np array
                simulator.reset_storage()

                dqn_agent.save_parameter()

