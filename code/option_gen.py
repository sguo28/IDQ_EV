###import information
import argparse
import time
import numpy as np
import pandas as pd
from collections import OrderedDict
from common.time_utils import get_local_datetime
from config.hex_setting import HEX_SHP_PATH, CS_SHP_PATH, NUM_NEAREST_CS, TRIP_FILE, TRAVEL_TIME_FILE, \
    TIMESTEP, START_OFFSET, SIM_DAYS, N_EPISODE, START_TIME, TRAINING_CYCLE, UPDATE_CYCLE, STORE_TRANSITION_CYCLE, CNN_RESUME, OPTION_DIM
# from dqn_option_agent.dqn_option_agent import DeepQNetworkOptionAgent
from dqn_agent.dqn_agent_with_cnn.cnn_dqn_agent import DeepQNetworkAgent
from simulator.simulator_cnn import Simulator
from dqn_option_agent.f_approx_agent import F_Agent
from dqn_agent.dqn_agent_with_cnn.cnn_dqn_agent_fh import DeepQNetworkAgent
from dqn_option_agent.h_agent import H_Agent
from dqn_agent.replay_buffer import Prime_ReplayMemory

#load simulator
start_time = 0  # start_time = 0
print("Simulate Episode Start Datetime: {}".format(get_local_datetime(start_time)))
end_time = start_time + int(60 * 60 * 24 * SIM_DAYS)
print("Simulate Episode End Datetime : {}".format(get_local_datetime(end_time)))
simulator = Simulator(start_time, TIMESTEP, isoption=True, ischarging=False,islocal=True) #local matching
simulator.init(HEX_SHP_PATH, CS_SHP_PATH, TRIP_FILE, TRAVEL_TIME_FILE, NUM_NEAREST_CS)
hex_diff=simulator.hex_diffusions
#lets get the hex coordinates as well
hc=simulator.hex_zone_collection
hex_to_xy=np.array([[hc[i].row_id,hc[i].col_id] for i in range(len(simulator.hex_zone_collection.values()))])

h_options=[]
# f_functions=F_Agent(num_options=0,hex_diffusion=hex_diff)

total_opts = int(3)
episode_f = 3 #number of episode to train f
episode_h = 3
n_steps = int(3600 * 24 / TIMESTEP)
for i in range(1):
    #run simulator at random at train f
    n_steps = int(3600 * 24 / TIMESTEP)  # number of time ticks per day
    dqn_net=DeepQNetworkAgent(option_num=i,hex_diffusion=hex_diff,h_network=list(h_options))
    f_functions = F_Agent(num_options=i, hex_diffusion=hex_diff)

    #update the terminal states
    if i>0:
        for m in simulator.match_zone_collection:
            m.terminal_states=dqn_net.middle_terminal
    #
    H_global=OrderedDict()
    H_memory=Prime_ReplayMemory(int(episode_f*1e6)) #this is used to train h
    # #
    for episode in range(episode_f):
        simulator.reset(start_time=episode * (end_time - start_time), timestep=TIMESTEP)
        #set the seed
        seed=episode%3
        for hex in simulator.hex_zone_collection.values():
            hex.seed=seed

        print('Running f_training number of options={}, episode={},step={}'.format(i, episode, 1440))
        for j in range(n_steps):
            print('Running f_training number of options={}, episode={},step={}'.format(i, episode, j))
            tick = simulator.get_current_time()
            start_tick = time.time()
            global_state = simulator.get_global_state()
            local_state_batches, num_valid_relos, assigned_option_ids = simulator.get_local_states()
            if len(local_state_batches) > 0:
                action_selected, action_to_execute, assigned_opts, contd_opts = dqn_net.get_actions(
                    local_state_batches, num_valid_relos, global_state, assigned_option_ids)
                if  i== 0:
                    simulator.attach_actions_to_vehs(action_selected, action_to_execute)
                else:
                    simulator.attach_actions_to_vehs(action_selected, action_to_execute, assigned_opts, contd_opts)
            simulator.step()
            simulator.update()  # update time, get metrics.
            if tick % STORE_TRANSITION_CYCLE == 0:
                simulator.store_f_action_from_veh()
                simulator.store_prime_action_from_veh()
                states, next_states = simulator.dump_f_transitions()
                if states is not None:
                    [f_functions.add_data([state, next_state]) for
                     state, next_state in
                     zip(states, next_states)]
                # print('Number of f training data per hour=', [len(hh) for hh in f_functions.training_data])
                # print('Replay buffer size=', len(H_memory.memory))
                #add prime actions
                states, actions, next_states, trip_flags, time_steps, valid_action_nums_ = simulator.dump_prime_action_to_dqn()
                if states is not None:
                    [H_memory.push(state,action,next_state,flag,time,valid_loc) for state,action,next_state,flag,time,valid_loc in zip(states, actions, next_states, trip_flags, time_steps, valid_action_nums_)]
                #append global state
                gstates = simulator.dump_global()
                for key in gstates.keys():
                    if key not in H_global.keys():
                        H_global[key] = gstates[key]
               # now reset transition and global state
                simulator.reset_storage()

        print('simulation episode completed, now start F training')

        f_functions.train(1)
        f_functions.reset_memory()

    print('saving f values')
    f_functions.save_f_vals(i)  #save the f values as a csv


    h_agent=H_Agent(hex_diffusion=hex_diff,num_option=i,H_memory=H_memory,H_global=H_global)
    h_agent.hex_xy=hex_to_xy

    for train_step in range(1440*episode_f): #train for 20000 steps
        h_agent.train()
        if train_step%500==0: #update target networsk
            h_agent.copy_parameter()

    #update the terminal sets used by the simulator
    # for episode in range(episode_h):
    #     simulator.reset(start_time=episode * (end_time - start_time), timestep=TIMESTEP)
    #     print('Running h training, number of options={}, episode={},step={}'.format(i, episode, n_steps))
    #     for j in range(n_steps):
    #         tick = simulator.get_current_time()
    #         start_tick = time.time()
    #
    #         global_state = simulator.get_global_state()
    #         local_state_batches, num_valid_relos, assigned_option_ids = simulator.get_local_states()
    #         if len(local_state_batches) > 0:
    #             action_selected, action_to_execute, assigned_opts, contd_opts = dqn_net.get_actions(
    #                 local_state_batches, num_valid_relos, global_state, assigned_option_ids)
    #             if  i== 0:
    #                 simulator.attach_actions_to_vehs(action_selected, action_to_execute)
    #             else:
    #                 simulator.attach_actions_to_vehs(action_selected, action_to_execute, assigned_opts, contd_opts)
    #
    #         simulator.step()
    #         simulator.update()  # update time, get metrics.
    #         if tick % STORE_TRANSITION_CYCLE == 0:
    #             #simulator.store_f_action_from_veh()
    #             simulator.store_prime_action_from_veh()
    #             states, actions, next_states, trip_flags, time_steps, valid_action_nums_ = simulator.dump_prime_action_to_dqn()
    #             if states is not None:
    #                 [h_agent.add_transition(state,action,next_state,flag,time,valid_loc) for state,action,next_state,flag,time,valid_loc in zip(states, actions, next_states, trip_flags, time_steps, valid_action_nums_)]
    #             #append global state
    #             gstates = simulator.dump_global()
    #             h_agent.add_global_state_dict(gstates)
    #            # now reset transition and global state
    #             simulator.reset_storage()
    #
    #         h_agent.train()
    #
    #         if tick%(500*60)==0:
    #             print('Update h target network parameter')
    #             h_agent.copy_parameter()


    h_agent.save_parameter() #save the network
    # for episode in range(episode_h):
    #     train....
    # #next train h
    # h_agent=





