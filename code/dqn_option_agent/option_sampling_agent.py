import os
import random
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.setting import LEARNING_RATE, GAMMA, REPLAY_BUFFER_SIZE, BATCH_SIZE, RELOCATION_DIM, CHARGING_DIM, \
    INPUT_DIM, OPTION_DIM, FINAL_EPSILON, HIGH_SOC_THRESHOLD, LOW_SOC_THRESHOLD, CLIPPING_VALUE, START_EPSILON, \
    EPSILON_DECAY_STEPS, OPTION_SAVE_PATH, SAVING_CYCLE, CNN_RESUME, STORE_TRANSITION_CYCLE, OPTION_OUTPUT_DIM, \
    NUM_REACHABLE_HEX, DQN_OUTPUT_DIM, OPTION_BUFFER_SIZE, OPTION_BATCH_SIZE
from .option_network import Option_Network, Option_Target_Network
from .dqn_option_feature_constructor import FeatureConstructor
from torch.optim.lr_scheduler import StepLR
from .replay_buffers import OptionReplayMemory
from collections import deque


class H_Agent:
    """
    todo 1: logic is to first train options
    todo 2: enlarge the action space to 12+10, additional 10 are options that are sampled ~epsilon greedy policy. [done]
    todo 3: embed an Option-DQN module to generate the 10 options.
    todo 4: carefully design the training module of Option-DQN, with the training loss being delta f_value, refer to deep covering option.
    todo 5: how to covert option-id to hex_id, so that we can correctly interpret them and dispatch the vehicle to right place.
    """
    def __init__(self,hex_diffusion, device, isoption=False,islocal=True,ischarging=True):
        self.learning_rate = LEARNING_RATE  # 1e-4
        self.gamma = GAMMA
        self.start_o_epsilon = START_EPSILON
        self.final_o_epsilon = FINAL_EPSILON
        self.epsilon_o_steps = EPSILON_DECAY_STEPS
        self.option_memory = OptionReplayMemory(OPTION_BUFFER_SIZE)  # 1e4
        self.option_batch_size = OPTION_BATCH_SIZE
        self.clipping_value = CLIPPING_VALUE
        self.input_dim = INPUT_DIM
        self.option_dim = OPTION_DIM
        self.option_output_dim = OPTION_OUTPUT_DIM # should be 12 then sampling
        self.device = device
        self.path = OPTION_SAVE_PATH
        self.state_feature_constructor = FeatureConstructor()
        self.decayed_o_epsilon = self.start_o_epsilon
        # init option network
        self.option_network = Option_Network(self.input_dim, self.option_output_dim)  # output a policy
        self.option_target_network = Option_Target_Network(self.input_dim, self.option_output_dim)
        self.option_optimizer = torch.optim.Adam(self.option_network.parameters(), lr=self.learning_rate)
        self.option_lr_scheduler = StepLR(optimizer=self.option_optimizer, step_size=1000, gamma=0.99)
        self.option_train_step = 0
        self.option_network.to(self.device)
        self.option_target_network.to(self.device)

        self.record_list = []
        self.global_state_dict = defaultdict()
        self.time_interval = int(0)
        self.global_state_capacity = 5*1440 # we store 5 days' global states to fit replay buffer size.
        self.with_option = isoption
        self.with_charging = ischarging
        self.local_matching = islocal
        self.hex_diffusion = hex_diffusion
        self.option_queue = deque()
        self.init_func = None
        self.term_func = None
        self.init_dist = 0.05
        self.term_dist = 0.05
        self.upper_threshold = 1e5
        self.lower_threshold = 1e5

    def get_f_value_by_full_state(self,local_state,global_state):
        return self.option_network.forward(local_state)  # ,global_state)

    def get_f_value(self, hex_ids):

        hex_diffusions = [np.tile(self.hex_diffusion[int(hex_id)], (1, 1, 1)) for hex_id in hex_ids]  # state[1] is hex_id
        return self.option_network.forward(torch.from_numpy(np.array(hex_diffusions)).to(dtype=torch.float32, device=self.device))

    def get_target_hex_id(self):
        """
        todo: use soft max to enumerate 1347 hexes ==> softmax(min)? ==> select target hex id
        todo: change beta later
        :return:
        """
        beta = 0.2
        f_values = self.get_f_value([i for i in range(NUM_REACHABLE_HEX)])
        f_total = sum([np.exp(-beta * fv) for fv in f_values])
        f_softmin = [np.exp(-beta * fv) / f_total for fv in f_values]
        hex_id = np.random.choice(NUM_REACHABLE_HEX, 1, p=f_softmin)[0] # select the hex indicates the terminate state.
        return hex_id

    def is_initial(self,state):
        if self.init_func is None:
            return True
        else:
            f_value = self.get_f_value(state[1])
            return self.init_func(f_value)

    def is_terminal(self,state):
        if self.term_func is None:
            return True
        else:
            f_value = self.get_f_value(state[1])
            return self.term_func(f_value)


    def train_option_network(self):
        self.option_train_step += 1
        if len(self.option_memory) < self.option_batch_size:
            print('batches in option replay buffer is {}'.format(len(self.option_memory)))
            return

        transitions = self.option_memory.sample(self.option_batch_size)
        obatch = self.option_memory.Transition(*zip(*transitions))

        # global_state_reps = [self.global_state_dict[int(state[0] / 60)] for state in
        #                      obatch.state]  # should be list of np.array
        #
        # global_next_state_reps = [self.global_state_dict[int(state_[0] / 60)] for state_ in
        #                           obatch.next_state]  # should be list of np.array
        #
        # state_reps = [self.state_feature_constructor.construct_state_features(state) for state in obatch.state]
        # next_state_reps = [self.state_feature_constructor.construct_state_features(state_) for state_ in
        #                    obatch.next_state]

        # state_batch = torch.from_numpy(np.array(state_reps)).to(dtype=torch.float32, device=self.device)
        time_step_batch = torch.from_numpy(np.array(obatch.time_steps)).unsqueeze(1).to(dtype=torch.float32, device=self.device)

        # next_state_batch = torch.from_numpy(np.array(next_state_reps)).to(device=self.device, dtype=torch.float32)
        # global_state_batch = torch.from_numpy(np.concatenate([np.array(global_state_reps),np.array(hex_diffusion)],axis=1)).to(dtype=torch.float32, device=self.device)
        # global_next_state_batch = torch.from_numpy(np.concatenate([np.array(global_next_state_reps), np.array(hex_diffusion_)],axis=1)).to(dtype=torch.float32,
        #                                                                                 device=self.device)

        f_values = torch.from_numpy(np.array(self.get_f_value([next_state[1] for next_state in obatch.state]))).to(dtype=torch.int64, device=self.device)
        # add a mask
        f_values_ = torch.from_numpy(np.array(self.get_f_value([next_state[1] for next_state in obatch.next_state]))).to(dtype=torch.int64, device=self.device)

        y = time_step_batch + f_values_
        loss = F.smooth_l1_loss(f_values, y)

        self.option_optimizer.zero_grad()
        loss.backward()
        self.option_optimizer.step()
        self.option_lr_scheduler.step()
        self.upper_threshold, self.lower_threshold = self.sample_f_value(f_values,self.init_dist,self.term_dist)
        self.term_func = lambda x: x< self.lower_threshold
        self.init_func = lambda x: x< self.upper_threshold

    def sample_f_value(self, f_values,init_percentile, term_percentile):

        f_sort = np.sort(f_values)
        init_threshold = f_sort[int(len(self.option_batch_size)*init_percentile)]
        term_threshold = f_sort[int(len(self.option_batch_size)*term_percentile)]
        return init_threshold, term_threshold
