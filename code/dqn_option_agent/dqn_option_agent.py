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
    NUM_REACHABLE_HEX, DQN_OUTPUT_DIM, OPTION_GENERATION_CYCLE, TRAIN_OPTION_CYCLE, DQN_OUTPUT_DIM
from .dqn_option_network import DQN_network, DQN_target_network
from .dqn_option_feature_constructor import FeatureConstructor
from .replay_buffers import BatchReplayMemory
from torch.optim.lr_scheduler import StepLR
from .option_sampling_agent import OptionAgent
from collections import deque
class DeepQNetworkOptionAgent:
    def __init__(self,hex_diffusion, isoption=False,islocal=True,ischarging=True):
        self.learning_rate = LEARNING_RATE  # 1e-4
        self.gamma = GAMMA
        self.start_epsilon = START_EPSILON
        self.final_epsilon = FINAL_EPSILON
        self.epsilon_steps = EPSILON_DECAY_STEPS
        self.memory = BatchReplayMemory(REPLAY_BUFFER_SIZE)
        self.batch_size = BATCH_SIZE
        self.clipping_value = CLIPPING_VALUE
        self.input_dim = INPUT_DIM
        self.relocation_dim = RELOCATION_DIM
        self.charging_dim = CHARGING_DIM
        self.num_terminate_state = OPTION_DIM
        self.output_dim = DQN_OUTPUT_DIM  #  10+7+5 = 22
        self.option_output_dim = OPTION_OUTPUT_DIM # should be 12, but currently 10.
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.path = OPTION_SAVE_PATH
        self.state_feature_constructor = FeatureConstructor()

        # init higher level DQN network
        self.q_network = DQN_network(self.input_dim, self.output_dim)
        self.target_q_network = DQN_target_network(self.input_dim, self.output_dim)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.lr_scheduler = StepLR(optimizer=self.optimizer,step_size=1000, gamma=0.99) # 1.79 e-6 at 0.5 million step.
        self.train_step = 0
        self.load_network()
        self.q_network.to(self.device)
        self.target_q_network.to(self.device)
        self.decayed_epsilon = self.start_epsilon
        # init option network
        self.option_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.lr_scheduler = StepLR(optimizer=self.option_optimizer, step_size=1000, gamma=0.99)

        self.record_list = []
        self.global_state_dict = defaultdict()
        self.time_interval = int(0)
        self.global_state_capacity = 5*1440 # we store 5 days' global states to fit replay buffer size.
        self.with_option = isoption
        self.with_charging = ischarging
        self.local_matching = islocal
        self.hex_diffusion = hex_diffusion
        self.option_queue = deque()

    def load_network(self, RESUME = False):
        if RESUME:
            lists = os.listdir(self.path)
            lists.sort(key=lambda fn: os.path.getmtime(self.path + "/" + fn))
            newest_file = os.path.join(self.path, lists[-1])
            path_checkpoint = newest_file  #'logs/test/cnn_dqn_model/duel_dqn_69120.pkl'  #
            checkpoint = torch.load(path_checkpoint)

            self.q_network.load_state_dict(checkpoint['net'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            self.train_step = checkpoint['step']
            self.copy_parameter()
            # self.optimizer.load_state_dict(checkpoint['optimizer'])
            print('Successfully load saved network starting from {}!'.format(str(self.train_step)))

    def get_actions(self, states, num_valid_relos, global_state):
        """
        :param global_states:
        :param states: tuple of (tick, hex_id, SOC) and SOC is 0 - 100%
        :param num_valid_relos: only relocation to ADJACENT hexes / charging station is valid
        :states:
        :return:
        """
        with torch.no_grad():
            self.decayed_epsilon = max(self.final_epsilon, (self.start_epsilon - self.train_step * (
                    self.start_epsilon - self.final_epsilon) / self.epsilon_steps))
            option_mask = self.get_option_mask(states) # omit the options that consider state as terminate.
            if random.random() > self.decayed_epsilon:  # epsilon = 0.1
                state_reps = [self.state_feature_constructor.construct_state_features(state) for state in states]
                hex_diffusions = [np.tile(self.hex_diffusion[state[1]],(1,1,1)) for state in states]  # state[1] is hex_id
                full_action_values = self.q_network.forward(
                    torch.from_numpy(np.array(state_reps)).to(dtype=torch.float32, device=self.device),
                    torch.from_numpy(np.concatenate([np.tile(global_state,(len(states),1,1,1)),np.array(hex_diffusions)],axis=1)).to(dtype=torch.float32, device=self.device))
                mask = self.get_action_mask(states, num_valid_relos)
                # print('take a look at processed mask {}'.format(mask))
                full_action_values[mask] = -9e10
                full_action_values[option_mask] = -9e10
                action_indexes = torch.argmax(full_action_values, dim=1).tolist()
                option_target_hex_ids = [self.option_queue[action_id].get_target_hex_id() if action_id < self.option_dim else -1 for action_id in action_indexes ]
            else:
                full_action_values = np.random.random(
                    (len(states), self.output_dim))  # generate a matrix with values from 0 to 1
                for i, state in enumerate(states):
                    full_action_values[i][(self.option_dim+num_valid_relos[i]):(self.option_dim+self.relocation_dim)] = -1
                    full_action_values[i][:self.option_dim] = -option_mask[i] # convert 1 to -1
                    if state[-1] > HIGH_SOC_THRESHOLD:
                        full_action_values[i][(self.option_dim+self.relocation_dim):] = -1  # no charging, must relocate
                    elif state[-1] < LOW_SOC_THRESHOLD:
                        full_action_values[i][:(self.option_dim+self.relocation_dim)] = -1  # no relocation, must charge

                action_indexes = np.argmax(full_action_values, 1).tolist()
                option_target_hex_ids = np.random.randint(NUM_REACHABLE_HEX, size=len(action_indexes)) # we generate a enough long list of random numbers
                option_target_hex_ids = [option_target_hex_ids[i] if action_id < self.option_dim else -1 for i, action_id in enumerate(action_indexes)]

            #  the randomly selected 10 options corresponds to top 10 destination hex_ids that are generated by OptionAgent.
            # need to translate option ids to hex_ids from Option Agent.

        return action_indexes, option_target_hex_ids

    def higher_level_dqn_update(self):
        if self.train_step % OPTION_GENERATION_CYCLE == 0:
            self.append_new_option()
        if self.train_step % TRAIN_OPTION_CYCLE == 0:
            [op.train_option_network() for op in self.option_queue]

    def append_new_option(self):
        option = OptionAgent(self.hex_diffusion)
        self.option_queue.append(option)

    def add_global_state_dict(self, global_state_list):
        current_tick = self.time_interval
        for tick in range(current_tick, int(current_tick + STORE_TRANSITION_CYCLE / 60)):
            self.global_state_dict[tick] = global_state_list[int(tick % (STORE_TRANSITION_CYCLE / 60))]
        if len(self.global_state_dict.keys()) > self.global_state_capacity:
            for previous_tick in range(int(current_tick - self.global_state_capacity), int(
                    current_tick+ STORE_TRANSITION_CYCLE / 60 - self.global_state_capacity)):
                self.global_state_dict.pop(previous_tick)
        # self.global_state_dict[self.time_interval:self.time_interval+STORE_TRANSITION_CYCLE] = global_state
        self.time_interval += int(STORE_TRANSITION_CYCLE / 60)

    def add_transition(self, state, action, next_state, reward, terminate_flag, time_steps, valid_action):
        self.memory.push(state, action, next_state, reward, terminate_flag, time_steps, valid_action)

    def batch_sample(self):
        samples = self.memory.sample(self.batch_size)  # random.sample(self.memory, self.batch_size)
        return samples
        # state, action, next_state, reward = zip(*samples)
        # return state, action, next_state, reward

    def get_main_Q(self, local_state, global_state):
        return self.q_network.forward(local_state, global_state)

    def get_target_Q(self, local_state, global_state):
        return self.target_q_network.forward(local_state, global_state)

    def copy_parameter(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def train(self, record_hist):
        self.train_step += 1
        if len(self.memory) < self.batch_size:
            print('batches in replay buffer is {}'.format(len(self.memory)))
            return

        transitions = self.batch_sample()
        batch = self.memory.Transition(*zip(*transitions))

        global_state_reps = [self.global_state_dict[int(state[0] / 60)] for state in
                             batch.state]  # should be list of np.array

        global_next_state_reps = [self.global_state_dict[int(state_[0] / 60)] for state_ in
                                  batch.next_state]  # should be list of np.array

        state_reps = [self.state_feature_constructor.construct_state_features(state) for state in batch.state]
        next_state_reps = [self.state_feature_constructor.construct_state_features(state_) for state_ in
                           batch.next_state]

        hex_diffusion = [np.tile(self.hex_diffusion[state[1]],(1,1,1)) for state in batch.state]
        hex_diffusion_ = [np.tile(self.hex_diffusion[state_[1]],(1,1,1)) for state_ in batch.next_state]

        state_batch = torch.from_numpy(np.array(state_reps)).to(dtype=torch.float32, device=self.device)
        action_batch = torch.from_numpy(np.array(batch.action)).unsqueeze(1).to(dtype=torch.int64, device=self.device)
        reward_batch = torch.from_numpy(np.array(batch.reward)).unsqueeze(1).to(dtype=torch.float32, device=self.device)
        time_step_batch = torch.from_numpy(np.array(batch.time_steps)).unsqueeze(1).to(dtype=torch.float32, device=self.device)

        next_state_batch = torch.from_numpy(np.array(next_state_reps)).to(device=self.device, dtype=torch.float32)
        global_state_batch = torch.from_numpy(np.concatenate([np.array(global_state_reps),np.array(hex_diffusion)],axis=1)).to(dtype=torch.float32, device=self.device)
        global_next_state_batch = torch.from_numpy(np.concatenate([np.array(global_next_state_reps), np.array(hex_diffusion_)],axis=1)).to(dtype=torch.float32,
                                                                                        device=self.device)

        q_state_action = self.get_main_Q(state_batch, global_state_batch).gather(1, action_batch.long())
        # add a mask
        all_q_ = self.get_target_Q(next_state_batch, global_next_state_batch)
        mask = self.get_action_mask(batch.next_state, batch.valid_action_num)  # action mask for next state
        option_mask = self.get_option_mask(batch.next_state)
        all_q_[mask] = -9e10
        all_q_[option_mask] = -9e10
        maxq = all_q_.max(1)[0].detach().unsqueeze(1)
        delta_f_value = [self.option_queue[action_id].get_f_value([state_batch[id][1]]) - self.option_queue[action_id].get_f_value([next_state_batch[id][1]]) \
                             if action_id < self.option_dim else 0 for id,action_id in enumerate(batch.action)]
        y = delta_f_value + reward_batch + maxq*torch.pow(self.gamma,time_step_batch)
        loss = F.smooth_l1_loss(q_state_action, y)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.clipping_value)
        self.optimizer.step()
        self.lr_scheduler.step()

        self.record_list.append([self.train_step, round(float(loss),3), round(float(reward_batch.view(-1).mean()),3),self.optimizer.state_dict()['param_groups'][0]['lr'],batch.reward[0], batch.state[0][-1]])
        self.save_parameter(record_hist)
        print('Training step is {}; Learning rate is {}; Epsilon is {}:'.format(self.train_step,self.lr_scheduler.get_lr(),round(self.decayed_epsilon,4)))

    def get_action_mask(self, batch_state, batch_valid_action):
        """
        the dim of action is 22 in total.
        first ten is for options, then next 7 is for reposition, last 5 is for charging.
        :param batch_state:
        :param batch_valid_action:
        :return:
        """
        mask = np.zeros([len(batch_state), self.output_dim])
        for i, state in enumerate(batch_state):
            mask[i][(self.option_dim+ batch_valid_action[i]):(self.option_dim+self.relocation_dim)] = 1
            # here the SOC in state is still continuous. the categorized one is in state reps.
            if state[-1] > HIGH_SOC_THRESHOLD:
                mask[i][(self.option_dim+self.relocation_dim):] = 1  # no charging, must relocate
            elif state[-1] < LOW_SOC_THRESHOLD:
                mask[i][:(self.option_dim+self.relocation_dim)] = 1  # no relocation, must charge

        mask = torch.from_numpy(mask).to(dtype=torch.bool, device=self.device)
        return mask

    def get_option_mask(self,states):
        """
        append masks to the options that think the states as terminate.
        if no option is generated yet, we keep the option masked as 1. (we initial the mask by np.ones)
        :param states:
        :return:
        """
        termiante_option_mask = np.ones((len(states),self.option_dim))
        if self.option_queue:
            for i,op in enumerate(self.option_queue):
                mask_col = [0 if op.get_f_value([state[1]]) > op.lower_threshold else 1 for state in states]
                termiante_option_mask[:,i] = mask_col # the j-th element of the i-th column is masked

        termiante_option_mask = torch.from_numpy(termiante_option_mask).to(dtype=torch.bool, device=self.device)
        return termiante_option_mask


    def save_parameter(self, record_hist):
        # torch.save(self.q_network.state_dict(), self.dqn_path)
        if self.train_step % SAVING_CYCLE == 0:
            checkpoint = {
                "net": self.q_network.state_dict(),
                # 'optimizer': self.optimizer.state_dict(),
                "step": self.train_step,
                "lr_scheduler": self.lr_scheduler.state_dict()
            }
            if not os.path.isdir(self.path):
                os.mkdir(self.path)
            # print('the path is {}'.format('logs/dqn_model/duel_dqn_%s.pkl'%(str(self.train_step))))
            torch.save(checkpoint, 'logs/test/cnn_dqn_model/duel_dqn_%d_%d_%d_%s.pkl' % (bool(self.with_option),bool(self.with_charging),bool(self.local_matching),str(self.train_step)))
            # record training process (stacked before)
            for item in self.record_list:
                record_hist.writelines('{},{},{},{},{},{}\n'.format(item[0], item[1], item[2], item[3], item[4], item[5]))
            print('Training step: {}, replay buffer size:{}, epsilon: {}, learning rate: {}'.format(self.record_list[-1][0],len(self.memory), self.decayed_epsilon,self.lr_scheduler.get_lr()))
            self.record_list = []

