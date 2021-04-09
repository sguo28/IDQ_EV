import os
import random
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.hex_setting import LEARNING_RATE, GAMMA, REPLAY_BUFFER_SIZE, BATCH_SIZE, RELOCATION_DIM, CHARGING_DIM, \
    INPUT_DIM, OUTPUT_DIM, FINAL_EPSILON, HIGH_SOC_THRESHOLD, LOW_SOC_THRESHOLD, CLIPPING_VALUE, START_EPSILON, \
    EPSILON_DECAY_STEPS, CNN_SAVE_PATH, SAVING_CYCLE, CNN_RESUME, STORE_TRANSITION_CYCLE
from dqn_agent.dqn_feature_constructor import FeatureConstructor
from dqn_agent.replay_buffer import ReplayMemory
from .cnn_dqn_network import DQN_network, DQN_target_network
from torch.optim.lr_scheduler import StepLR


class DeepQNetworkAgent:
    def __init__(self,NO_CHARGE=False, GLOBAL_MATCHING = False):
        self.learning_rate = LEARNING_RATE  # 1e-4
        self.gamma = GAMMA
        self.start_epsilon = START_EPSILON
        self.final_epsilon = FINAL_EPSILON
        self.epsilon_steps = EPSILON_DECAY_STEPS
        self.memory = ReplayMemory(REPLAY_BUFFER_SIZE)
        self.batch_size = BATCH_SIZE
        self.clipping_value = CLIPPING_VALUE
        self.input_dim = INPUT_DIM
        self.relocation_dim = RELOCATION_DIM
        self.charging_dim = CHARGING_DIM
        self.output_dim = OUTPUT_DIM
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.path = CNN_SAVE_PATH
        self.state_feature_constructor = FeatureConstructor()
        self.q_network = DQN_network(self.input_dim, self.output_dim)
        self.target_q_network = DQN_target_network(self.input_dim, self.output_dim)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.lr_scheduler = StepLR(optimizer=self.optimizer,step_size=1250, gamma=0.99) # 1.79 e-6 at 0.5 million step.
        self.train_step = 0
        self.load_network()
        self.q_network.to(self.device)
        self.target_q_network.to(self.device)
        self.decayed_epsilon = self.start_epsilon
        self.record_list = []
        self.global_state_dict = defaultdict()
        self.time_interval = int(0)
        self.global_state_capacity = 5*1440 # we store 5 days' global states to fit replay buffer size.
        self.no_charge = NO_CHARGE
        self.global_matching = GLOBAL_MATCHING

    def load_network(self):
        if CNN_RESUME:
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
            if random.random() > self.decayed_epsilon:  # epsilon = 0.1
                state_reps = [self.state_feature_constructor.construct_state_features(state) for state in states]
                full_action_values = self.q_network.forward(
                    torch.from_numpy(np.array(state_reps)).to(dtype=torch.float32, device=self.device),
                    torch.from_numpy(np.tile(global_state,(len(states),1,1,1))).to(dtype=torch.float32, device=self.device))
                mask = self.get_action_mask(states, num_valid_relos)
                # print('take a look at processed mask {}'.format(mask))
                full_action_values[mask] = -9e10

                action_indexes = torch.argmax(full_action_values, dim=1).tolist()
            else:
                full_action_values = np.random.random(
                    (len(states), self.output_dim))  # generate a matrix with values from 0 to 1
                for i, state in enumerate(states):
                    full_action_values[i][num_valid_relos[i]:self.relocation_dim] = -1
                    if state[-1] > HIGH_SOC_THRESHOLD:
                        full_action_values[i][self.relocation_dim:] = -1  # no charging, must relocate
                    elif state[-1] < LOW_SOC_THRESHOLD:
                        full_action_values[i][:self.relocation_dim] = -1  # no relocation, must charge

                action_indexes = np.argmax(full_action_values, 1).tolist()

        return action_indexes

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

    def add_transition(self, state, action, next_state, reward, time_steps, valid_action):
        self.memory.push(state, action, next_state, reward, time_steps, valid_action)

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

        state_batch = torch.from_numpy(np.array(state_reps)).to(dtype=torch.float32, device=self.device)
        action_batch = torch.from_numpy(np.array(batch.action)).unsqueeze(1).to(dtype=torch.int64, device=self.device)
        reward_batch = torch.from_numpy(np.array(batch.reward)).unsqueeze(1).to(dtype=torch.float32, device=self.device)
        time_step_batch = torch.from_numpy(np.array(batch.time_steps)).unsqueeze(1).to(dtype=torch.float32, device=self.device)

        next_state_batch = torch.from_numpy(np.array(next_state_reps)).to(device=self.device, dtype=torch.float32)

        global_state_batch = torch.from_numpy(np.array(global_state_reps)).to(dtype=torch.float32, device=self.device)
        global_next_state_batch = torch.from_numpy(np.array(global_next_state_reps)).to(dtype=torch.float32,
                                                                                        device=self.device)

        q_state_action = self.get_main_Q(state_batch, global_state_batch).gather(1, action_batch.long())
        # add a mask
        all_q_ = self.get_target_Q(next_state_batch, global_next_state_batch)
        mask = self.get_action_mask(batch.next_state, batch.valid_action_num)  # action mask for next state
        all_q_[mask] = -9e10
        maxq = all_q_.max(1)[0].detach().unsqueeze(1)
        y = reward_batch + maxq*torch.pow(self.gamma,time_step_batch)
        loss = F.smooth_l1_loss(q_state_action, y)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.clipping_value)
        self.optimizer.step()
        self.lr_scheduler.step()

        self.record_list.append([self.train_step, round(float(loss),3), round(float(reward_batch.view(-1).mean()),3),self.optimizer.state_dict()['param_groups'][0]['lr'],batch.reward[0], batch.state[0][-1]])
        self.save_parameter(record_hist)
        print('Training step is {}; Learning rate is {}; Epsilon is {}:'.format(self.train_step,self.lr_scheduler.get_lr(),round(self.decayed_epsilon,4)))

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
            torch.save(checkpoint, 'logs/test/cnn_dqn_model/duel_dqn_%d_%d_%s.pkl' % (bool(self.no_charge),bool(self.global_matching),str(self.train_step)))
            # record training process (stacked before)
            for item in self.record_list:
                record_hist.writelines('{},{},{},{},{},{}\n'.format(item[0], item[1], item[2], item[3], item[4], item[5]))
            print('Training step: {}, replay buffer size:{}, epsilon: {}, learning rate: {}'.format(self.record_list[-1][0],len(self.memory), self.decayed_epsilon,self.lr_scheduler.get_lr()))
            self.record_list = []

    def get_action_mask(self, batch_state, batch_valid_action):
        mask = np.zeros([len(batch_state), self.output_dim])
        for i, state in enumerate(batch_state):
            mask[i][batch_valid_action[i]:self.relocation_dim] = 1
            # here the SOC in state is still continuous. the categorized one is in state reps.
            if state[-1] > HIGH_SOC_THRESHOLD:
                mask[i][self.relocation_dim:] = 1  # no charging, must relocate
            elif state[-1] < LOW_SOC_THRESHOLD:
                mask[i][:self.relocation_dim] = 1  # no relocation, must charge

        mask = torch.from_numpy(mask).to(dtype=torch.bool, device=self.device)
        return mask
