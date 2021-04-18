from config.setting import LEARNING_RATE, GAMMA, REPLAY_BUFFER_SIZE, BATCH_SIZE, RELOCATION_DIM, \
    INPUT_DIM, FINAL_EPSILON, HIGH_SOC_THRESHOLD, LOW_SOC_THRESHOLD, CLIPPING_VALUE, START_EPSILON, \
    EPSILON_DECAY_STEPS, H_AGENT_SAVE_PATH, SAVING_CYCLE, H_AGENT_REPLAY_BUFFER_SIZE, NUM_REACHABLE_HEX, \
    TERMINAL_STATE_SAVE_PATH
from .option_network import TargetOptionNetwork, OptionNetwork
from .replay_buffers import OptionReplayMemory
from collections import deque, defaultdict, OrderedDict
import os
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dqn_agent.dqn_feature_constructor import FeatureConstructor
from torch.optim.lr_scheduler import StepLR


class H_Agent:

    def __init__(self, hex_diffusion, xy_coords, num_option, isoption=True, islocal=True, ischarging=False):
        self.num_option =int(num_option)
        self.learning_rate = 2e-3
        self.gamma = GAMMA
        self.start_epsilon = START_EPSILON
        self.final_epsilon = FINAL_EPSILON
        self.epsilon_steps = EPSILON_DECAY_STEPS
        self.memory = OptionReplayMemory(H_AGENT_REPLAY_BUFFER_SIZE)  # 1e4
        self.batch_size = 256
        self.clipping_value = CLIPPING_VALUE
        self.input_dim = INPUT_DIM
        self.relocation_dim = RELOCATION_DIM
        self.option_dim = 3  # put 0 first, change it later. OPTION_DIM  # type: int
        self.premitive_action_dim = 1 + 6 + 5
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.path = H_AGENT_SAVE_PATH
        self.state_feature_constructor = FeatureConstructor()
        self.option_network = OptionNetwork(self.input_dim, self.premitive_action_dim)
        self.target_option_network = TargetOptionNetwork(self.input_dim, self.premitive_action_dim)
        self.optimizer = torch.optim.Adam(self.option_network.parameters(), lr=self.learning_rate)
        self.lr_scheduler = StepLR(optimizer=self.optimizer, step_size=1000,
                                   gamma=0.99)  # 1.79 e-6 at 0.5 million step.
        self.train_step = 0
        self.option_network.to(self.device)
        self.target_option_network.to(self.device)
        self.decayed_epsilon = self.start_epsilon

        self.hex_diffusion = hex_diffusion
        self.init_f_values_and_terminate_state()

        self.record_list = []
        self.global_state_dict = OrderedDict()
        self.time_interval = int(0)
        self.global_state_capacity = 10 * 1440  # we store 5 days' global states to fit replay buffer size.
        self.with_option = isoption
        self.with_charging = ischarging
        self.local_matching = islocal
        self.xy_coords = xy_coords
        self.option_queue = deque()
        self.init_func = None
        self.term_func = None
        self.init_dist = 0.05
        self.term_dist = 0.05
        self.upper_threshold = 1e5
        self.lower_threshold = 1e5

    def init_f_values_and_terminate_state(self):

        all_f_by_hour = np.zeros((24, NUM_REACHABLE_HEX))
        with open('logs/hex_p_value_%d.csv'%(self.num_option), 'r') as f:
            next(f)
            for lines in f:
                line = lines.strip().split(',')
                hr, hid, f_value = line  # hour, oridin, dest, trip_time/num of trip
                all_f_by_hour[int(hr)][int(hid)] = f_value
        self.all_f_values = all_f_by_hour
        self.f_means_per_hour = np.median(all_f_by_hour, axis=1)

        f_threshold_dict = defaultdict()
        middle_mask = defaultdict()
        term_percentile = 10

        for hr in range(24):
            f_middle_sorted = np.sort(self.all_f_values[hr])
            f_lower_threshold = np.percentile(f_middle_sorted ,(50 - term_percentile))
            f_higher_threshold = np.percentile(f_middle_sorted ,(50 + term_percentile))
            f_threshold_dict[hr] = [f_lower_threshold, f_higher_threshold]

            middle_mask[hr] = [np.sign(f_value - f_threshold_dict[hr][0]) * np.sign(f_threshold_dict[hr][1] - f_value)
                               for f_value in self.all_f_values[hr]]
        self.middle_mask = middle_mask  # 24 by 1347
        with open(TERMINAL_STATE_SAVE_PATH+'term_states_%d.csv'%(self.num_option),'w') as ts:
            for hr in range(24):
                for hex_id,term_flag in enumerate(middle_mask[hr]):
                    if term_flag == 1:
                        ts.writelines('{},{}\n'.format(hr,hex_id))
            print('finished record terminal state!!!')

    def get_f_values(self, state_batch):
        """
        get f values from pre-stored dict.
        :param state_batch:
        :return:
        """
        return [self.all_f_values[state[0] // (60 * 60) % 24][state[1]] for state in state_batch]

    def get_f_mean_by_hour(self, state_batch):
        return [self.f_means_per_hour[state[0] // (60 * 60) % 24] for state in state_batch]

    def is_initial(self, states):
        """
        :param states: includes time and hex_id
        :return:
        """
        return [1 if self.middle_mask[state[0] // (60 * 60) % 24][state[1]] == -1 else 0 for state in states]

    def is_middle_terminal(self, states):
        return [1 if self.middle_mask[state[0] // (60 * 60) % 24][state[1]] == 1 else 0 for state in states]

    def add_global_state_dict(self, global_state_list):
        for tick in global_state_list.keys():
            if tick not in self.global_state_dict.keys():
                self.global_state_dict[tick] = global_state_list[tick]
        if len(self.global_state_dict.keys()) > self.global_state_capacity:  # capacity limit for global states
            for _ in range(len(self.global_state_dict.keys()) - self.global_state_capacity):
                self.global_state_dict.popitem(last=False)

    # shuffle ==> minibatch ==> deep learning
    def add_transition(self, state, action, next_state, reward, terminate_flag, time_steps, valid_action):
        self.memory.push(state, action, next_state, reward, terminate_flag, time_steps, valid_action)

    def batch_sample(self):
        samples = self.memory.sample(self.batch_size)  # random.sample(self.memory, self.batch_size)
        return samples

    def get_main_Q(self, local_state, global_state):
        return self.option_network.forward(local_state, global_state)

    def get_target_Q(self, local_state, global_state):
        return self.target_option_network.forward(local_state, global_state)

    def soft_target_update(self, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(self.target_option_network.parameters(), self.option_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def train(self, hist):
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

        hex_diffusion = [np.tile(self.hex_diffusion[state[1]], (1, 1, 1)) for state in batch.state]
        hex_diffusion_ = [np.tile(self.hex_diffusion[state_[1]], (1, 1, 1)) for state_ in batch.next_state]

        state_batch = torch.from_numpy(np.array(state_reps)).to(dtype=torch.float32, device=self.device)

        action_batch = torch.from_numpy(np.array(batch.action)).unsqueeze(1).to(dtype=torch.int64, device=self.device) - self.option_dim

        time_step_batch = torch.from_numpy(np.array(batch.time_steps)).unsqueeze(1).to(dtype=torch.float32, device=self.device)
        next_state_batch = torch.from_numpy(np.array(next_state_reps)).to(device=self.device, dtype=torch.float32)
        global_state_batch = torch.from_numpy(
            np.concatenate([np.array(global_state_reps), np.array(hex_diffusion)], axis=1)).to(dtype=torch.float32,
                                                                                               device=self.device)
        global_next_state_batch = torch.from_numpy(
            np.concatenate([np.array(global_next_state_reps), np.array(hex_diffusion_)], axis=1)).to(
            dtype=torch.float32, device=self.device)

        f_s = torch.from_numpy(np.array(self.get_f_values(batch.state))).to(dtype=torch.float32, device=self.device)
        f_s_ = torch.from_numpy(np.array(self.get_f_values(batch.next_state))).to(dtype=torch.float32,
                                                                                  device=self.device)
        f_median = torch.from_numpy(np.array(self.get_f_mean_by_hour(batch.state))).to(dtype=torch.float32,
                                                                                       device=self.device)
        matched_flag = torch.from_numpy(np.array(batch.terminate_flag)).to(dtype=torch.float32, device=self.device)
        middle_terminal_flag = torch.from_numpy(np.array((self.is_middle_terminal(batch.state)))).to(
            dtype=torch.float32, device=self.device)

        middle_next_terminal_flag = torch.from_numpy(np.array((self.is_middle_terminal(batch.next_state)))).to(
            dtype=torch.float32, device=self.device)
        # demand_s = [self.global_state_dict[state[0]/60][0,self.xy_coords[state[1]]] for state in batch.state]
        # demand_s_ = [self.global_state_dict[state_[0]/60][0,self.xy_coords[state_[1]]] for state_ in batch.next_state]

        # demand_s_batch = torch.from_numpy(np.array(demand_s)).to(dtype=torch.float32, device=self.device)
        # next_demand_s_batch = torch.from_numpy(np.array(demand_s_)).to(dtype=torch.float32, device=self.device)

        q_state_action = self.get_main_Q(state_batch, global_state_batch).gather(1, action_batch.long())
        # add a mask
        all_q_ = self.get_target_Q(next_state_batch, global_next_state_batch)
        mask = self.get_action_mask(batch.next_state, batch.valid_action_num)  # action mask for next state
        all_q_[mask] = -9e10
        maxq = all_q_.max(1)[0].detach().unsqueeze(1)
        pseudo_reward = np.sign(f_s) * (f_s - f_s_) + np.abs(f_s - f_median) * matched_flag
        y = (1 - middle_terminal_flag) * (pseudo_reward + (1 - middle_next_terminal_flag) * maxq*torch.pow(self.gamma,time_step_batch))
        loss = F.smooth_l1_loss(q_state_action,y)

        print('Max Q={}, Max target Q={}, Loss = {}, Gamma={}'.format(torch.max(q_state_action), torch.max(maxq),loss, self.gamma))

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.option_network.parameters(), self.clipping_value)
        self.optimizer.step()
        self.record_list.append([self.train_step, round(float(loss), 4), torch.max(maxq)])
        self.lr_scheduler.step()
        self.save_parameter(hist)

    def save_parameter(self, record_hist):
        # torch.save(self.q_network.state_dict(), self.dqn_path)
        if self.train_step % SAVING_CYCLE == 0:
            checkpoint = {
                "net": self.option_network.state_dict()
                # "step": self.train_step,
                # "lr_scheduler": self.lr_scheduler.state_dict()
            }
            if not os.path.isdir(self.path):
                os.mkdir(self.path)
            print('h_agent is saving at {}'.format(self.path + 'ht_network_option_%d_%d_%d_%d_%d.pkl' % (self.num_option,
            bool(self.with_option), bool(self.with_charging), bool(self.local_matching), self.train_step)))

            torch.save(checkpoint, self.path + 'ht_network_option_%d_%d_%d_%d_%d.pkl' % (self.num_option,
            bool(self.with_option), bool(self.with_charging), bool(self.local_matching), self.train_step))
            for item in self.record_list:
                record_hist.writelines('{},{},{}\n'.format(item[0], item[1], item[2]))
            self.record_list = []

    def get_action_mask(self, batch_state, batch_valid_action):
        mask = np.zeros([len(batch_state), self.premitive_action_dim])
        for i, state in enumerate(batch_state):
            mask[i][batch_valid_action[i]:self.relocation_dim] = 1
            # here the SOC in state is still continuous. the categorized one is in state reps.
            if state[-1] > HIGH_SOC_THRESHOLD:
                mask[i][self.relocation_dim:] = 1  # no charging, must relocate
            elif state[-1] < LOW_SOC_THRESHOLD:
                mask[i][:self.relocation_dim] = 1  # no relocation, must charge

        mask = torch.from_numpy(mask).to(dtype=torch.bool, device=self.device)
        return mask

    # def get_option_mask(self,states):
    #     """
    #     append masks to the options that think the states as terminate.
    #     if no option is generated yet, we keep the option masked as 1. (we initial the mask by np.ones)
    #     :param states:
    #     :return:
    #     """
    #     termiante_option_mask = np.ones((len(states),self.option_dim))
    #     for hr,op in enumerate(self.option_queue):
    #         for state in states:
    #             if self.is_initial(state):
    #                 termiante_option_mask[:,] = 0 # the j-th element of the i-th column is masked
    #
    #     termiante_option_mask = torch.from_numpy(termiante_option_mask).to(dtype=torch.bool, device=self.device)
    #     return termiante_option_mask
