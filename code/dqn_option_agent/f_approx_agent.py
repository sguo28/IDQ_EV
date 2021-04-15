import os
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from config.setting import LEARNING_RATE, GAMMA, SAVING_CYCLE, EPSILON_DECAY_STEPS, F_AGENT_SAVE_PATH,\
    TRAJECTORY_BATCH_SIZE, TRAJECTORY_BUFFER_SIZE
from .f_approx_network import F_Network
from torch.optim.lr_scheduler import StepLR
from .replay_buffers import TrajReplayMemory
from collections import deque


class F_Agent:
    """
    F agent is to train the approximator of second eigenvector by hour.
    """
    def __init__(self,hex_diffusion,  isoption=False,islocal=True,ischarging=True):
        self.learning_rate = 1e-4 # 5e-4
        self.epsilon_f_steps = EPSILON_DECAY_STEPS
        self.traj_memory = TrajReplayMemory(TRAJECTORY_BUFFER_SIZE) # [TrajReplayMemory(REPLAY_BUFFER_SIZE) for _ in range(24)] # 24 hours
        self.f_batch_size = TRAJECTORY_BATCH_SIZE
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.path = F_AGENT_SAVE_PATH
        # init option network
        self.f_network = F_Network()  # output a policy
        self.f_optimizer = torch.optim.Adam(self.f_network.parameters(), lr=self.learning_rate)
        self.f_train_step = 0
        self.f_network.to(self.device)

        self.record_list = []
        self.global_state_dict = defaultdict()
        self.with_option = isoption
        self.with_charging = ischarging
        self.local_matching = islocal
        self.hex_diffusion = hex_diffusion
        self.init_func = None
        self.term_func = None
        self.init_dist = 0.05
        self.term_dist = 0.05
        self.upper_threshold = 1e5
        self.lower_threshold = 1e5
        self.record_list = []

    def get_f_values(self, hex_ids):
        hex_diffusions = [np.tile(self.hex_diffusion[int(hex_id)], (1, 1, 1)) for hex_id in hex_ids]  # state[1] is hex_id
        return self.f_network.forward(torch.from_numpy(np.array(hex_diffusions)).to(dtype=torch.float32, device=self.device))

    def add_hex_pair(self,current_hex,next_hex):
        self.traj_memory.push(current_hex,next_hex)

    def train_f_function(self,hr,hist_file):
        if len(self.traj_memory) < self.f_batch_size:
            print('batches in replay buffer for Hour {} is {} at step {}'.format(hr,len(self.traj_memory),self.f_train_step))
            return
        self.f_train_step += 1
        transitions = self.traj_memory.sample(self.f_batch_size)
        traj_batch = self.traj_memory.Transition(*zip(*transitions))

        f_values = self.get_f_values(traj_batch.current_hex)
        # add a mask
        f_values_ = self.get_f_values(traj_batch.next_hex)
        eta = 2.0 # lagrangian multiplier, it was assumed as 1.0 in all scenarios, so we also try 1.0.
        loss = 0.5 * F.mse_loss(f_values_ , f_values) + eta*((f_values_.pow(2) - 1)*(f_values.pow(2)-1) + f_values_.pow(2)*f_values.pow(2)).mean()  # + (f_values-f_values_).mean())
        self.f_optimizer.zero_grad()
        loss.backward()
        self.f_optimizer.step()
        self.record_list.append([self.f_train_step, round(float(loss), 4)])
        self.save_parameter(hr,hist_file)
        print("Step:{}, Loss:{}".format(self.f_train_step,loss))

    def save_parameter(self, hr, hist_file):
        # torch.save(self.q_network.state_dict(), self.dqn_path)
        if self.f_train_step % SAVING_CYCLE == 0:
            checkpoint = {
                "net": self.f_network.state_dict()
            }
            if not os.path.isdir(self.path):
                os.mkdir(self.path)
            print('f_approx is saved at {}'.format(self.path+'f_network_%d.pkl' % (hr)))
            torch.save(checkpoint, self.path+'f_network_%d.pkl' % (hr))
            # (bool(self.with_option),bool(self.with_charging),bool(self.local_matching)))
        for item in self.record_list:
            hist_file.writelines('{},{},{}\n'.format(item[0],hr, item[1]))
        self.record_list = []
