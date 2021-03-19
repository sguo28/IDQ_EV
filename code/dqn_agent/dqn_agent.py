import random
import numpy as np
import torch
import torch.nn.functional as F
from .dqn_network import DQN_network
from collections import namedtuple
from config import hex_setting

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward','flag'))
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DeepQNetworkAgent():
    def __init__(self):
        self.learning_rate = hex_setting.learning_rate
        self.gamma = hex_setting.gamma
        self.memory = ReplayMemory(int(hex_setting.replay_buffer_size)) # deque(maxlen=hex_setting.replay_buffer_size)
        self.batch_size = hex_setting.batch_size
        # hex_setting.action_space #[i for i in range(1+6+5)] # stay still:1 , move to adjacent hexs:6, nearest 5 charging stations:5
        self.input_dim = hex_setting.input_dim
        self.relocation_dim = hex_setting.relocation_dim 
        self.charging_dim = hex_setting.charging_dim
        self.output_dim = hex_setting.output_dim # len(self.action_space)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.dqn_path = dqn_path
        self.q_network = DQN_network(self.input_dim,self.output_dim) # function
        self.q_network.to(self.device)
        self.target_q_network = DQN_network(self.input_dim,self.output_dim)
        self.target_q_network.to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(),lr=self.learning_rate)
        # self.reachable_hex = None
        self.train_step = 0
        self.mile_of_range = 220
    def get_q_network(self):
        # if self.dqn_path:
        #     return torch.load(self.dqn_path)
        # else:
        # return DQN_network(self.input_dim,self.output_dim)    
        return self.q_network

    def get_actions(self,states,num_valid_relos):
        '''
        todo: update esilon value later, for now keep it as large values
        :param state: tuple of (tick, veh_id, hex_id,off-duty flag, SOC) and SOC is 0 - 100%
        :param num_valid_relo:
        :states: 
        :return:
        '''
        with torch.no_grad():
            epsilon = hex_setting.epsilon
            if random.random()>1-epsilon:
            # print(state, state[-1])
                full_action_values = self.q_network.forward(torch.from_numpy(np.array(states)).to(dtype=torch.float32,device=self.device))
                mask = np.zeros([self.output_dim,len(states)])
                for i in range(len(states)):
                    mask[i][num_valid_relos[i]:self.relocation_dim]=[1 for _ in range(len(self.relocation_dim-num_valid_relos[i]))]
                
                mask = torch.from_numpy(mask).to(dtype=torch.bool,device=self.device)
                full_action_values[mask] = -9e10
                # full_action_values[num_valid_relo:self.relocation_dim]=-9e10  #infeasible solutions, make them to very small values
                # if states[:,-1] >=0.50:
                #     # print(state[-2],"larger")
                #     action_index=torch.argmax(full_action_values)[:num_valid_relo].item()
                # elif states[:,-1] >=0:
                #     action_index=torch.argmax(full_action_values).item()
                # else:
                #     action_indexes = []
                action_indexes=[torch.argmax(full_action_values)[:num_valid_relo].item() \
                    if state[-1]>0.50 else torch.argmax(full_action_values).item() \
                    if state[-1]>=0 else [] for state,num_valid_relo in zip(states,num_valid_relos)]
            else:
                action_indexes=[random.randrange(num_valid_relo) \
                    if state[-1]>0.50 else random.choice(list(range(num_valid_relo))+list(range(self.relocation_dim,self.output_dim))) \
                    if state[-1]>=0 else [] for state,num_valid_relo in zip(states,num_valid_relos)]
                # if states[-1] >= 0.50:
                #     action_index = random.randrange(self.relocation_dim)
                # elif states[-1] >= 0:
                #     action_index = random.randrange(self.output_dim)
                # else:
                    # action_index = []

                # same as above
                

        return action_indexes

    def add_transition(self, state,action,next_state,reward,flag): # flag: off-duty
        self.memory.push(state,action,next_state,reward,flag)

    def batch_sample(self):
        samples = random.sample(self.memory, self.batch_size)
        state, action, next_state, reward, flags = zip(*samples)
        return state, action, next_state, reward, flags

    def get_main_Q(self,state):
        return self.q_network.forward(state)

    def get_target_Q(self,state):
        return self.target_q_network.forward(state)

    def copy_parameter(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def train(self):
        self.train_step+=1
        if self.memory.__len__()< self.batch_size:
            return
        self.optimizer.zero_grad()
        transitions = self.memory.sample(self.batch_size)
        batch = self.memory.Transition(*zip(*transitions))
        # state_batch = torch.cat(list(batch.state))
        # action_batch = torch.stack(batch.action)
        # reward_batch = torch.stack(batch.reward)
        state_batch = torch.tensor(batch.state,device = self.device, dtype = torch.float32)
        action_batch = torch.tensor(batch.action,device = self.device, dtype = torch.int64).view(-1,1)
        reward_batch = torch.tensor(batch.reward,device = self.device, dtype = torch.float32)
        
        non_terminal_indices = torch.tensor(
        tuple(map(lambda s: s is not 1, batch.flag)),
        device=self.device,
        dtype=torch.bool,
        )
        on_duty_next_states = torch.tensor(batch.next_state,device = self.device, dtype = torch.float32)
        
        q_state_action = self.get_main_Q(state_batch).gather(1,action_batch)
        # print(q_state_action)
        maxq = self.get_main_Q(on_duty_next_states).max(1)[0].detach()
        additional_qs = torch.zeros(self.batch_size, device=self.device)
        additional_qs[non_terminal_indices] = maxq
        # print(q_state_action,reward_batch.unsqueeze(1),additional_qs.unsqueeze(1))
        y = reward_batch.unsqueeze(1) + self.gamma *additional_qs.unsqueeze(1)
        loss = F.mse_loss(q_state_action,y)
        loss.backward()
        self.optimizer.step()
        # f.writelines('{},{}\n'.format(self.train_step,reward))
        return loss






