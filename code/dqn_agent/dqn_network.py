import torch.nn as nn

import torch.nn.functional as F

class DQN_network(nn.Module):
    '''
    Full connected layer with Relu
    '''

    def __init__(self, input_features,output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_features,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,output_dim)
    
    def forward(self,input_state):
       fc1_out = F.relu(self.fc1(input_state))
       fc2_out = F.relu(self.fc2(fc1_out))
       q_value = self.fc3(fc2_out)

       return q_value