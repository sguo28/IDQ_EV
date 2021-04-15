import tensorflow as tf
import numpy as np
import random
import tflearn
import os

from simple_rl.mdp.StateClass import State
from simple_rl.agents.func_approx.ExperienceBuffer import ExperienceBuffer
from simple_rl.agents.func_approx.Features import Fourier, Monte, Subset, AgentPos
from simple_rl.agents import DQNAgent, DDPGAgent, LinearQAgent, RandomAgent, RandomContAgent, DiaynAgent


class CoveringOption():
    """
    Wrapper to describe options
    """
    def __init__(self, sess=None, experience_buffer=None, option_b_size=None, sp_training_steps=100, obs_dim=None,
                 obs_bound=None, action_dim=None, action_bound=None, num_actions=None, low_method='linear',
                 f_func='fourier', n_units=16, init_all=True, reversed_dir=False, init_around_goal=False, init_dist=0.9,
                 term_dist=0.1, restore=None, name=None):
        self.init_dist = init_dist
        self.term_dist = term_dist

        if sess is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True  # TODO: conv dumps error without this
            self.sess = tf.Session(config=config)
        else:
            self.sess = sess
        self.option_b_size = option_b_size
        self.sp_training_steps = sp_training_steps

        self.low_method = low_method
        self.f_func = f_func
        self.n_units = n_units
        self.init_all = init_all
        self.reversed_dir = reversed_dir
        self.name = name  # self.name + "_inst" + str(self.curr_instances) + "_spc" + op_name

        self.obs_dim = obs_dim
        self.obs_bound = obs_bound
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.num_actions = num_actions

        self.init_around_goal = init_around_goal

        self.init_fn = None
        self.term_fn = None

        if restore is None:
            self.setup_networks()
        if experience_buffer is not None:
            self.train_f_function(experience_buffer)

    def setup_networks(self):
        if self.f_func == 'nn':
            self.f_function = SpectrumNetwork(self.sess, obs_dim=self.obs_dim, n_units=self.n_units, name=self.name)
        elif self.f_func == 'nnf':
            features = Monte()
            self.f_function = SpectrumNetwork(self.sess, obs_dim=self.obs_dim, feature=features, n_units=self.n_units,
                                              name=self.name)
        elif self.f_func == 'nns':
            features = Subset(state_dim=self.obs_dim, feature_indices=[0, 1])  # TODO: parameterize
            self.f_function = SpectrumNetwork(self.sess, obs_dim=self.obs_dim, feature=features, n_units=self.n_units,
                                              name=self.name)
        elif self.f_func == 'nnc':
            # Convolutions
            self.f_function = SpectrumNetwork(self.sess, obs_dim=self.obs_dim, n_units=self.n_units, conv=True,
                                              name=self.name)
        elif self.f_func == 'rand':
            self.f_function = None
        else:
            print('f_func =', self.f_func)
            # print('len(ffnc)=', len(self.f_func))
            assert (False)

        if self.f_function is not None:
            self.f_function.initialize()
        self.agent = DQNAgent(self.sess, obs_dim=self.obs_dim, num_actions=self.num_actions, gamma=0.99,name=self.name)

        self.agent.reset()

    def is_initiation(self, state):
        assert (isinstance(state, State))
        if self.init_fn is None:
            return True
        elif self.init_all:
            # The option can be initialized anywhere except its termination state
            return not self.is_terminal(state)
        else:
            # TODO: We want to make it to "if > min f + epsilon"
            # print('fvalue = ', self.f_function(np.reshape(state, (1, state.shape[0]))))
            # state_d = state.data.flatten()
            # f_value = self.f_function(np.reshape(state_d, (1, state_d.shape[0]))).flatten()[0]
            f_value = self.f_function(state)[0][0]
            # print('is_init: val=', f_value)
            return self.init_fn(f_value)

    def is_terminal(self, state):
        assert (isinstance(state, State))

        if self.term_fn is None:
            return True
        else:
            f_value = self.f_function(state)[0][0]

            bound = self.lower_th

            # print('f_value, bound = ', f_value, bound)
            # if f_value < bound:
            #     print('f<b so terminates')
            # else:
            #     print('f>b so continue')
            # state_d = state.data.flatten()
            # f_value = self.f_function(np.reshape(state_d, (1, state_d.shape[0]))).flatten()[0]
            # print('is_term: val=', f_value)
            # return f_value < 0.03
            return self.term_fn(f_value)

    def act(self, state):
        return self.agent.act(state, 0, learning=False)

    def train_f_function(self, experience_buffer):
        assert (self.option_b_size is not None)

        self.f_function.initialize()

        for _ in range(self.sp_training_steps):
            s, a, r, s2, t = experience_buffer.sample(self.option_b_size)

            # Even if we switch the order of s and s2, we get the same eigenfunction.
            # next_f_value = self.f_function(s)
            # self.f_function.train(s2, next_f_value)

            next_f_value = self.f_function(s2)
            self.f_function.train(s, next_f_value)

        self.upper_th, self.lower_th = self.sample_f_val(experience_buffer, self.init_dist, self.term_dist)

        # print('init_th, term_th = ', init_th, term_th)
        if self.reversed_dir:
            self.term_fn = lambda x: x > self.upper_th
            if self.init_around_goal:
                self.init_fn = lambda x: x > self.lower_th
            else:
                self.init_fn = lambda x: x < self.lower_th
        else:
            self.term_fn = lambda x: x < self.lower_th
            if self.init_around_goal:
                self.init_fn = lambda x: x < self.lower_th
            else:
                self.init_fn = lambda x: x > self.upper_th

    def sample_f_val(self, experience_buffer, upper, lower):
        buf_size = experience_buffer.size()

        # n_samples = min(buf_size, 1024)
        n_samples = buf_size

        s = [experience_buffer.buffer[i][0] for i in range(experience_buffer.size())]

        # s, _, _, _, _ = experience_buffer.sample(n_samples)
        f_values = self.f_function(s)
        if type(f_values) is list:
            f_values = np.asarray(f_values)
        f_values = f_values.flatten()

        f_srt = np.sort(f_values)

        print('f_srt=', f_srt)

        init_th = f_srt[int(n_samples * upper)]
        term_th = f_srt[int(n_samples * lower)]

        print('init_th, term_th=', init_th, term_th)

        assert (init_th > term_th)
        return init_th, term_th

    def train(self, experience_buffer, batch_size):
        # Training the policy of the agent
        s, a, r, s2, t = experience_buffer.sample(batch_size)

        if self.f_function is None:
            self.agent.train_batch(s, a, r, s2, t, batch_size=batch_size)
        else:
            r_shaped = []

            for i in range(batch_size):
                # Reward is given if it minimizes the f-value
                # r_s = self.f_function(np.reshape(s[i].data, (1, s[i].data.shape[0]))) - self.f_function(np.reshape(s2[i].data, (1, s2[i].data.shape[0]))) + r[i]

                if self.reversed_dir:
                    r_s = self.f_function(s2[i]) - self.f_function(s[i]) + r[i]
                else:
                    r_s = self.f_function(s[i]) - self.f_function(s2[i]) + r[i]
                r_shaped.append(r_s)
                # print('reward=',  r[i] ,' shaped-reward=', r_s)
            self.agent.train_batch(s, a, r_shaped, s2, t, batch_size=batch_size)

    def restore(self, directory):
        # Restore
        # 1. f function
        # 2. init threshold, term threshold
        # 3. agent
        with open(directory + '/meta', 'r') as f:
            self.f_func = f.readline().split(' ')[1].strip()
            self.upper_th = float(f.readline().split(' ')[1].strip())
            self.lower_th = float(f.readline().split(' ')[1].strip())
            self.low_method = f.readline().split(' ')[1].strip()
            self.init_all = f.readline().split(' ')[1].strip() == 'True'
            self.reversed_dir = f.readline().split(' ')[1].strip() == 'True'

        if self.reversed_dir:
            print('restored reversed direction')
            self.init_fn = lambda x: x < self.lower_th
            self.term_fn = lambda x: x > self.upper_th
        else:
            self.init_fn = lambda x: x > self.upper_th
            self.term_fn = lambda x: x < self.lower_th

            # print('f_func=', self.f_func)

        self.setup_networks()

        self.f_function.restore(directory)
        # self.agent.restore(directory, rev=self.reversed_dir)
        self.agent.restore(directory)

        # print(self.f_function)

    def save(self, directory, rev=False):
        if not os.path.exists(directory):
            os.mkdir(directory)
        with open(directory + '/meta', 'w') as f:
            f.write('f_func: ' + self.f_func + '\n')
            f.write('upper_th: ' + str(self.upper_th) + '\n')
            f.write('lower_th: ' + str(self.lower_th) + '\n')
            f.write('low_method: ' + self.low_method + '\n')
            f.write('init_all: ' + str(self.init_all) + '\n')
            f.write('reversed_dir: ' + str(self.reversed_dir) + '\n')

        # Save f-function
        self.f_function.save(directory)
        # Save agent policy
        if rev:
            self.agent.save(directory, name=self.name + 'rev')
        else:
            self.agent.save(directory)

class SpectrumNetwork():
    NAME = "spectrum"
    def __init__(self, sess, obs_dim=None, learning_rate=0.001, training_steps=100, batch_size=32, n_units=16, beta=2.0,
                 delta=0.1, feature=None, conv=False, name=NAME):
        # Beta  : Lagrange multiplier. Higher beta would make the vector more orthogonal.
        # delta : Orthogonality parameter.
        self.sess = sess
        self.learning_rate = learning_rate
        self.obs_dim = obs_dim

        self.n_units = n_units

        # self.beta = 1000000.0
        self.beta = beta
        self.delta = 0.05
        # self.delta = delta

        self.feature = feature

        self.conv = conv

        self.name = name

        self.obs, self.f_value = self.network(scope=name + "_eval")

        self.next_f_value = tf.placeholder(tf.float32, [None, 1], name=name + "_next_f")

        # TODO: Is this what we are looking for?
        self.loss = tflearn.mean_square(self.f_value, self.next_f_value) + \
                    self.beta * tf.reduce_mean(tf.multiply(self.f_value - self.delta, self.next_f_value - self.delta)) + \
                    self.beta * tf.reduce_mean(self.f_value * self.f_value * self.next_f_value * self.next_f_value) + \
                    self.beta * (self.f_value - self.next_f_value)  # This is to let f(s) <= f(s').

        # with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
        self.optimizer = tf.train.AdamOptimizer(learning_rate)

        self.optimize = self.optimizer.minimize(self.loss)

        self.network_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name + "_eval")
        self.initializer = tf.initializers.variables(self.network_params + self.optimizer.variables())

        # print('network param names for ', self.name)
        # for n in self.network_params:
        #     print(n.name)

        self.saver = tf.train.Saver(self.network_params)

    def network(self, scope):
        # TODO: What is the best NN?
        if self.feature is None:
            indim = self.obs_dim
        else:
            indim = self.feature.num_features()

        obs = tf.placeholder(tf.float32, [None, indim], name=self.name + "_obs")

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            if self.conv:
                reshaped_obs = tf.reshape(obs, [-1, 105, 80, 3])
                net = tflearn.conv_2d(reshaped_obs, 32, 8, strides=4, activation='relu')
                net = tflearn.conv_2d(net, 64, 4, strides=2, activation='relu')
                out = tflearn.fully_connected(net, 1,
                                              weights_init=tflearn.initializations.uniform(minval=-0.003, maxval=0.003))
            else:
                net = tflearn.fully_connected(obs, self.n_units, name='d1',
                                              weights_init=tflearn.initializations.truncated_normal(
                                                  stddev=1.0 / float(indim)))
                net = tflearn.fully_connected(net, self.n_units, name='d2',
                                              weights_init=tflearn.initializations.truncated_normal(
                                                  stddev=1.0 / float(self.n_units)))
                net = tflearn.fully_connected(net, self.n_units, name='d3',
                                              weights_init=tflearn.initializations.truncated_normal(
                                                  stddev=1.0 / float(self.n_units)))
                # net = tflearn.layers.normalization.batch_normalization(net)
                # net = tf.contrib.layers.batch_norm(net)
                net = tflearn.activations.relu(net)

                w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
                out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return obs, out

    def train(self, obs1, next_f_value):
        # obs1 is
        obs = []
        for state in obs1:
            if self.feature is None:
                o = state.data.flatten()
            else:
                o = self.feature.feature(state, 0)
            obs.append(o)

        # print('next_f_value=', next_f_value)
        # print('type(obs)=', type(obs))
        # print('type(next_f_value)=', type(next_f_value))
        self.sess.run(self.optimize, feed_dict={
            self.obs: obs,
            self.next_f_value: next_f_value
        })

    def initialize(self):
        self.sess.run(self.initializer, feed_dict={})

    def f_ret(self, state):
        assert (isinstance(state, State))

        if self.feature is None:
            obs = np.reshape(state.data, (1, state.data.shape[0]))
        else:
            obs = self.feature.feature(state, 0)
            obs = np.asarray(obs)
            obs = np.reshape(obs, (1, self.feature.num_features()))

        return self.sess.run(self.f_value, feed_dict={
            self.obs: obs
        })

    def f_from_features(self, features):
        assert (isinstance(features, np.ndarray))
        return self.sess.run(self.f_value, feed_dict={
            self.obs: features
        })

    def __call__(self, obs):
        if type(obs) is list:
            ret = []
            for o in obs:
                ret.append(self.f_ret(o)[0])
            return ret
        return self.f_ret(obs)

    def restore(self, directory, name='spectrum_nn'):
        self.saver.restore(self.sess, directory + '/' + name)

    def save(self, directory, name='spectrum_nn'):
        self.saver.save(self.sess, directory + '/' + name)

    # if self.low_method == 'linear':
    #     # low_bound = np.asarray([0.0, 0.0, -2.0, -2.0])
    #     # up_bound = np.asarray([1.0, 1.0, 2.0, 2.0])
    #     features = Fourier(state_dim=self.obs_dim, bound=self.obs_bound, order=3)
    #     self.agent = LinearQAgent(actions=range(self.num_actions), feature=features, name=self.name)
    # elif self.low_method == 'ddpg':
    #     # TODO: Using on-policy method is not good for options? is DDPG off-policy?
    #     self.agent = DDPGAgent(self.sess, obs_dim=self.obs_dim, action_dim=self.action_dim,
    #                            action_bound=self.action_bound, name=self.name)
    # elif self.low_method == 'dqn':
    #     self.agent = DQNAgent(self.sess, obs_dim=self.obs_dim, num_actions=self.num_actions, gamma=0.99,
    #                           name=self.name)
    # elif self.low_method == 'rand':
    #     if self.num_actions is None:
    #         self.agent = RandomContAgent(action_dim=self.action_dim, action_bound=self.action_bound, name=self.name)
    #     else:
    #         self.agent = RandomAgent(range(self.num_actions), name=self.name)
    # else:
    #     print('low_method=', self.low_method)
    #     assert (False)
