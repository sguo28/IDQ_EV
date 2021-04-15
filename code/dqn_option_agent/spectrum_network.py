


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
