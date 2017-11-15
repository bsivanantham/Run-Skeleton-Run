import tensorflow as tf
import baselines.baselines_common.tf_util as U
from baselines.baselines_common.mpi_running_mean_std import RunningMeanStd
from baselines.baselines_common.distributions import make_pdtype, DiagGaussianPdType, BernoulliPdType


def mlp_block(x, name, num_hid_layers, hid_size, activation_fn=tf.nn.tanh):
    with tf.variable_scope(name_or_scope=name):
        for i in range(num_hid_layers):
            x = U.dense(
                x, hid_size,
                name="fc%i" % (i + 1), weight_init=U.normc_initializer(1.0))
            x = activation_fn(x)
        return x


def feature_net(x, name, num_hid_layers, hid_size, activation_fn=tf.nn.tanh):
    with tf.variable_scope(name_or_scope=name):
        x = mlp_block(
            x, name="mlp",
            hid_size=hid_size, num_hid_layers=num_hid_layers, activation_fn=activation_fn)
        return x


class Actor(object):
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True, noise_type=None):
        if noise_type == "gaussian":
            self.pdtype = pdtype = DiagGaussianPdType(ac_space.shape[0])
        else:
            self.pdtype = pdtype = make_pdtype(ac_space)

        ob = U.get_placeholder(
            name="ob", dtype=tf.float32,
            shape=[None] + list(ob_space.shape))

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)
        obz = (ob - self.ob_rms.mean) / self.ob_rms.std
        obz = tf.clip_by_value(obz, -5.0, 5.0)

        # critic net (value network)
        last_out = feature_net(
            obz, name="vf",
            num_hid_layers=num_hid_layers, hid_size=hid_size,
            activation_fn=tf.nn.tanh)
        self.vpred = U.dense(
            last_out, 1,
            name="vf_final", weight_init=U.normc_initializer(1.0))[:, 0]

        # actor net (policy network)
        last_out = feature_net(
            obz, name="pol",
            num_hid_layers=num_hid_layers, hid_size=hid_size,
            activation_fn=tf.nn.tanh)

        if gaussian_fixed_var and isinstance(self.pdtype, DiagGaussianPdType):
            mean = U.dense(
                last_out, pdtype.param_shape()[0] // 2,
                name="pol_final", weight_init=U.normc_initializer(0.01))
            logstd = tf.get_variable(
                name="logstd", shape=[1, pdtype.param_shape()[0] // 2],
                initializer=tf.zeros_initializer())
            pdparam = U.concatenate([mean, mean * 0.0 + logstd], axis=1)
        else:
            pdparam = U.dense(
                last_out, pdtype.param_shape()[0],
                name="pol_final", weight_init=U.normc_initializer(0.01))

        # pd - probability distribution
        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 = self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []
