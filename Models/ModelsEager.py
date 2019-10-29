import tensorflow as tf
from tensorflow import keras
import numpy as np
from Losses.Losses import Losses
import tensorflow.contrib.slim as slim


class Linear:

    # from https://github.com/awjuliani/DeepRL-Agents

    def __init__(self, input_shape, h_size, n_actions, scope_var, device):

        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(8,input_shape=[1],kernel_initializer='normal',activation='linear'))
        self.model.add(keras.layers.Dense(4, kernel_initializer='normal'))
        self.model.compile(loss='mean_squared_error', optimizer='adam')


    def prediction(self, sess, s):

        s=np.array(s)

        a = self.model.predict(s)

        return np.argmax(a,1)

    def Qprediction(self, sess, s):

        return self.model.predict(s)

    def train(self, sess, s, targetQ, imp_w):

        self.model.fit(s, targetQ, epochs=1, verbose=0)

        return [1,2]


class DenseModel(keras.Model):

    def __init__(self, h_size, n_actions, input_shape):
        super(DenseModel, self).__init__(name="DenseModel")
        self.dense1 = keras.layers.Dense(h_size, activation='elu', kernel_initializer='he_normal')
        self.dense2 = keras.layers.Dense(h_size, activation='elu', kernel_initializer='he_normal')
        self.Qout = keras.layers.Dense(n_actions, activation='linear')

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.Qout(x)
        return x


class DenseAdvantageModel(keras.Model):

    def __init__(self, h_size, n_actions, input_shape):
        super(DenseAdvantageModel, self).__init__(name="DenseAdvantageModel")
        self.dense1 = keras.layers.Dense(h_size, activation='elu', kernel_initializer='he_normal')
        #self.dense2 = keras.layers.Dense(h_size, activation='elu', kernel_initializer='he_normal')

        # Have the network estimate the Advantage function as an intermediate layer
        self.advt_f = keras.layers.Dense(h_size/2, activation='elu', kernel_initializer='he_normal')
        self.advt = keras.layers.Dense(n_actions, activation='linear')

        self.val_f = keras.layers.Dense(h_size/2, activation='elu', kernel_initializer='he_normal')
        self.value = keras.layers.Dense(1, activation='linear')

        #combine the two streams
        self.A = keras.layers.Lambda(lambda advt: advt - tf.reduce_mean(advt, axis=-1, keep_dims=True))
        self.V = keras.layers.Lambda(lambda value: tf.tile(value, [1, n_actions]))
        self.Qout = keras.layers.Add()

    def call(self, x):
        x = self.dense1(x)
        #x = self.dense2(x)
        A = self.advt_f(x)
        A = self.advt(A)
        V = self.val_f(x)
        V = self.value(V)
        A = self.A(A)
        V = self.V(V)
        x = self.Qout([V, A])
        return x


class REINFORCE_DenseModel(keras.Model):
    def __init__(self, h_size, n_actions, input_shape):
        super(REINFORCE_DenseModel, self).__init__(name="REINFORCE_DenseModel")
        self.dense1 = keras.layers.Dense(h_size, activation='elu', kernel_initializer='he_normal')
        self.dense2 = keras.layers.Dense(h_size, activation='elu', kernel_initializer='he_normal')
        self.out = keras.layers.Dense(n_actions, activation=keras.activations.softmax)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.out(x)
        return x


class REINFORCE_ConvModel(keras.Model):
    def __init__(self, h_size, n_actions, input_shape):
        super(REINFORCE_ConvModel, self).__init__(name="REINFORCE_ConvModel")
        self.conv1 = keras.layers.Conv2D(32, 8, (4, 4), padding='VALID', activation='elu', kernel_initializer='he_normal')
        self.conv2 = keras.layers.Conv2D(64, 4, (2, 2), padding='VALID', activation='elu', kernel_initializer='he_normal')
        self.conv3 = keras.layers.Conv2D(64, 3, (1, 1), padding='VALID', activation='elu', kernel_initializer='he_normal')
        self.flat1 = keras.layers.Flatten()
        self.dense = keras.layers.Dense(h_size, activation='elu', kernel_initializer='he_normal')
        self.out = keras.layers.Dense(n_actions, activation=keras.activations.softmax)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flat1(x)
        x = self.dense(x)
        x = self.out(x)
        return x


class ConvolutionalLayersShared:

    sharedConv_index = 0

    def __init__(self):
        self.conv1 = keras.layers.Conv2D(32, 8, (4, 4), padding='VALID', activation='elu', kernel_initializer='he_normal')
        self.conv2 = keras.layers.Conv2D(64, 4, (2, 2), padding='VALID', activation='elu', kernel_initializer='he_normal')
        self.conv3 = keras.layers.Conv2D(64, 3, (1, 1), padding='VALID', activation='elu', kernel_initializer='he_normal')


class ConvModel(keras.Model):

    def __init__(self, h_size, n_actions, input_shape, ConvolutionalLayersShared=None):
        super(ConvModel, self).__init__(name="ConvQnetwork")
        if ConvolutionalLayersShared is not None:
            self.conv1 = ConvolutionalLayersShared.conv1
            self.conv2 = ConvolutionalLayersShared.conv2
            self.conv3 = ConvolutionalLayersShared.conv3
        else:
            self.conv1 = keras.layers.Conv2D(32, 8, (4, 4), padding='VALID', activation='elu', kernel_initializer='he_normal')
            self.conv2 = keras.layers.Conv2D(64, 4, (2, 2), padding='VALID', activation='elu', kernel_initializer='he_normal')
            self.conv3 = keras.layers.Conv2D(64, 3, (1, 1), padding='VALID', activation='elu', kernel_initializer='he_normal')

        #self.conv4 = keras.layers.Conv2D(32, 2, (2, 2), padding='VALID', activation='elu')
        self.flat1 = keras.layers.Flatten()
        self.dense = keras.layers.Dense(h_size, activation='elu', kernel_initializer='he_normal')
        self.Qout = keras.layers.Dense(n_actions, activation='linear')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flat1(x)
        x = self.dense(x)
        x = self.Qout(x)
        return x


class Dueling_ConvModel(keras.Model):

    def __init__(self, h_size, n_actions, input_shape, ConvolutionalLayersShared=None):
        super(Dueling_ConvModel, self).__init__(name="ConvQnetwork")
        if ConvolutionalLayersShared is not None:
            self.conv1 = ConvolutionalLayersShared.conv1
            self.conv2 = ConvolutionalLayersShared.conv2
            self.conv3 = ConvolutionalLayersShared.conv3
        else:
            self.conv1 = keras.layers.Conv2D(32, 8, (4, 4), padding='VALID', activation='elu', kernel_initializer='he_normal')
            self.conv2 = keras.layers.Conv2D(64, 4, (2, 2), padding='VALID', activation='elu', kernel_initializer='he_normal')
            self.conv3 = keras.layers.Conv2D(64, 3, (1, 1), padding='VALID', activation='elu', kernel_initializer='he_normal')

        #self.conv4 = keras.layers.Conv2D(32, 2, (2, 2), padding='VALID', activation='elu')
        self.flat1 = keras.layers.Flatten()
        #self.dense = keras.layers.Dense(h_size, activation='elu', kernel_initializer='he_normal')

        self.advt_f = keras.layers.Dense(h_size/2, activation='elu', kernel_initializer='he_normal')
        self.advt = keras.layers.Dense(n_actions, activation='linear')

        self.val_f = keras.layers.Dense(h_size/2, activation='elu', kernel_initializer='he_normal')
        self.value = keras.layers.Dense(1, activation='linear')

        #combine the two streams
        self.A = keras.layers.Lambda(lambda advt: advt - tf.reduce_mean(advt, axis=-1, keep_dims=True))
        self.V = keras.layers.Lambda(lambda value: tf.tile(value, [1, n_actions]))
        self.Qout = keras.layers.Add()

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flat1(x)
        A = self.advt_f(x)
        A = self.advt(A)
        V = self.val_f(x)
        V = self.value(V)
        A = self.A(A)
        V = self.V(V)
        x = self.Qout([V, A])
        return x


class QnetworkEager:

    def __init__(self, input_shape, h_size, n_actions, scope_var, device, model, ConvolutionalLayersShared=None):


        with tf.device(device):
            if ConvolutionalLayersShared is None:
                self.model = model(h_size, n_actions, input_shape)
            else:
                self.model = model(h_size, n_actions, input_shape, ConvolutionalLayersShared)

            self.optimizer = tf.train.RMSPropOptimizer(0.001)
            self.global_step = tf.Variable(0)

    def prediction(self, s):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        a = self.model(s)

        return np.argmax(a, 1)

    def Qprediction(self, s):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        return self.model(s).numpy()

    def grad(self, model, inputs, targets, weights):

        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss_value = Losses.mse_loss_imp_w(outputs, targets, weights)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def train(self, s, targetQ, imp_w, max_grad_norm=5):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        loss_value, grads = self.grad(self.model, s, targetQ, imp_w)

        # print("Step: {}, Initial Loss: {}".format(self.global_step.numpy(),loss_value.numpy()))

        grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables), self.global_step)

        # loss_value, _ = self.grad(self.model, s, targetQ, imp_w)

        # print("Step: {},         Loss: {}".format(self.global_step.numpy(), loss_value.numpy()))

        return [None, None]


class ReinforceEager:

    def __init__(self, input_shape, h_size, n_actions, scope_var, device, model, learning_rate):

        tf.set_random_seed(1)
        with tf.device(device):
            self.model = model(h_size, n_actions, input_shape)

            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            self.global_step = tf.Variable(0)

    def prediction(self, s):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        a = self.model(s)

        return np.argmax(a, 1)

    def PolicyPrediction(self, s):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        return self.model(s).numpy()

    def grad(self, model, inputs, targets, advantage):

        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss_value = Losses.reinforce_loss(outputs, targets, advantage)

        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def train(self, s, y, advantage):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        loss_value, grads = self.grad(self.model, s, y, advantage)

        # print("Step: {}, Initial Loss: {}".format(self.global_step.numpy(),loss_value.numpy()))

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables), self.global_step)

        return [None, None]


class SharedConvLayers(keras.Model):
    def __init__(self):
        super(SharedConvLayers, self).__init__(name="SharedConvLayers")
        self.conv1 = keras.layers.Conv2D(32, 8, (4, 4), padding='VALID', activation='elu', kernel_initializer='he_normal')
        self.conv2 = keras.layers.Conv2D(64, 4, (2, 2), padding='VALID', activation='elu', kernel_initializer='he_normal')
        self.conv3 = keras.layers.Conv2D(64, 3, (1, 1), padding='VALID', activation='elu', kernel_initializer='he_normal')
        self.flatten = keras.layers.Flatten()
        self.dense = keras.layers.Dense(256)

    def call(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)

        return [x] # super importante ricordati che negli actor e critic modelli stai indicizzando a 0 ho bisogno di questo per la vae observation


class VAEObservationModel(keras.Model):

    def __init__(self):
        super(VAEObservationModel, self).__init__(name="VAEObservationModel")
        self.conv1 = keras.layers.Conv2D(32, 8, (4, 4), padding='VALID', activation='elu', input_shape=(84, 84, 3))
        self.conv2 = keras.layers.Conv2D(64, 5, (2, 2), padding='VALID', activation='elu')
        self.conv3 = keras.layers.Conv2D(64, 3, (1, 1), padding='VALID', activation='elu')
        self.flatten = keras.layers.Flatten()
        self.dense = keras.layers.Dense(512)
        self.z_mean = keras.layers.Dense(32, name='z_mean')
        self.z_log_var = keras.layers.Dense(32, name='z_log_var')
        self.z = keras.layers.Lambda(self.sampling, name='z')
        self.deconv1 = keras.layers.Conv2DTranspose(64, 3, (1, 1), padding='VALID', activation='elu')
        self.deconv2 = keras.layers.Conv2DTranspose(32, 6, (2, 2), padding='VALID', activation='elu')
        self.deconv3 = keras.layers.Conv2DTranspose(3, 8, (4, 4), padding='VALID', activation='elu')

    def call(self, x):
        #print("input: ", "input shape: ", x.shape)
        x = self.conv1(x)
        #print("conv1: ", "output shape: ", x.numpy().shape)
        x = self.conv2(x)
        #print("conv2: ", "output shape: ", x.numpy().shape)
        conv3 = self.conv3(x)
        #print("conv3: ", "output shape: ", conv3.numpy().shape)
        x = self.flatten(conv3)
        #print("flatten: ", "output shape: ", x.numpy().shape)
        dense = self.dense(x)
        #print("dense: ", "output shape: ", dense.numpy().shape)
        shape = conv3.numpy().shape
        z_mean = self.z_mean(dense)
        #print("z_mean: ", "output shape: ", z_mean.numpy().shape)
        z_log_var = self.z_log_var(dense)
        #print("z_log_var: ", "output shape: ", z_log_var.numpy().shape)
        x = self.z([z_mean, z_log_var])
        #print("z: ", "output shape: ", x.numpy().shape)
        x = keras.layers.Dense(shape[1]*shape[2]*shape[3], activation='elu')(x)
        x = keras.layers.Reshape((shape[1], shape[2], shape[3]))(x)
        #print("reshape:", "output shape: ", x.numpy().shape)
        x = self.deconv1(x)
        #print("deconv1:", "output shape: ", x.numpy().shape)
        x = self.deconv2(x)
        #print("deconv2:", "output shape: ", x.numpy().shape)
        x = self.deconv3(x)
        #print("deconv3:", "output shape: ", x.numpy().shape)

        return dense, z_mean, z_log_var, x

    # reparameterization trick
    # instead of sampling from Q(z|X), sample eps = N(0,I)
    # then z = z_mean + sqrt(var)*eps
    def sampling(self, args):

        """Reparameterization trick by sampling fr an isotropic unit Gaussian.
        # Arguments
            args (tensor): mean and log of variance of Q(z|X)
        # Returns
            z (tensor): sampled latent vector
        """

        z_mean, z_log_var = args
        batch = keras.backend.shape(z_mean)[0]
        dim = keras.backend.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = keras.backend.random_normal(shape=(batch, dim))

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class SharedDenseLayers(keras.Model):
    def __init__(self, h_size):
        super(SharedDenseLayers, self).__init__(name="SharedDenseLayers")
        self.dense1 = keras.layers.Dense(h_size, activation='elu', kernel_initializer='he_normal')
        self.dense2 = keras.layers.Dense(h_size, activation='elu', kernel_initializer='he_normal')

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)

        return [x]


class CriticNetwork(keras.Model):
    def __init__(self, h_size):
        super(CriticNetwork, self).__init__(name="CriticNetwork")
        self.dense1 = keras.layers.Dense(h_size, activation='elu', kernel_initializer='he_normal')
        self.dense2 = keras.layers.Dense(h_size, activation='elu', kernel_initializer='he_normal')
        self.out = keras.layers.Dense(1, activation='linear')

    def call(self, x):

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.out(x)
        return x


class ActorNetwork(keras.Model):
    def __init__(self, h_size, n_actions):
        super(ActorNetwork, self).__init__(name="ActorNetwork")
        self.dense1 = keras.layers.Dense(h_size, activation='elu', kernel_initializer='he_normal')
        self.dense2 = keras.layers.Dense(h_size, activation='elu', kernel_initializer='he_normal')
        self.out = keras.layers.Dense(n_actions, activation=keras.activations.softmax)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.out(x)
        return x


class ActorCriticNetwork(keras.Model):

    def __init__(self, critic_model, actor_model, shared_observation_model=None):
        super(ActorCriticNetwork, self).__init__(name="ActorCriticNetwork")
        self.shared_observation_model = shared_observation_model
        self.critic_model = critic_model
        self.actor_model = actor_model

    def call(self, x):

        if self.shared_observation_model is not None:

            x = self.shared_observation_model(x)[0] # Just the dense output

        actor = self.actor_model(x)

        critic = self.critic_model(x)

        return actor, critic


class A2CEagerSync:

    def __init__(self, h_size, n_actions, scope_var, device, model_critic, model_actor, learning_rate, weight_mse, weight_ce, shared_observation_model=None, train_observation=False):

        tf.set_random_seed(1)
        with tf.device(device):
            self.shared_observation_model = shared_observation_model
            self.model_critic = model_critic(h_size)
            self.model_actor = model_actor(h_size, n_actions)
            self.model_actor_critic = ActorCriticNetwork(self.model_critic, self.model_actor, self.shared_observation_model)
            self.train_observation = train_observation
            self.weight_mse = weight_mse
            self.weight_ce = weight_ce

            print("\n ACTOR CRITIC MODEL \n")

            slim.model_analyzer.analyze_vars(self.model_actor_critic.trainable_variables, print_info=True)

            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            if self.train_observation:
                self.optimizer_observation = tf.train.RMSPropOptimizer(learning_rate=1e-3)
            self.global_step = tf.Variable(0)

    def prediction_actor(self, s):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        return self.model_actor_critic(s)[0].numpy()

    def prediction_critic(self, s):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        return self.model_actor_critic(s)[1].numpy()


    def grad(self, model_actor_critic, inputs, targets, one_hot_a, advantage, weight_ce, weight_mse = 0.5):

        with tf.GradientTape() as tape:
            softmax_logits, value_critic = model_actor_critic(inputs)
            loss_pg = Losses.reinforce_loss(softmax_logits, one_hot_a, advantage)
            loss_ce = Losses.entropy_exploration_loss(softmax_logits)
            loss_critic = Losses.mse_loss(value_critic, targets)
            loss_value = (weight_mse * loss_critic) + (loss_pg - weight_ce * loss_ce)

        return loss_value, tape.gradient(loss_value, model_actor_critic.trainable_variables)

    def grad_observation(self, model, inputs):

        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss_value = Losses.vae_loss(inputs, outputs[3], outputs[1], outputs[2])

        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def train(self, s, y, one_hot_a, advantage, max_grad_norm=5):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        loss_value, grads = self.grad(self.model_actor_critic, s, y, one_hot_a, advantage, self.weight_ce, self.weight_mse)

        grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        self.optimizer.apply_gradients(zip(grads, self.model_actor_critic.trainable_variables), self.global_step)

        return [None, None]

    def train_obs(self, s):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        loss_value, grads = self.grad_observation(self.shared_observation_model, s)

        print("OBSERVATION", loss_value)

        # grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        self.optimizer_observation.apply_gradients(zip(grads, self.shared_observation_model.trainable_variables),
                                                   self.global_step)

        return [None, None]


class A2C_SIL_EagerSync:

    def __init__(self, h_size, n_actions, scope_var, device, model_critic, model_actor, learning_rate_online, learning_rate_imitation, weight_mse, weight_sil_mse, weight_ce, shared_observation_model=None, train_observation=False):

        tf.set_random_seed(1)
        with tf.device(device):
            self.shared_observation_model = shared_observation_model
            self.model_critic = model_critic(h_size)
            self.model_actor = model_actor(h_size, n_actions)
            self.model_actor_critic = ActorCriticNetwork(self.model_critic, self.model_actor, self.shared_observation_model)
            self.train_observation = train_observation
            self.weight_mse = weight_mse
            self.weight_sil_mse = weight_sil_mse
            self.weight_ce = weight_ce

            print("\n ACTOR CRITIC MODEL \n")

            slim.model_analyzer.analyze_vars(self.model_actor_critic.trainable_variables, print_info=True)

            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate_online)
            #self.optimizer_imitation = tf.train.RMSPropOptimizer(learning_rate=learning_rate_imitation)
            if self.train_observation:
                self.optimizer_observation = tf.train.RMSPropOptimizer(learning_rate=1e-3)
            self.global_step = tf.Variable(0)

    def prediction(self, s):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        a = self.model_actor(s)

        return np.argmax(a, 1)

    def prediction_actor(self, s):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        return self.model_actor_critic(s)[0].numpy()

    def prediction_critic(self, s):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        return self.model_actor_critic(s)[1].numpy()

    def grad(self, model_actor_critic, inputs, targets, one_hot_a, advantage, weight_ce, weight_mse = 0.5):

        with tf.GradientTape() as tape:
            softmax_logits, value_critic = model_actor_critic(inputs)
            loss_pg = Losses.reinforce_loss(softmax_logits, one_hot_a, advantage)
            loss_ce = Losses.entropy_exploration_loss(softmax_logits)
            loss_critic = Losses.mse_loss(value_critic, targets)
            loss_value = (weight_mse * loss_critic) + (loss_pg - weight_ce * loss_ce)

        return loss_value, tape.gradient(loss_value, model_actor_critic.trainable_variables)

    def grad_imitation(self, model_actor_critic, inputs, targets, one_hot_a, advantage, imp_w, weight_mse=0.5, weight_sil_mse=0.01):

        with tf.GradientTape() as tape:
            softmax_logits, value_critic = model_actor_critic(inputs)
            loss_pg_imitation = Losses.reinforce_loss_imp_w(softmax_logits, one_hot_a, advantage, imp_w)
            loss_critic_imitation = (weight_mse * Losses.mse_loss_self_imitation_learning_imp_w(value_critic, targets, imp_w))
            loss_value = weight_sil_mse * loss_critic_imitation + loss_pg_imitation

        return loss_value, tape.gradient(loss_value, model_actor_critic.trainable_variables)

    def grad_observation(self, model, inputs):

        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss_value = Losses.vae_loss(inputs, outputs[3], outputs[1], outputs[2])

        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def train(self, s, y, one_hot_a, advantage, max_grad_norm=5):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        loss_value, grads = self.grad(self.model_actor_critic, s, y, one_hot_a, advantage, self.weight_ce, self.weight_mse)

        grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        self.optimizer.apply_gradients(zip(grads, self.model_actor_critic.trainable_variables), self.global_step)

        return [None, None]

    def train_imitation(self, s, y, one_hot_a, advantage, imp_w, max_grad_norm=5):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        loss_value, grads = self.grad_imitation(self.model_actor_critic, s, y, one_hot_a, advantage, imp_w, self.weight_mse, self.weight_sil_mse)

        grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        self.optimizer.apply_gradients(zip(grads, self.model_actor_critic.trainable_variables), self.global_step)   # separate optimizer why doesn't work?

    def train_obs(self, s):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        loss_value, grads = self.grad_observation(self.shared_observation_model, s)

        print("OBSERVATION", loss_value)

        # grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        self.optimizer_observation.apply_gradients(zip(grads, self.shared_observation_model.trainable_variables),
                                                   self.global_step)

        return [None, None]


class A2CEager:

    def __init__(self, h_size, n_actions, scope_var, device, model_critic, model_actor, learning_rate_actor, learning_rate_critic, shared_observation_model=None, train_observation=False):

        tf.set_random_seed(1)
        with tf.device(device):
            self.shared_observation_model = shared_observation_model
            self.model_critic = model_critic(h_size, self.shared_observation_model)
            self.model_actor = model_actor(h_size, n_actions, self.shared_observation_model)
            self.train_observation = train_observation

            print("\n OBSERVATION ENCODING MODEL \n")

            slim.model_analyzer.analyze_vars(self.shared_observation_model.trainable_variables, print_info=True)

            print("\n ACTOR MODEL \n")

            slim.model_analyzer.analyze_vars(self.model_actor.trainable_variables, print_info=True)

            print("\n CRITIC MODEL \n")

            slim.model_analyzer.analyze_vars(self.model_critic.trainable_variables, print_info=True)

            self.optimizer_critic = tf.train.RMSPropOptimizer(learning_rate=learning_rate_critic)
            self.optimizer_actor = tf.train.RMSPropOptimizer(learning_rate=learning_rate_actor)
            if self.train_observation:
                self.optimizer_observation = tf.train.RMSPropOptimizer(learning_rate=1e-3)
            self.global_step = tf.Variable(0)

    def prediction(self, s):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        a = self.model_actor(s)

        return np.argmax(a, 1)

    def prediction_actor(self, s):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        return self.model_actor(s).numpy()

    def prediction_critic(self, s):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        return self.model_critic(s).numpy()

    def grad(self, model_actor, model_critic, inputs, targets, one_hot_a, advantage, weight_ce, weight_mse = 0.5):

        with tf.GradientTape() as tape:
            softmax_logits = model_actor(inputs)
            value_critic = model_critic(inputs)
            loss_pg = Losses.reinforce_loss(softmax_logits, one_hot_a, advantage)
            loss_ce = Losses.entropy_exploration_loss(softmax_logits)
            loss_critic = Losses.mse_loss(value_critic, targets)
            loss_value = (weight_mse * loss_critic) + (loss_pg - weight_ce * loss_ce)

        return loss_value, tape.gradient(loss_value, [model_actor.trainable_variables, model_critic.trainable_variables])

    def train(self, s, y, one_hot_a, advantage, weight_ce, weight_mse = 0.5, max_grad_norm=5):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        loss_value, grads = self.grad(self.model_actor, self.model_critic, s, y, one_hot_a, advantage, weight_ce, weight_mse)

        #grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        self.optimizer_critic.apply_gradients(zip(grads, [self.model_actor.trainable_variables, self.model_critic.trainable_variables]), self.global_step)

        return [None, None]


    def grad_actor(self, model, inputs, one_hot_a, advantage, weight_ce):

        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss_pg = Losses.reinforce_loss(outputs, one_hot_a, advantage)
            loss_ce = Losses.entropy_exploration_loss(outputs)
            loss_value = loss_pg - weight_ce * loss_ce

        #print(outputs.numpy(), loss_pg.numpy(), loss_ce.numpy(), loss_value.numpy())

        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def grad_critic(self, model, inputs, targets, weight_mse=0.5):

        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss_value = weight_mse * Losses.mse_loss(outputs, targets)

        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def grad_observation(self, model, inputs):

        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss_value = Losses.vae_loss(inputs, outputs[3], outputs[1], outputs[2])

        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def train_critic(self, s, y, max_grad_norm=5):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        loss_value, grads = self.grad_critic(self.model_critic, s, y)

        #print("CRITIC", loss_value)

        grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        #print(grad_norm)

        self.optimizer_critic.apply_gradients(zip(grads, self.model_critic.trainable_variables), self.global_step)

        return [None, None]

    def train_actor(self, s, one_hot_a, advantage, weight_ce = 0, max_grad_norm=5):
        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        loss_value, grads = self.grad_actor(self.model_actor, s, one_hot_a, advantage, weight_ce)

        #print("ACTOR", loss_value)

        grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        #print(grad_norm)

        self.optimizer_actor.apply_gradients(zip(grads, self.model_actor.trainable_variables),
                                             self.global_step)

        return [None, None]

    def train_obs(self, s):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        loss_value, grads = self.grad_observation(self.shared_observation_model, s)

        print("OBSERVATION", loss_value)

        # grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        self.optimizer_observation.apply_gradients(zip(grads, self.shared_observation_model.trainable_variables),
                                                   self.global_step)

        return [None, None]


class A2C_SIL_Eager:

    def __init__(self, h_size, n_actions, scope_var, device, model_critic, model_actor, learning_rate_actor, learning_rate_critic, shared_observation_model=None, train_observation=False):

        tf.set_random_seed(1)
        with tf.device(device):
            self.shared_observation_model = shared_observation_model
            self.model_critic = model_critic(h_size, self.shared_observation_model)
            self.model_actor = model_actor(h_size, n_actions, self.shared_observation_model)
            self.train_observation = train_observation

            print("\n OBSERVATION ENCODING MODEL \n")

            slim.model_analyzer.analyze_vars(self.shared_observation_model.trainable_variables, print_info=True)

            print("\n ACTOR MODEL \n")

            slim.model_analyzer.analyze_vars(self.model_actor.trainable_variables, print_info=True)

            print("\n CRITIC MODEL \n")

            slim.model_analyzer.analyze_vars(self.model_critic.trainable_variables, print_info=True)

            self.optimizer_critic = tf.train.RMSPropOptimizer(learning_rate=learning_rate_critic)
            self.optimizer_actor = tf.train.RMSPropOptimizer(learning_rate=learning_rate_actor)
            if self.train_observation:
                self.optimizer_observation = tf.train.RMSPropOptimizer(learning_rate=1e-3)
            self.global_step = tf.Variable(0)

    def prediction(self, s):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        a = self.model_actor(s)

        return np.argmax(a, 1)

    def prediction_actor(self, s):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        return self.model_actor(s).numpy()

    def prediction_critic(self, s):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        return self.model_critic(s).numpy()


    def grad_actor(self, model, inputs, one_hot_a, advantage, weight_ce):

        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss_pg = Losses.reinforce_loss(outputs, one_hot_a, advantage)
            loss_ce = Losses.entropy_exploration_loss(outputs)
            loss_value = loss_pg - weight_ce * loss_ce

        #print(outputs.numpy(), loss_pg.numpy(), loss_ce.numpy(), loss_value.numpy())

        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def grad_actor_imitation(self, model, inputs, one_hot_a, advantage):

        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss_pg = Losses.reinforce_loss(outputs, one_hot_a, advantage)
            loss_value = loss_pg

        #print(outputs.numpy(), loss_pg.numpy(), loss_ce.numpy(), loss_value.numpy())

        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def grad_critic(self, model, inputs, targets, weight_mse=0.5):

        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss_value = weight_mse * Losses.mse_loss(outputs, targets)

        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def grad_critic_imitation(self, model, inputs, targets, weight_mse=0.5, weight_sil_mse=0.01):

        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss_value = weight_sil_mse * (weight_mse * Losses.mse_loss_self_imitation_learning(outputs, targets))

        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def grad_observation(self, model, inputs):

        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss_value = Losses.vae_loss(inputs, outputs[3], outputs[1], outputs[2])

        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def train_critic(self, s, y, max_grad_norm=0.5):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        loss_value, grads = self.grad_critic(self.model_critic, s, y)

        #print("CRITIC", loss_value)

        #grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        self.optimizer_critic.apply_gradients(zip(grads, self.model_critic.trainable_variables), self.global_step)

        return [None, None]

    def train_actor(self, s, one_hot_a, advantage, weight_ce = 0, max_grad_norm=0.5):
        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        loss_value, grads = self.grad_actor(self.model_actor, s, one_hot_a, advantage, weight_ce)

        #print("ACTOR", loss_value)

        #grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        self.optimizer_actor.apply_gradients(zip(grads, self.model_actor.trainable_variables),
                                             self.global_step)

        return [None, None]

    def train_actor_imitation(self, s, one_hot_a, advantage, max_grad_norm=0.5):
        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        loss_value, grads = self.grad_actor_imitation(self.model_actor, s, one_hot_a, advantage)

        #print("ACTOR", loss_value)

        #grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        self.optimizer_actor.apply_gradients(zip(grads, self.model_actor.trainable_variables),
                                             self.global_step)

        return [None, None]

    def train_obs(self, s):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        loss_value, grads = self.grad_observation(self.shared_observation_model, s)

        print("OBSERVATION", loss_value)

        # grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        self.optimizer_observation.apply_gradients(zip(grads, self.shared_observation_model.trainable_variables),
                                                   self.global_step)

        return [None, None]

    def train_critic_imitation(self, s, y, max_grad_norm=0.5):
        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        loss_value, grads = self.grad_critic_imitation(self.model_critic, s, y)

        # print("CRITIC", loss_value)

        # grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        self.optimizer_critic.apply_gradients(zip(grads, self.model_critic.trainable_variables), self.global_step)

        return [None, None]


class A2C_SIL_Eager_IW:

    def __init__(self, h_size, n_actions, scope_var, device, model_critic, model_actor, learning_rate_actor, learning_rate_critic, shared_observation_model=None, train_observation=False):

        tf.set_random_seed(1)
        with tf.device(device):
            self.shared_observation_model = shared_observation_model
            self.model_critic = model_critic(h_size, self.shared_observation_model)
            self.model_actor = model_actor(h_size, n_actions, self.shared_observation_model)
            self.train_observation = train_observation

            print("\n OBSERVATION ENCODING MODEL \n")

            slim.model_analyzer.analyze_vars(self.shared_observation_model.trainable_variables, print_info=True)

            print("\n ACTOR MODEL \n")

            slim.model_analyzer.analyze_vars(self.model_actor.trainable_variables, print_info=True)

            print("\n CRITIC MODEL \n")

            slim.model_analyzer.analyze_vars(self.model_critic.trainable_variables, print_info=True)

            self.optimizer_critic = tf.train.RMSPropOptimizer(learning_rate=learning_rate_critic)
            self.optimizer_actor = tf.train.RMSPropOptimizer(learning_rate=learning_rate_actor)
            self.optimizer_critic_imitation = tf.train.RMSPropOptimizer(learning_rate=learning_rate_critic)
            self.optimizer_actor_imitation = tf.train.RMSPropOptimizer(learning_rate=learning_rate_actor)
            if self.train_observation:
                self.optimizer_observation = tf.train.RMSPropOptimizer(learning_rate=1e-3)
            self.global_step = tf.Variable(0)

    def prediction(self, s):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        a = self.model_actor(s)

        return np.argmax(a, 1)

    def prediction_actor(self, s):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        return self.model_actor(s).numpy()

    def prediction_critic(self, s):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        return self.model_critic(s).numpy()

    def grad_actor(self, model, inputs, one_hot_a, advantage, weight_ce):

        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss_pg = Losses.reinforce_loss(outputs, one_hot_a, advantage)
            loss_ce = Losses.entropy_exploration_loss(outputs)
            loss_value = loss_pg - weight_ce * loss_ce

        #print(outputs.numpy(), loss_pg.numpy(), loss_ce.numpy(), loss_value.numpy())

        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def grad_critic(self, model, inputs, targets, weight_mse=0.5):

        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss_value = weight_mse * Losses.mse_loss(outputs, targets)

        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def grad_actor_imitation(self, model, inputs, one_hot_a, advantage, imp_w):

        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss_pg = Losses.reinforce_loss_imp_w(outputs, one_hot_a, advantage, imp_w)
            loss_value = loss_pg

        #print(outputs.numpy(), loss_pg.numpy(), loss_ce.numpy(), loss_value.numpy())

        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def grad_critic_imitation(self, model, inputs, targets, imp_w, weight_mse=0.5, weight_sil_mse=0.01):

        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss_value = weight_sil_mse * (weight_mse * Losses.mse_loss_self_imitation_learning_imp_w(outputs, targets, imp_w))

        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def grad_observation(self, model, inputs):

        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss_value = Losses.vae_loss(inputs, outputs[3], outputs[1], outputs[2])

        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def train_critic(self, s, y, max_grad_norm=0.5):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        loss_value, grads = self.grad_critic(self.model_critic, s, y)

        #print("CRITIC", loss_value)

        grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        self.optimizer_critic.apply_gradients(zip(grads, self.model_critic.trainable_variables), self.global_step)

        return [None, None]

    def train_actor(self, s, one_hot_a, advantage, weight_ce = 0, max_grad_norm=0.5):
        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        loss_value, grads = self.grad_actor(self.model_actor, s, one_hot_a, advantage, weight_ce)

        #print("ACTOR", loss_value)

        grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        self.optimizer_actor.apply_gradients(zip(grads, self.model_actor.trainable_variables),
                                             self.global_step)

        return [None, None]

    def train_obs(self, s):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        loss_value, grads = self.grad_observation(self.shared_observation_model, s)

        print("OBSERVATION", loss_value)

        # grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        self.optimizer_observation.apply_gradients(zip(grads, self.shared_observation_model.trainable_variables),
                                                   self.global_step)

        return [None, None]

    def train_critic_imitation(self, s, y, imp_w, max_grad_norm=0.5):
        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        loss_value, grads = self.grad_critic_imitation(self.model_critic, s, y, imp_w)

        # print("CRITIC", loss_value)

        grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        self.optimizer_critic_imitation.apply_gradients(zip(grads, self.model_critic.trainable_variables), self.global_step)

        return [None, None]

    def train_actor_imitation(self, s, one_hot_a, advantage, imp_w, max_grad_norm=0.5):
        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        loss_value, grads = self.grad_actor_imitation(self.model_actor, s, one_hot_a, advantage, imp_w)

        # print("CRITIC", loss_value)

        grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        self.optimizer_actor_imitation.apply_gradients(zip(grads, self.model_actor.trainable_variables), self.global_step)

        return [None, None]
