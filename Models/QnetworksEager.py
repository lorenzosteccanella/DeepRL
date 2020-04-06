import tensorflow as tf
from tensorflow import keras
import numpy as np
from Losses.Losses import Losses
import inspect

class SharedConvLayers(keras.Model):
    def __init__(self, learning_rate_observation_adjust=1):
        super(SharedConvLayers, self).__init__(name="SharedConvLayers")
        self.conv1 = keras.layers.Conv2D(32, 8, (4, 4), padding='VALID', activation='elu', kernel_initializer='he_normal', )
        self.conv2 = keras.layers.Conv2D(64, 4, (2, 2), padding='VALID', activation='elu', kernel_initializer='he_normal')
        self.conv3 = keras.layers.Conv2D(64, 3, (1, 1), padding='VALID', activation='elu', kernel_initializer='he_normal')
        self.flatten = keras.layers.Flatten()
        self.dense = keras.layers.Dense(256, activation='elu', kernel_initializer='he_normal')
        self.normalization_layer = keras.layers.LayerNormalization()
        self.learning_rate_adjust = learning_rate_observation_adjust

    def call(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        denseOut = self.dense(x)
        #denseOut = self.normalization_layer(denseOut)
        x = self.learning_rate_adjust * denseOut + (1-self.learning_rate_adjust) * tf.stop_gradient(denseOut)  # U have to test this!!!

        return [x, denseOut] # super importante ricordati che negli actor e critic modelli stai indicizzando a 0 ho bisogno di questo per la vae observation

    def prediction_h(self, s):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        return self.call(s)[1].numpy()





class SharedDenseLayers(keras.Model):
    def __init__(self, h_size=256, learning_rate_observation_adjust=1):
        super(SharedDenseLayers, self).__init__(name="SharedDenseLayers")
        self.dense1 = keras.layers.Dense(h_size, activation='elu', kernel_initializer='he_normal')
        self.dense2 = keras.layers.Dense(h_size, activation='elu', kernel_initializer='he_normal')
        self.learning_rate_adjust = learning_rate_observation_adjust

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.learning_rate_adjust * x + (1 - self.learning_rate_adjust) * tf.stop_gradient(x)  # U have to test this!!!

        return [x, x]


class QNetwork(keras.Model):

    def __init__(self, head_model, shared_observation_model=None):
        super(QNetwork, self).__init__(name="QNetwork")
        self.shared_observation_model = shared_observation_model
        self.head_model = head_model

    def call(self, x):

        if self.shared_observation_model is not None:

            obs = self.shared_observation_model(x)[0] # Just the dense output

        denseOut = self.shared_observation_model(x)[1]

        Q = self.head_model(obs)

        return Q, denseOut


class DenseModel(keras.Model):

    def __init__(self, h_size, n_actions):
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

    def __init__(self, h_size, n_actions):
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


class ConvModel(keras.Model):

    def __init__(self, h_size, n_actions, ConvolutionalLayersShared=None):
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

class Dueling_head(keras.Model):

    def __init__(self, h_size, n_actions) :
        super(Dueling_head, self).__init__(name="Dueling_head")

        self.advt_f = keras.layers.Dense(h_size/2, activation='elu', kernel_initializer='he_normal')
        self.advt = keras.layers.Dense(n_actions, activation='linear')

        self.val_f = keras.layers.Dense(h_size/2, activation='elu', kernel_initializer='he_normal')
        self.value = keras.layers.Dense(1, activation='linear')

        #combine the two streams
        self.A = keras.layers.Lambda(lambda advt: advt - tf.reduce_mean(advt, axis=-1, keep_dims=True))
        self.V = keras.layers.Lambda(lambda value: tf.tile(value, [1, n_actions]))
        self.Qout = keras.layers.Add()

    def call(self, x):
        A = self.advt_f(x)
        A = self.advt(A)
        V = self.val_f(x)
        V = self.value(V)
        A = self.A(A)
        V = self.V(V)
        x = self.Qout([V, A])
        return x


class Dueling_ConvModel(keras.Model):

    def __init__(self, h_size, n_actions, ConvolutionalLayersShared=None):
        super(Dueling_ConvModel, self).__init__(name="ConvQnetwork")
        if ConvolutionalLayersShared is not None:
            self.conv1 = ConvolutionalLayersShared.conv1
            self.conv2 = ConvolutionalLayersShared.conv2
            self.conv3 = ConvolutionalLayersShared.conv3
        else:
            self.conv1 = keras.layers.Conv2D(32, 8, (4, 4), padding='VALID', activation='elu', kernel_initializer='he_normal')
            self.conv2 = keras.layers.Conv2D(64, 4, (2, 2), padding='VALID', activation='elu', kernel_initializer='he_normal')
            self.conv3 = keras.layers.Conv2D(64, 3, (1, 1), padding='VALID', activation='elu', kernel_initializer='he_normal')

        self.flat1 = keras.layers.Flatten()
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

    def __init__(self, h_size, n_actions, head_model, learning_rate, shared_observation_model=None, learning_rate_observation_adjust=1):

        if inspect.isclass(shared_observation_model):
            self.shared_observation_model = shared_observation_model(learning_rate_observation_adjust)
        else:
            self.shared_observation_model = shared_observation_model

        if inspect.isclass(head_model):
            self.head_model = head_model(h_size, n_actions)
        else:
            self.head_model = head_model

        self.model = QNetwork(self.head_model, self.shared_observation_model)

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate)
        self.global_step = tf.Variable(0)

    def prediction(self, s):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        a = self.model(s)[0].numpy()

        return np.argmax(a, 1)

    def Qprediction(self, s):

        s = np.array(s, dtype=np.float32)

        s = tf.convert_to_tensor(s)

        return self.model(s)[0].numpy()

    def grad(self, model, inputs, targets, weights):

        with tf.GradientTape() as tape:
            outputs = model(inputs)[0]
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