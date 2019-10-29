import tensorflow as tf
from tensorflow import keras
import numpy as np
from Losses.Losses import Losses


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

    def __init__(self, h_size, n_actions, model, ConvolutionalLayersShared=None):

        if ConvolutionalLayersShared is None:
            self.model = model(h_size, n_actions)
        else:
            self.model = model(h_size, n_actions, ConvolutionalLayersShared)

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