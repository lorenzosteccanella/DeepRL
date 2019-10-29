

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