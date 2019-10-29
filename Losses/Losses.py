import tensorflow as tf


class Losses:

    # Huber loss.
    # https://en.wikipedia.org/wiki/Huber_loss

    @staticmethod
    def huber_loss(x, y):

        return tf.losses.huber_loss(labels=y, predictions=x, delta=2.0)

    @staticmethod
    def huber_loss_imp_w(x, y, weights):

        return tf.losses.huber_loss(labels=y, predictions=x, weights=weights, delta=2.0)

    @staticmethod
    def mse_loss(x, y):
        predictions = tf.to_float(x)
        labels = tf.to_float(y)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        losses = tf.math.squared_difference(predictions, labels)
        loss = tf.reduce_mean(losses)
        return loss

    @staticmethod
    def mse_loss_imp_w(x, y, imp_w):
        predictions = tf.to_float(x)
        labels = tf.to_float(y)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        losses = tf.math.squared_difference(predictions, labels)
        loss = tf.losses.compute_weighted_loss(losses, imp_w)
        return loss

    @staticmethod
    def reinforce_loss(softmax_logits, one_hot_a, advantage):

        neg_log_policy = - tf.log(tf.clip_by_value(softmax_logits, 1e-7, 1))

        reinforce_loss = tf.reduce_mean(tf.reduce_sum(neg_log_policy * one_hot_a, axis=1) * advantage)

        return reinforce_loss

    @staticmethod
    def reinforce_loss_imp_w(softmax_logits, one_hot_a, advantage, imp_w):

        neg_log_policy = - tf.log(tf.clip_by_value(softmax_logits, 1e-7, 1))

        reinforce_losses = tf.reduce_sum(neg_log_policy * one_hot_a, axis=1) * advantage

        imp_w = tf.squeeze(imp_w)

        reinforce_loss = tf.losses.compute_weighted_loss(reinforce_losses, imp_w)

        return reinforce_loss

    @staticmethod
    def reinforce_loss_2(logits, one_hot_a, advantage):

        softmax_logits = softmax_logits * one_hot_a

        action_probs = tf.reduce_max(softmax_logits, axis=1)

        reinforce_loss = tf.reduce_sum(- tf.log(tf.clip_by_value(action_probs, 1e-7, 1))*advantage)

        #neg_log_prob = tf.nn.weighted_cross_entropy_with_logits(targets=y, logits=x, pos_weight=advantage)

        #loss = neg_log_prob

        #print("\n\n Logits: \n\n", x, "\n\n Labels: \n\n", y, "\n\nAdvantages: \n\n", advantage, "\n\nneg_log_prob: \n\n", neg_log_prob, "\n\nloss: \n\n", loss)

        return reinforce_loss

    @staticmethod
    def entropy_exploration_loss(x):

        neg_log_policy = - tf.log(tf.clip_by_value(x, 1e-7, 1)) # probability of softmax can't be more then 1
        loss = tf.reduce_mean(tf.reduce_sum(x * neg_log_policy, axis=1))

        return loss

    @staticmethod
    def kl_loss(z_log_var, z_mean):
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.math.reduce_sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        return kl_loss

    @staticmethod
    def vae_loss(img_in, img_out, z_log_var, z_mean):
        shape = img_in.shape
        reconstruction_loss = Losses.mse_loss(img_in, img_out)
        #reconstruction_loss *= shape[1] * shape[2] * shape[3]
        kl_loss = Losses.kl_loss(z_log_var, z_mean)

        return tf.reduce_mean(reconstruction_loss+kl_loss)

    @staticmethod
    def mse_loss_self_imitation_learning(x, y):
        predictions = tf.to_float(x)
        labels = tf.to_float(y)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        difference = labels - predictions
        difference = tf.clip_by_value(difference, 0, tf.float32.max)
        losses = difference ** 2
        loss = tf.reduce_mean(losses)
        return loss

    @staticmethod
    def mse_loss_self_imitation_learning_imp_w(x, y, imp_w):
        predictions = tf.to_float(x)
        labels = tf.to_float(y)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        difference = labels - predictions
        difference = tf.clip_by_value(difference, 0, tf.float32.max)
        losses = difference ** 2
        loss = tf.losses.compute_weighted_loss(losses, imp_w)
        return loss