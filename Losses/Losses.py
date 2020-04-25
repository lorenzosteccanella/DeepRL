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
    def ppo_loss(logits, old_logits, one_hot_a, advantage, e_clip=0.2):
        softmax_logits = tf.nn.softmax(logits)
        old_softmax_logits = tf.nn.softmax(old_logits)
        log_policy = tf.log(tf.clip_by_value(softmax_logits, 1e-7, 1))
        old_log_policy = tf.log(tf.clip_by_value(old_softmax_logits, 1e-7, 1))
        tensor_one_hot = tf.convert_to_tensor(one_hot_a, dtype=tf.float32)
        tensor_advantage = tf.convert_to_tensor(advantage, dtype=tf.float32)
        prob_ratio = tf.exp(log_policy - old_log_policy)
        clip_prob = tf.clip_by_value(prob_ratio, 1. - e_clip, 1. + e_clip)
        ppo_loss = - tf.reduce_mean(tf.minimum((tf.reduce_sum(prob_ratio * tensor_one_hot, axis=1) * tf.stop_gradient(tensor_advantage)),
                                               (tf.reduce_sum(clip_prob * tensor_one_hot, axis=1) * tf.stop_gradient(tensor_advantage))))
        return ppo_loss

    @staticmethod
    def reinforce_loss(logits, one_hot_a, advantage):
        softmax_logits = tf.nn.softmax(logits)
        neg_log_policy = - tf.log(tf.clip_by_value(softmax_logits, 1e-7, 1))
        tensor_one_hot = tf.convert_to_tensor(one_hot_a, dtype=tf.float32)
        tensor_advantage = tf.convert_to_tensor(advantage, dtype=tf.float32)
        #reinforce_loss = tf.reduce_mean(tf.reduce_sum(neg_log_policy * tensor_one_hot, axis=1) * tf.stop_gradient(tensor_advantage))
        reinforce_loss = tf.reduce_sum(tf.reduce_sum(neg_log_policy * tensor_one_hot, axis=1) * tf.stop_gradient(tensor_advantage))
        #reinforce_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tensor_one_hot, logits=logits) * \
        #                                 tf.stop_gradient(tensor_advantage)
        return reinforce_loss

    @staticmethod
    def reinforce_loss_imp_w(logits, one_hot_a, advantage, imp_w):

        softmax_logits = tf.nn.softmax(logits)
        neg_log_policy = - tf.log(tf.clip_by_value(softmax_logits, 1e-7, 1))
        tensor_one_hot = tf.convert_to_tensor(one_hot_a, dtype=tf.float32)
        tensor_advantage = tf.convert_to_tensor(advantage, dtype=tf.float32)

        reinforce_losses = tf.reduce_sum(neg_log_policy * tensor_one_hot, axis=1) * tf.stop_gradient(tensor_advantage)

        imp_w = tf.squeeze(imp_w)

        reinforce_loss = tf.losses.compute_weighted_loss(reinforce_losses, imp_w)

        return reinforce_loss

    @staticmethod
    def reinforce_loss_2(logits, one_hot_a, advantage):
        softmax_logits = tf.nn.softmax(logits)
        softmax_logits = softmax_logits * one_hot_a

        action_probs = tf.reduce_max(softmax_logits, axis=1)

        reinforce_loss = tf.reduce_sum(- tf.log(tf.clip_by_value(action_probs, 1e-7, 1))*advantage)

        #neg_log_prob = tf.nn.weighted_cross_entropy_with_logits(targets=y, logits=x, pos_weight=advantage)

        #loss = neg_log_prob

        #print("\n\n Logits: \n\n", x, "\n\n Labels: \n\n", y, "\n\nAdvantages: \n\n", advantage, "\n\nneg_log_prob: \n\n", neg_log_prob, "\n\nloss: \n\n", loss)

        return reinforce_loss

    @staticmethod
    def entropy_exploration_loss(x):
        softmax_logits = tf.nn.softmax(x)
        neg_log_policy = - tf.log(tf.clip_by_value(softmax_logits, 1e-7, 1)) # probability of softmax can't be more then 1
        loss = tf.reduce_sum(tf.reduce_sum(softmax_logits * neg_log_policy, axis=1))
        #entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=softmax_logits, logits=x)
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