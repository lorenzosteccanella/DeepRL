import tensorflow as tf


class Losses:

    @staticmethod
    def mse_loss(x, y):
        predictions = tf.cast(x, dtype=tf.float32)
        labels = tf.cast(y, dtype=tf.float32)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        losses = tf.math.squared_difference(predictions, labels)
        loss = tf.reduce_mean(losses)
        return loss

    @staticmethod
    def mse_loss_imp_w(x, y, imp_w):
        predictions = tf.cast(x, dtype=tf.float32)
        labels = tf.cast(y, dtype=tf.float32)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        losses = tf.math.squared_difference(predictions, labels)
        loss = tf.losses.compute_weighted_loss(losses, imp_w)
        return loss

    @staticmethod
    def ppo_loss(logits, old_logits, one_hot_a, advantage, e_clip=0.2):
        softmax_logits = tf.nn.softmax(logits)
        old_softmax_logits = tf.nn.softmax(old_logits)
        log_policy = tf.math.log(tf.clip_by_value(softmax_logits, 1e-7, (1-1e-7)))
        old_log_policy = tf.math.log(tf.clip_by_value(old_softmax_logits, 1e-7, (1-1e-7)))
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
        neg_log_policy = - tf.math.log(tf.clip_by_value(softmax_logits, 1e-7, (1-1e-7))) # need to clip here I don't know what are min and max value of logits
        tensor_one_hot = tf.convert_to_tensor(one_hot_a, dtype=tf.float32)
        tensor_advantage = tf.convert_to_tensor(advantage, dtype=tf.float32)
        reinforce_loss = tf.reduce_mean(tf.reduce_sum(neg_log_policy * tensor_one_hot, axis=1) * tf.stop_gradient(tensor_advantage))
        return reinforce_loss

    @staticmethod
    def reinforce_loss_imp_w(logits, one_hot_a, advantage, imp_w):
        softmax_logits = tf.nn.softmax(logits)
        neg_log_policy = - tf.math.log(tf.clip_by_value(softmax_logits, 1e-7, 1))
        tensor_one_hot = tf.convert_to_tensor(one_hot_a, dtype=tf.float32)
        tensor_advantage = tf.convert_to_tensor(advantage, dtype=tf.float32)
        reinforce_losses = tf.reduce_sum(neg_log_policy * tensor_one_hot, axis=1) * tf.stop_gradient(tensor_advantage)
        imp_w = tf.stop_gradient(tf.squeeze(imp_w))
        reinforce_losses = tf.compat.v1.losses.compute_weighted_loss(reinforce_losses, imp_w, reduction=tf.compat.v1.losses.Reduction.NONE)
        reinforce_loss = tf.reduce_mean(reinforce_losses)

        return reinforce_loss

    @staticmethod
    def entropy_exploration_loss(x):
        softmax_logits = tf.nn.softmax(x)
        neg_log_policy = - tf.math.log(tf.clip_by_value(softmax_logits, 1e-7, (1-1e-7))) # need to clip here I don't know what are min and max value of logits
        loss = tf.reduce_mean(tf.reduce_sum(softmax_logits * neg_log_policy, axis=1))
        return loss

    @staticmethod
    def mse_loss_self_imitation_learning(x, y):
        predictions = tf.cast(x, dtype=tf.float32)
        labels = tf.cast(y, dtype=tf.float32)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        difference = labels - predictions
        difference = tf.clip_by_value(difference, 0, tf.float32.max)
        losses = difference ** 2
        loss = tf.reduce_mean(losses)
        return loss

    @staticmethod
    def mse_loss_self_imitation_learning_imp_w(x, y, imp_w):
        predictions = tf.cast(x, dtype=tf.float32)
        labels = tf.cast(y, dtype=tf.float32)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        difference = labels - predictions
        difference = tf.clip_by_value(difference, 0, tf.float32.max)
        losses = difference ** 2
        losses = tf.compat.v1.losses.compute_weighted_loss(losses, imp_w, reduction=tf.compat.v1.losses.Reduction.NONE)
        loss = tf.reduce_mean(losses)
        return loss
