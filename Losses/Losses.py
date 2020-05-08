import tensorflow as tf


class Losses:

    @staticmethod
    def mse_loss(x, y):
        predictions = tf.to_float(x)
        labels = tf.to_float(y)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        losses = tf.math.squared_difference(predictions, labels)
        loss = tf.reduce_mean(losses)
        return loss

    @staticmethod
    def ppo_loss(logits, old_logits, one_hot_a, advantage, e_clip=0.2):
        softmax_logits = tf.nn.softmax(logits)
        old_softmax_logits = tf.nn.softmax(old_logits)
        log_policy = tf.log(tf.clip_by_value(softmax_logits, 1e-7, (1-1e-7)))
        old_log_policy = tf.log(tf.clip_by_value(old_softmax_logits, 1e-7, (1-1e-7)))
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
        neg_log_policy = - tf.log(tf.clip_by_value(softmax_logits, 1e-7, (1-1e-7))) # need to clip here I don't know what are min and max value of logits
        tensor_one_hot = tf.convert_to_tensor(one_hot_a, dtype=tf.float32)
        tensor_advantage = tf.convert_to_tensor(advantage, dtype=tf.float32)
        reinforce_loss = tf.reduce_mean(tf.reduce_sum(neg_log_policy * tensor_one_hot, axis=1) * tf.stop_gradient(tensor_advantage))
        return reinforce_loss

    @staticmethod
    def entropy_exploration_loss(x):
        softmax_logits = tf.nn.softmax(x)
        neg_log_policy = - tf.log(tf.clip_by_value(softmax_logits, 1e-7, (1-1e-7))) # need to clip here I don't know what are min and max value of logits
        loss = tf.reduce_mean(tf.reduce_sum(softmax_logits * neg_log_policy, axis=1))
        return loss