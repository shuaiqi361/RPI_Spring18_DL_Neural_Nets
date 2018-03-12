import tensorflow as tf
import numpy as np


hm_class = 10
batch_size = 50


def compute_ms_loss(all_weights, output, label, lambda_reg):

    # Mean Squared Loss and L2 regularization
    loss = 0.5 * tf.reduce_sum((output - label) ** 2) + 0.5 * lambda_reg * (tf.reduce_sum(all_weights['w1'] ** 2)
                                                                            + tf.reduce_sum(
        all_weights['w2'] ** 2) + tf.reduce_sum(all_weights['w3'] ** 2))

    return loss


def compute_softmax(w3, last_hid_output):

    # avoid overflow
    # b = tf.reduce_max(tf.matmul(w3, last_hid_output), axis=1, keepdims=True)
    exp_term = tf.exp(tf.matmul(w3, last_hid_output))

    prob = exp_term / tf.reduce_sum(exp_term, axis=0, keepdims=True)
    max_idx = tf.argmax(prob, axis=0)

    return max_idx, prob


def softmax_grads_eq(w3, last_hid_output):
    _, prob = compute_softmax(w3, last_hid_output)
    return prob * (1 - prob)


def relu_activation(inputs):
    return (1 + tf.sign(inputs)) / 2 * inputs


def relu_gradient(outputs):
    return (1 + tf.sign(outputs)) / 2


def softmax_gradient(output, label, w3, last_hid_output):
    # delta y and softmax_grads
    y_grads = output - label
    softmax_eq_grads = softmax_grads_eq(w3, last_hid_output)
    _, softmax_neq_grads = compute_softmax(w3, last_hid_output)

    # compute the entries of the off-diagonal
    output_grads = tf.matmul(-softmax_neq_grads, tf.transpose(softmax_neq_grads))
    # print('output_grad1', output_grads.get_shape().as_list())

    # mask all elements on the diagonal to 0 and add the true diagonal value
    mask = np.abs(np.eye(hm_class, dtype=np.float32) - 1)
    output_grads = output_grads * mask + softmax_eq_grads * np.ones((hm_class, hm_class), dtype=np.float32) \
                                         * np.eye((hm_class), dtype=np.float32)

    y_grads = tf.transpose(np.ones((hm_class, hm_class)) * y_grads)
    output_grads *= y_grads

    return tf.reduce_sum(output_grads, axis=1, keepdims=True)


def compute_accuracy(data, label, all_weights, lambda_reg, hm_data):
    # hm_data, hm_class = label.get_shape().as_list()

    s1 = tf.matmul(data, tf.transpose(all_weights['w1']))
    x1 = relu_activation(s1)
    x1 = tf.concat([x1, tf.ones((hm_data, 1))], 1)

    s2 = tf.matmul(x1, tf.transpose(all_weights['w2']))
    x2 = relu_activation(s2)
    x2 = tf.concat([x2, tf.ones((hm_data, 1))], 1)

    s3 = tf.matmul(x2, tf.transpose(all_weights['w3']))
    b = tf.reduce_max(s3, axis=1, keepdims=True)
    exp_term = tf.exp(s3 - b)

    prob = exp_term / tf.reduce_sum(exp_term, axis=1, keepdims=True)
    correct = tf.equal(tf.argmax(prob, axis=1), tf.argmax(label, axis=1))
    acc = tf.reduce_mean(tf.cast(correct, 'float'))
    loss = compute_ms_loss(all_weights, prob, label, lambda_reg)

    return acc, loss



def forward_back_props(all_weights, data, label, lambda_reg, hm_data):
    # hm_data, hm_class = label.get_shape().as_list()

    all_weight_grads = {'w1': all_weights['w1'] * 0, 'w2': all_weights['w2'] * 0, 'w3': all_weights['w3'] * 0}
    for j in range(hm_data):
        # forward propagation
        hid1_input = tf.matmul(all_weights['w1'], data[j, :][:, None])
        hid1_output = relu_activation(hid1_input)
        hid1_output = tf.concat([hid1_output, tf.ones([1, 1], tf.float32)], 0)

        hid2_input = tf.matmul(all_weights['w2'], hid1_output)
        hid2_output = relu_activation(hid2_input)
        hid2_output = tf.concat([hid2_output, tf.ones([1, 1], tf.float32)], 0)

        _, output = compute_softmax(all_weights['w3'], hid2_output)

        # backward propagation
        delta3 = softmax_gradient(output, label[j, :][:, None], all_weights['w3'], hid2_output)
        all_weight_grads['w3'] = tf.add(all_weight_grads['w3'], tf.matmul(delta3, tf.transpose(hid2_output)))

        temp_grad2 = tf.matmul(tf.transpose(all_weights['w3']), delta3)
        delta2 = temp_grad2[0:-1] * relu_gradient(hid2_input)
        all_weight_grads['w2'] = tf.add(all_weight_grads['w2'], tf.matmul(delta2, tf.transpose(hid1_output)))

        temp_grad1 = tf.matmul(tf.transpose(all_weights['w2']), delta2)
        delta1 = temp_grad1[0:-1] * relu_gradient(hid1_input)
        all_weight_grads['w1'] = tf.add(all_weight_grads['w1'], tf.matmul(delta1, tf.transpose(data[j, :][:, None])))

    return all_weight_grads
