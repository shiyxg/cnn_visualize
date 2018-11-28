import tensorflow as tf
import numpy as np
from unit.BN import BN

def nn_layer(input, conf, act = tf.nn.relu, keep_prob=None):
    '''
    :param input: the input para, need shape:[batch, length]
    :param conf: about this layer's neuron number
    :return: result after a act(w*x+b) action
    建立一个完整的NN层
    '''
    input_length = input.shape.as_list()[1]
    output_length = conf['num']
    weight_size = [input_length, output_length]

    # # Use Glorot and Bengio(2010)'s init method
    stddev = np.sqrt(2) / np.sqrt(input_length + output_length)

    weight_init = np.random.uniform(low=-np.sqrt(3)*stddev,
                                    high=np.sqrt(3)*stddev,
                                    size=weight_size).astype('float32')

    with tf.name_scope(conf['name']):
        w = tf.Variable(weight_init, name='weight')
        b = tf.Variable(np.zeros(output_length).astype('float32'), name='bias')
        result = input
        result = tf.nn.xw_plus_b(x=result, weights=w, biases=b)
        tf.add_to_collection('wx_result', result)
        result = BN(result, conf)
        if keep_prob is not None:
            result = tf.nn.dropout(result, keep_prob=keep_prob)
        tf.add_to_collection('BN_result', result)
        tf.add_to_collection('weight', w)
        tf.add_to_collection('bias', b)

        result = act(result, name='act')
    return result


conf_sample = {
    'name': 'NNL',
    'num': 1024
}