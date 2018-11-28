import tensorflow as tf
import numpy as np
from unit.conv import conv2d
from unit.BN import *
from unit.NN import nn_layer
from unit.pool import *


# the following is to build a CNN graph of TF to train
class TrainGraph:
    def __init__(self, input_shape=[64, 64, 1]):
        self.graph = tf.Graph()
        self.feed = []
        self.cores = []
        self.weight = []
        self.bias = []
        self.bn = []
        self.pool = []
        self.train = None
        self.INPUT_SHAPE = input_shape
        self.para = []

    def build_graph(self, NUM):
        with self.graph.as_default():
            # INPUT
            INPUT_SHAPE = self.INPUT_SHAPE
            with tf.name_scope('ImagesLabels'):
                images = tf.placeholder(tf.float32, shape=[None, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]])
                labels_r = tf.placeholder(tf.float32, shape=[None, 2])
                labels = labels_r
            learning_rate = tf.placeholder(tf.float32, name='LR')
            keep = tf.placeholder(tf.float32, name='keep')
            is_training = tf.placeholder(tf.bool, name='BN_controller')

            conf1 = {'is_training': is_training}
            conf2 = {'filterSize': None, 'outputChn': None, 'strideSize':[1]}
            with tf.name_scope('normalize'):
                x = BN(images, conf1)
            # layer1
            with tf.name_scope('conv1'):
                conf2['filterSize'] = [3]
                conf2['outputChn'] = 32
                x = conv2d(x, conf2)
                x = BN(x, conf1)
                x = tf.nn.relu(x)
            # layer2
            with tf.name_scope('conv2'):
                conf2['filterSize'] = [3]
                conf2['outputChn'] = 32
                x = conv2d(x, conf2)
                x = BN(x, conf1)
                x = tf.nn.relu(x)
            x, index = max_pool_argmax(x, {'name': 'pool1'})
            self.pool.append(index)

            with tf.name_scope('conv3'):
                conf2['filterSize'] = [3]
                conf2['outputChn'] = 64
                x = conv2d(x, conf2)
                x = BN(x, conf1)
                x = tf.nn.relu(x)
            x, index = max_pool_argmax(x, {'name': 'pool2'})
            self.pool.append(index)

            with tf.name_scope('conv4'):
                conf2['filterSize'] = [3]
                conf2['outputChn'] = 128
                x = conv2d(x, conf2)
                x = BN(x, conf1)
                x = tf.nn.relu(x)
            x, index = max_pool_argmax(x, {'name': 'pool3'})
            self.pool.append(index)

            with tf.name_scope('conv5'):
                conf2['filterSize'] = [3]
                conf2['outputChn'] = 256
                x = conv2d(x, conf2)
                x = BN(x, conf1)
                x = tf.nn.relu(x)
            x, index = max_pool_argmax(x, {'name': 'pool4'})
            self.pool.append(index)
            x = tf.nn.dropout(x, noise_shape=[1, 1, 1, 256], keep_prob=keep)

            self.cores = tf.get_collection('conv_core')


            with tf.name_scope('flatten'):
                shape = x.shape.as_list()
                num = shape[1] * shape[2] * shape[3]
                x = tf.reshape(x, [-1, num], name='flat')

            x = nn_layer(x, {'num':1024, 'name':'NN1', 'is_training': is_training}, keep_prob=keep)
            x = nn_layer(x, {'num':1024, 'name':'NN2', 'is_training': is_training}, keep_prob=keep)
            x = nn_layer(x, {'num':2,    'name':'NN3', 'is_training': is_training})

            for i in range(len(tf.get_collection('shift'))):
                self.bn.append([tf.get_collection('shift')[i], tf.get_collection('scale')[i], tf.get_collection('moving_mean')[i], tf.get_collection('moving_var')[i]])
            for i in range(len(tf.get_collection('bias'))):
                self.bias.append(tf.get_collection('bias')[i])
            for i in range(len(tf.get_collection('weight'))):
                self.weight.append(tf.get_collection('weight')[i])

            result = tf.argmax(tf.nn.softmax(x), axis=1)
            with tf.name_scope('Loss'):
                loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=x)
            train = tf.train.AdamOptimizer(learning_rate).minimize(tf.reduce_mean(loss))

            with tf.name_scope('summary'):
                tf.summary.scalar('loss', tf.reduce_mean(loss))
                tf.summary.image('INPUT', tf.transpose(images, [0, 2, 1, 3]))
                # tf.summary.image('INPUT_Label', tf.transpose(labels, [0, 2, 1, 3]))
                tf.summary.histogram('conv1', self.cores[0])
                tf.summary.histogram('conv2', self.cores[1])
                tf.summary.histogram('conv3', self.cores[2])
                tf.summary.histogram('conv4', self.cores[3])
                tf.summary.histogram('conv5', self.cores[4])
                tf.summary.histogram('result', result)

                summary = tf.summary.merge_all()
            self.feed = [images, labels_r, learning_rate, is_training, keep]
            self.train = train
            self.para = [loss, result, summary]

'''
the following graph is build a deconvNet of CNN or other conv network
'''
class DeconvGraph:
    def __init__(self, input_shape, para_pwd=None):
        self.graph = tf.Graph()
        self.feed = []
        self.cores = []
        self.bn = []
        self.pool = []
        self.train = None
        self.INPUT_SHAPE = input_shape
        self.para = []
        self.is_training = None
        self.conv_result = []

        # this item will decide which conv layer will be the last one in deconvNet
        self.deconv_item=None

        if para_pwd is not None:
            pass

    def build_graph(self, NUM=1):
        if self.cores==[] or self.bn==[]:
            raise ValueError('NO INIT VALUE')

        with self.graph.as_default():
            INPUT_SHAPE = self.INPUT_SHAPE
            h = INPUT_SHAPE[0]
            w = INPUT_SHAPE[1]
            with tf.name_scope('ImagesLabels'):
                images = tf.placeholder(tf.float32, shape=[NUM, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]])
                labels_r = tf.placeholder(tf.float32, shape=[NUM, 2])
                labels = labels_r
            learning_rate = tf.placeholder(tf.float32, name='LR')
            keep = tf.placeholder(tf.float32, name='keep')
            is_training = tf.placeholder(tf.bool, name='BN_controller')

            self.is_training = is_training
            with tf.name_scope('normalize'):
                x = self.BN(images, 0)
            with tf.name_scope('conv1'):
                a = tf.constant(self.cores[0].astype('float32'))
                x = tf.nn.conv2d(x, a, strides=[1,1,1,1], padding='SAME')
                x = self.BN(x, 1)
                x = tf.nn.relu(x)
                self.conv_result.append(x)
            with tf.name_scope('conv2'):
                a = tf.constant(self.cores[1].astype('float32'))
                x = tf.nn.conv2d(x, a, strides=[1, 1, 1, 1], padding='SAME')
                x = self.BN(x, 2)
                x = tf.nn.relu(x)
                self.conv_result.append(x)
            # use max_argmax to unpool
            x, index = max_pool_argmax(x, {'name': 'pool1'})
            self.pool.append(index)

            with tf.name_scope('conv3'):
                a = tf.constant(self.cores[2].astype('float32'))
                x = tf.nn.conv2d(x, a, strides=[1, 1, 1, 1], padding='SAME')
                x = self.BN(x, 3)
                x = tf.nn.relu(x)
                self.conv_result.append(x)
            x, index = max_pool_argmax(x, {'name': 'pool2'})
            self.pool.append(index)

            with tf.name_scope('conv4'):
                a = tf.constant(self.cores[3].astype('float32'))
                x = tf.nn.conv2d(x, a, strides=[1, 1, 1, 1], padding='SAME')
                x = self.BN(x, 4)
                x = tf.nn.relu(x)
                self.conv_result.append(x)
            x, index = max_pool_argmax(x, {'name': 'pool3'})
            self.pool.append(index)

            with tf.name_scope('conv5'):
                a = tf.constant(self.cores[4].astype('float32'))
                x = tf.nn.conv2d(x, a, strides=[1, 1, 1, 1], padding='SAME')
                x = self.BN(x, 5)
                x = tf.nn.relu(x)
                self.conv_result.append(x)
            x, index = max_pool_argmax(x, {'name': 'pool4'})
            self.pool.append(index)

            x = max_unpool_2d(x, self.pool[-1], {'name': 'unpool4', 'output_shape': [h//2//2//2, w//2//2//2]})

            if self.deconv_item == 5:
                # the last layer of deconvNet is layer5
                x = self.conv_result[-1]
            with tf.name_scope('deconv5'):
                x = self.re_BN(x, 5)
                x = tf.nn.relu(x)
                a = tf.constant(np.transpose(self.cores[4].astype('float32'), [1, 0, 3, 2]))
                x = tf.nn.conv2d(x, a, strides=[1, 1, 1, 1], padding='SAME')
            x = max_unpool_2d(x, self.pool[-2], {'name': 'unpool3', 'output_shape': [h//2//2, w//2//2]})

            if self.deconv_item == 4:
                # the last layer of deconvNet is layer4
                x = self.conv_result[-2]
            with tf.name_scope('deconv4'):
                x = self.re_BN(x, 4)
                x = tf.nn.relu(x)
                a = tf.constant(np.transpose(self.cores[3].astype('float32'), [1, 0, 3, 2]))
                x = tf.nn.conv2d(x, a, strides=[1, 1, 1, 1], padding='SAME')
            x = max_unpool_2d(x, self.pool[-3], {'name': 'unpool2', 'output_shape': [h // 2, w // 2]})

            if self.deconv_item == 3:
                # the last layer of deconvNet is layer3
                x = self.conv_result[-3]
            with tf.name_scope('deconv3'):
                x = self.re_BN(x, 3)
                x = tf.nn.relu(x)
                a = tf.constant(np.transpose(self.cores[2].astype('float32'), [1, 0, 3, 2]))
                x = tf.nn.conv2d(x, a, strides=[1, 1, 1, 1], padding='SAME')
            x = max_unpool_2d(x,self.pool[-4], {'name': 'unpool1', 'output_shape': [h, w]})

            if self.deconv_item == 2:
                # the last layer of deconvNet is layer2
                x = self.conv_result[-4]
            with tf.name_scope('deconv2'):
                x = self.re_BN(x, 2)
                x = tf.nn.relu(x)
                a = tf.constant(np.transpose(self.cores[1].astype('float32'), [1, 0, 3, 2]))
                x = tf.nn.conv2d(x, a, strides=[1, 1, 1, 1], padding='SAME')

            if self.deconv_item == 1:
                # the last layer of deconvNet is layer1
                x = self.conv_result[-5]
            with tf.name_scope('deconv1'):
                x = self.re_BN(x, 1)
                x = tf.nn.relu(x)
                a = tf.constant(np.transpose(self.cores[1].astype('float32'), [1, 0, 3, 2]))
                x = tf.nn.conv2d(x, a, strides=[1, 1, 1, 1], padding='SAME')
                x = tf.nn.relu(x)

            print(x.shape)
            self.feed = [images, labels_r, learning_rate, is_training, keep]

            self.para = [x]

    def BN(self, x, index):
        with tf.name_scope('BN'):
            params_shape = x.shape.as_list()[-1]
            axis = list(range(len(x.shape.as_list()) - 1))  # 得到需要计算batch的部分，除了最后一个维度不进行
            print(index)
            # do BN use the stored shift and scale and mean and var
            shift = tf.Variable(self.bn[index][0, :].astype('float32'), name='beta', trainable=False)
            scale = tf.Variable(self.bn[index][1, :].astype('float32'), name='gamma', trainable=False)
            moving_mean = tf.Variable(self.bn[index][2, :], trainable=False, name='moving_mean')
            moving_variance = tf.Variable(self.bn[index][3, :], trainable=False, name='moving_mean')
            batch_mean, batch_var = tf.nn.moments(x, axis)

            mean, var = tf.cond(self.is_training,
                                lambda: (batch_mean, batch_var),
                                lambda: (moving_mean, moving_variance))

            tf.add_to_collection('moving_mean', moving_mean)
            tf.add_to_collection('moving_var', moving_variance)
            tf.add_to_collection('mean', mean)
            tf.add_to_collection('var', var)
            result = tf.nn.batch_normalization(x=x, mean=mean, variance=var, offset=shift, scale=scale,
                                               variance_epsilon=BN_EPSILON)
        return result

    def re_BN(self, x, index):
        # it is useless for DeconvNet, deconvNet can not do any re_BN in the process of deconv
        return x
        with tf.name_scope('re_BN'):
            params_shape = x.shape.as_list()[-1]
            axis = list(range(len(x.shape.as_list()) - 1))  # 得到需要计算batch的部分，除了最后一个维度不进行
            shift = tf.Variable(-1*self.bn[index][0, :].astype('float32'), name='beta')
            scale = tf.Variable(1/self.bn[index][1,:].astype('float32'), name='gamma')
            moving_mean = tf.Variable(self.bn[index][2, :], trainable=False, name='moving_mean')
            moving_variance = tf.Variable(self.bn[index][3,:], trainable=False, name='moving_mean')
            batch_mean, batch_var = tf.nn.moments(x, axis)
            # if a value has no connection with the loss, and you dnt fetches it, it will not change
            mean, var = tf.cond(self.is_training,
                                lambda: (batch_mean, batch_var),
                                lambda: (moving_mean, moving_variance))
            result = tf.nn.batch_normalization(x=x, mean=mean, variance=var, offset=shift, scale=scale,
                                               variance_epsilon=BN_EPSILON)
        return result


# the ReGraph is build to build a graph of reversion method:
# http://yosinski.com/deepvis
class ReGraph:
    def __init__(self, para_pwd=None):
        self.graph = tf.Graph()
        self.feed = []
        self.cores = []
        self.bn = []
        self.pool = []
        self.train = None
        self.INPUT_SHAPE = [64, 64, 1]
        self.para = []
        self.is_training = None
        self.conv_result = []
        self.deconv_item=None
        self.bias = []
        self.weight = []
        if para_pwd is not None:
            pass

    def build_graph(self):
        if self.cores==[] or self.bn==[] or self.bias==[] or self.weight==[]:
            raise ValueError('NO INIT VALUE')

        with self.graph.as_default():
            INPUT_SHAPE = self.INPUT_SHAPE
            h = INPUT_SHAPE[0]
            w = INPUT_SHAPE[1]
            with tf.name_scope('ImagesLabels'):
                # init = np.reshape(np.random.uniform(-1, 1, size=INPUT_SHAPE).astype('float32'), [1, h, w, 1])
                # images = tf.Variable(init, name='IMAGE')
                images = tf.placeholder(tf.float32, [1, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]])

            is_training = tf.placeholder(tf.bool, name='BN_controller')

            self.is_training = is_training
            with tf.name_scope('normalize'):
                x = self.BN(images, 0)
            with tf.name_scope('conv1'):
                a = tf.constant(self.cores[0].astype('float32'))
                x = tf.nn.conv2d(x, a, strides=[1,1,1,1], padding='SAME')
                x = self.BN(x, 1)
                x = tf.nn.relu(x)
                self.conv_result.append(x)
            with tf.name_scope('conv2'):
                a = tf.constant(self.cores[1].astype('float32'))
                x = tf.nn.conv2d(x, a, strides=[1, 1, 1, 1], padding='SAME')
                x = self.BN(x, 2)
                x = tf.nn.relu(x)
                self.conv_result.append(x)
            x, index = max_pool_argmax(x, {'name': 'pool1'})
            self.pool.append(index)

            with tf.name_scope('conv3'):
                a = tf.constant(self.cores[2].astype('float32'))
                x = tf.nn.conv2d(x, a, strides=[1, 1, 1, 1], padding='SAME')
                x = self.BN(x, 3)
                x = tf.nn.relu(x)
                self.conv_result.append(x)
            x, index = max_pool_argmax(x, {'name': 'pool2'})
            self.pool.append(index)

            with tf.name_scope('conv4'):
                a = tf.constant(self.cores[3].astype('float32'))
                x = tf.nn.conv2d(x, a, strides=[1, 1, 1, 1], padding='SAME')
                x = self.BN(x, 4)
                x = tf.nn.relu(x)
                self.conv_result.append(x)
            x, index = max_pool_argmax(x, {'name': 'pool3'})
            self.pool.append(index)

            with tf.name_scope('conv5'):
                a = tf.constant(self.cores[4].astype('float32'))
                x = tf.nn.conv2d(x, a, strides=[1, 1, 1, 1], padding='SAME')
                x = self.BN(x, 5)
                x = tf.nn.relu(x)
                self.conv_result.append(x)
            x, index = max_pool_argmax(x, {'name': 'pool4'})
            self.pool.append(index)

            with tf.name_scope('flatten'):
                shape = x.shape.as_list()
                num = shape[1] * shape[2]*shape[3]
                x = tf.reshape(x, [-1, num], name='flat')

            with tf.name_scope('NN1'):
                w = tf.constant(self.weight[0].astype('float32'))
                b = tf.constant(self.bias[0].astype('float32'))
                x = tf.nn.xw_plus_b(x=x, weights=w, biases=b)
                x = self.BN(x, 6)
                x = tf.nn.relu(x)
            with tf.name_scope('NN2'):
                w = tf.constant(self.weight[1].astype('float32'))
                b = tf.constant(self.bias[1].astype('float32'))
                x = tf.nn.xw_plus_b(x=x, weights=w, biases=b)
                x = self.BN(x, 7)
                x = tf.nn.relu(x)
            with tf.name_scope('NN3'):
                w = tf.constant(self.weight[2].astype('float32'))
                b = tf.constant(self.bias[2].astype('float32'))
                x = tf.nn.xw_plus_b(x=x, weights=w, biases=b)
                x = self.BN(x, 8)
                x = tf.nn.relu(x)

            x = tf.nn.softmax(x)
            # decide which neuron will be reversed
            label_f = tf.slice(x, [0, 0], [1, 1])
            print(x.shape)
            print(label_f.shape)
            # label_f = x
            # train = tf.train.GradientDescentOptimizer(learning_rate).minimize(-label_f)
            gradient = tf.gradients(ys=label_f, xs=images)
            self.feed = [images, is_training]
            self.para = [gradient, x]

    def BN(self, x, index):
        with tf.name_scope('BN'):
            params_shape = x.shape.as_list()[-1]
            axis = list(range(len(x.shape.as_list()) - 1))  # 得到需要计算batch的部分，除了最后一个维度不进行
            print(index)
            shift = tf.Variable(self.bn[index][0, :].astype('float32'), name='beta', trainable=False)
            scale = tf.Variable(self.bn[index][1, :].astype('float32'), name='gamma', trainable=False)
            moving_mean = tf.Variable(self.bn[index][2, :], trainable=False, name='moving_mean')
            moving_variance = tf.Variable(self.bn[index][3, :], trainable=False, name='moving_mean')
            batch_mean, batch_var = tf.nn.moments(x, axis)

            if index==-1:
                mean, var = [batch_mean, batch_var]
            else:
                mean, var = tf.cond(self.is_training,
                                    lambda: (batch_mean, batch_var),
                                    lambda: (moving_mean, moving_variance))

            tf.add_to_collection('moving_mean', moving_mean)
            tf.add_to_collection('moving_var', moving_variance)
            tf.add_to_collection('mean', mean)
            tf.add_to_collection('var', var)
            result = tf.nn.batch_normalization(x=x, mean=mean, variance=var, offset=shift, scale=scale,
                                               variance_epsilon=BN_EPSILON)
        return result

# build the Graph of conv layer
class ReGraph_conv:
    def __init__(self, para_pwd=None):
        self.graph = tf.Graph()
        self.feed = []
        self.cores = []
        self.bn = []
        self.pool = []
        self.train = None
        self.para = []
        self.is_training = None
        self.conv_result = []
        self.deconv_item=None
        self.bias = []
        self.weight = []
        self.shape = {1:[3,3], 2:[5,5], 3:[10,10], 4:[20,20],5:[40,40]}
        self.chn = {1:32, 2:32, 3:64, 4:128,5:256}
        if para_pwd is not None:
            pass

    def build_graph(self, layer):

        # layer means the wanted layer to inversion
        if self.cores==[] or self.bn==[] or self.bias==[] or self.weight==[]:
            raise ValueError('NO INIT VALUE')

        with self.graph.as_default():
            # decide the shape of input for different conv layer
            shape=self.shape
            # decide the chn of different input
            chn = self.chn
            h, w = shape[layer]
            with tf.name_scope('ImagesLabels'):
                # init = np.reshape(np.random.uniform(-1, 1, size=INPUT_SHAPE).astype('float32'), [1, h, w, 1])
                # images = tf.Variable(init, name='IMAGE')
                images = tf.placeholder(tf.float32, [1, h, w, 1])
            is_training = tf.placeholder(tf.bool, name='BN_controller')

            self.is_training = is_training
            while 1:
                with tf.name_scope('normalize'):
                    x = self.BN(images, 0)
                with tf.name_scope('conv1'):
                    a = tf.constant(self.cores[0].astype('float32'))
                    x = tf.nn.conv2d(x, a, strides=[1,1,1,1], padding='VALID')
                    x = self.BN(x, 1)
                    if layer == 1:
                        break
                    x = tf.nn.relu(x)
                    self.conv_result.append(x)

                with tf.name_scope('conv2'):
                    a = tf.constant(self.cores[1].astype('float32'))
                    x = tf.nn.conv2d(x, a, strides=[1, 1, 1, 1], padding='VALID')
                    x = self.BN(x, 2)
                    if layer == 2:
                        break
                    x = tf.nn.relu(x)
                    self.conv_result.append(x)
                x, indices = max_pool_argmax(x, {'name': 'pool1'})
                self.pool.append(indices)

                with tf.name_scope('conv3'):
                    a = tf.constant(self.cores[2].astype('float32'))
                    x = tf.nn.conv2d(x, a, strides=[1, 1, 1, 1], padding='VALID')
                    x = self.BN(x, 3)
                    if layer == 3:
                        break
                    x = tf.nn.relu(x)
                    self.conv_result.append(x)

                x, indices = max_pool_argmax(x, {'name': 'pool2'})
                self.pool.append(indices)

                with tf.name_scope('conv4'):
                    a = tf.constant(self.cores[3].astype('float32'))
                    x = tf.nn.conv2d(x, a, strides=[1, 1, 1, 1], padding='VALID')
                    x = self.BN(x, 4)
                    if layer == 4:
                        break
                    x = tf.nn.relu(x)
                    self.conv_result.append(x)

                x, indices = max_pool_argmax(x, {'name': 'pool3'})
                self.pool.append(indices)

                with tf.name_scope('conv5'):
                    a = tf.constant(self.cores[4].astype('float32'))
                    x = tf.nn.conv2d(x, a, strides=[1, 1, 1, 1], padding='VALID')
                    x = self.BN(x, 5)
                    if layer == 5:
                        break
                    x = tf.nn.relu(x)
                    self.conv_result.append(x)

            x = tf.reshape(x, [1, chn[layer]])
            print(x.shape)
            gradient = []
            for i in range(chn[layer]):
                label_f = tf.slice(x, [0, i], [1, 1])
                gradient_i = tf.gradients(ys=label_f, xs=images)
                gradient.append(gradient_i)
                print(i)
            # label_f = x
            # train = tf.train.GradientDescentOptimizer(learning_rate).minimize(-label_f)

            print(len(gradient))
            self.feed = [images, is_training]
            self.para = [gradient, x]

    def BN(self, x, index):
        with tf.name_scope('BN'):
            params_shape = x.shape.as_list()[-1]
            # make the stored para to BN
            axis = list(range(len(x.shape.as_list()) - 1))  # 得到需要计算batch的部分，除了最后一个维度不进行
            print(index)
            shift = tf.Variable(self.bn[index][0, :].astype('float32'), name='beta', trainable=False)
            scale = tf.Variable(self.bn[index][1, :].astype('float32'), name='gamma', trainable=False)
            moving_mean = tf.Variable(self.bn[index][2, :], trainable=False, name='moving_mean')
            moving_variance = tf.Variable(self.bn[index][3, :], trainable=False, name='moving_mean')
            batch_mean, batch_var = tf.nn.moments(x, axis)

            if index==-1:
                mean, var = [batch_mean, batch_var]
            else:
                mean, var = tf.cond(self.is_training,
                                    lambda: (batch_mean, batch_var),
                                    lambda: (moving_mean, moving_variance))

            tf.add_to_collection('moving_mean', moving_mean)
            tf.add_to_collection('moving_var', moving_variance)
            tf.add_to_collection('mean', mean)
            tf.add_to_collection('var', var)
            result = tf.nn.batch_normalization(x=x, mean=mean, variance=var, offset=shift, scale=scale,
                                               variance_epsilon=BN_EPSILON)
        return result










