# -*- coding: utf-8 -*-

from __future__ import print_function

# Graph is built in Graph.py
import tensorflow as tf
import numpy as np
import nibabel as nb
import sys
import scipy.io as io
import os
import matplotlib as mpl
import matplotlib.cm as cm

# mpl.use('agg')
import matplotlib.pyplot as pyplot

# path = 'C:\\Users\\shi\\oneDrive\\FaultsDetection\\fault-test'
# path = '/gpfs/share/home/1400012437/FaultDetection/fault-test'
# sys.path.append(path)

from nii import *
from graph import *
from little_path import *


## libs
# change the learning rate with accuracy & loss & i
def learn_rate(accuracy, loss, i):
    LR_i = 1e-3
    if i < 1e5:
        LR_i = 1e-3
    else:
        LR_i = 1e-4
    return LR_i


def visual(data, sess, name, shape, graph):
    # it costs a long time to do
    [XL, YL, TL] = data.shape
    expand = data.expand
    layerIndex = [YL // 2]

    labelSample = np.zeros([len(layerIndex), XL, TL])
    labelOri = np.zeros([len(layerIndex), XL, TL])
    dataOri = np.zeros([len(layerIndex), XL, TL])
    g = graph
    for i in range(len(layerIndex)):
        j = 64
        k = 0

        [a, b] = data.pickLayer(layerIndex[i], sampleAxis=13)
        dataOri[i, :, :] = a
        labelOri[i, :, :] = b

        while j <= 128:
            index = np.array([j, layerIndex[i], k])
            batch = data.pick(index, shape=shape, sampleAxis=13, IMAGE=True)
            label_re = np.zeros([2,1])
            # print(shape)
            label_re[0, 0] = batch[1][shape[0]//2, shape[1]//2]==0
            label_re[1, 0] = batch[1][shape[0]//2, shape[1]//2]==1

            feedData = {g.feed[0]: np.reshape(batch[0], [1, shape[0], shape[1], 1]),
                        g.feed[1]: np.reshape(label_re, [1, 2]).astype('float32'),
                        g.feed[2]: 1e-3,
                        g.feed[3]: False,
                        g.feed[4]:1
                        }
            fetch = [g.para[1],tf.get_collection('BN_result')]
            result_ijk, bn = sess.run(fetch, feed_dict=feedData)
            labels_ijk = result_ijk
            # print(bn[-1])
            # if result_ijk[0]==1:
            #     print(j)
            #     print(result_ijk)
            #     print(labels_ijk)
            labelSample[i, (j - 0):(j + 1), (k - 0):(k + 1)] = labels_ijk

            k = k + 1
            j = k // TL * 1 + j
            k = k % TL


        train = np.reshape(dataOri[i, :, :] + 3 * labelSample[i, :, :], [YL, TL])
        ori = np.reshape(dataOri[i, :, :] + 3 * labelOri[i, :, :], [YL, TL])
        label = np.reshape(labelSample[i, :, :], [YL, TL])
        pyplot.imshow(train.T)
        pyplot.savefig(name + 'all.png')
        pyplot.imshow(label.T)
        pyplot.colorbar()
        pyplot.savefig(name + 'label.png')
        pyplot.close('all')

def result_visual(dataSets, sess, pwd, step, shape, graph):
    for data in [dataSets.train[0]]:
        name = pwd + '/train/' + data.model + '_%g_' % step
        visual(data, sess, name, shape, graph)

    # for data in dataSets.test:
    #     name = pwd + '/test/' + data.model + '_%g_' % step
    #     visual(data, sess, name, shape, graph)


def saveSamples(i, batch, pwd, shape):
    [record, label] = batch
    assert len(record) == len(label)

    for j in range(len(record)):
        a = np.reshape(record[j, :, :], shape)
        b = np.reshape(label[j, :, :], shape)
        # figure = a.T + b.T*10
        pyplot.pcolor(a.T)
        pyplot.savefig(pwd + '/sample/%g%g_train_record.png' % (i, j))

        pyplot.pcolor(a.T + b.T * 128 * 3)
        pyplot.savefig(pwd + '/sample/%g%g_train_all.png' % (i, j))


def saveSummary(step, data, sess, pwd, add, shape, trainWrite, testWrite, g):

    trainBatch = data.trainBatch(100, addProb=add, shape=shape)
    testBatch = data.testBatch(100, addProb=add, shape=shape)
    #self.feed = [images, labels_r, learning_rate, is_training, keep]
    feedData_train = {g.feed[0]: trainBatch[0], g.feed[1]: trainBatch[1],
                g.feed[2]: 1e-3, g.feed[3]: False, g.feed[4]: 1}
    feedData_test = {g.feed[0]: testBatch[0], g.feed[1]: testBatch[1],
                      g.feed[2]: 1e-3, g.feed[3]: False, g.feed[4]: 1}
    fetchVariables = [g.para[0], g.para[1], g.para[2]] # loss, result, summary

    [loss_train, result_train, summary_train
     ] = sess.run(fetches=fetchVariables, feed_dict=feedData_train)
    [loss_test, result_test, summary_test
     ] = sess.run(fetches=fetchVariables, feed_dict=feedData_test)

    trainWrite.add_summary(summary_train, step)
    testWrite.add_summary(summary_test, step)
    trainWrite.flush()
    testWrite.flush()
    print('******************************************')
    print(step)
    print([loss_train.sum(), loss_test.sum()])

## train & test
def train(trainPath, testPath, validPath):
    ## build a graph
    LR_i = 1e-3
    batch_i = 20
    add = 0.3
    keep = 0.5
    shape = [64, 64]

    g = TrainGraph(input_shape=[shape[0],shape[1], 1])
    g.build_graph(batch_i)

    ## data

    data = niiS(trainPath, testPath)
    ## train
    with tf.Session(graph=g.graph) as sess:
        pwd = 'C:/Users/shiyx/Documents/ML_log/CNN_visual/02'
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        # saver.restore(sess,'/home/shi/FaultDetection/fault-test/Session')

        trainWrite = tf.summary.FileWriter(pwd + '/train', tf.get_default_graph())
        testWrite = tf.summary.FileWriter(pwd + '/test', tf.get_default_graph())
        ops = tf.get_collection('ops')
        for i in range(50000):
            trainBatch = data.trainBatch(batch_i, addProb=add, shape=shape)

            # output the samples
            if i == 0:
                # saveSamples(i,trainBatch,pwd,shape)
                print(1)
            # train
            feedData = {g.feed[0]: trainBatch[0], g.feed[1]: trainBatch[1],
                        g.feed[2]: LR_i, g.feed[3]: True, g.feed[4]:keep}

            fetchVariables = [g.train, ops, g.para[0], g.para[1]]


            [_, _, loss_i, result] = sess.run(fetches=fetchVariables, feed_dict=feedData)
            # print(np.sum(loss_i ))
            if i % 300 == 0:
                print(trainBatch[1].argmax(axis=1))
                print(result)
                saveSummary(i, data, sess, pwd, add, shape, trainWrite, testWrite, g=g)
            if i % 10000 == 0 and i!=0:
                result_visual(data,sess,pwd,i,shape, graph = g)
                print(1)
            # if i % 250000 == 0 and i != 0:
            #     # saver.save(sess,pwd+'/para_%g/'%i)
            #     print(1)
        # result_visual(data, sess, pwd, i, shape, graph=g)
        saver.save(sess, pwd + '/para/')
        trainWrite.close()
        testWrite.close()
    return sess


def save_para():
    ## save para of TF to npy
    LR_i = 1e-3
    batch_i = 20
    add = 0.3
    keep = 0.5
    shape = [64, 64]

    g = TrainGraph(input_shape=[shape[0],shape[1], 1])
    g.build_graph(batch_i)
    print('built Graph')
    with tf.Session(graph=g.graph) as sess:
        pwd = 'C:/Users/shiyx/Documents/ML_log/CNN_visual/test01'
        # the path to store npy file:pwd+/para_npy
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess,'C:/Users/shiyx/Documents/ML_log/CNN_visual/test01/para/')

        feedData = {g.feed[0]: np.zeros([1,shape[0],shape[1], 1]), g.feed[1]: np.zeros([1,2]),
                    g.feed[2]: LR_i, g.feed[3]: True, g.feed[4]:1}

        fetchVariables = [g.cores, g.bn, g.bias, g.weight]
        [cores, bn,  bias, weight] = sess.run(fetches=fetchVariables, feed_dict=feedData)
        print('restored para')
        for i in range(len(cores)):
            np.save(pwd+'/para_npy/cores_%s'%i, cores[i])
        print('stored cores as npy')
        for i in range(len(bn)):
            np.save(pwd+'/para_npy/bn_%s'%i, np.array(bn[i]))
        print('stored bn as npy')
        for i in range(len(weight)):
            np.save(pwd+'/para_npy/weight_%s'%i, np.array(weight[i]))
        print('stored weight as npy')
        for i in range(len(bias)):
            np.save(pwd+'/para_npy/bias_%s'%i, np.array(bias[i]))
        print('stored bias as npy')
    return sess
# a = train(trainPath, testPath, validPath)
save_para()

