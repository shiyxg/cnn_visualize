import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from scipy.signal import convolve2d
import os
from graph import *
from nii import *

trainPath=['C:\\Users\\shiyx\\Documents\\Data\\SYN\\FCN1_OA3_f10_n10\\Nifti']
testPath=trainPath
validPath=trainPath


def deconvnet():
    ## build a graph
    LR_i = 1e-3
    batch_i = 1
    add = 0.5
    keep = 1
    shape = [128, 128]
    # para_path = 'C:/Users/shiyx/Documents/ML_log/CNN_visual/test01/para_npy/'
    para_path = 'C:/Users/shiyx/Documents/ML_log/CNN_visual/02/para_npy/'
    g = DeconvGraph(input_shape=[shape[0],shape[1], 1])
    for i in range(5):
        cores_i= np.load(para_path+'cores_%s.npy'%i)
        g.cores.append(cores_i)
    for i in range(6):
        bn_i= np.load(para_path+'bn_%s.npy'%i)
        g.bn.append(bn_i)

    g.deconv_item = 5

    g.build_graph()

    # data = niiS(trainPath, testPath, expand=64)
    ## train
    wave = np.load('C:\\Users\shiyx\Documents\ML_log\CNN_visual\\test01' + '/train/FCN1_OA3_f10_n10_250000_data.npy')
    with tf.Session(graph=g.graph) as sess:
        pwd = 'C:/Users/shiyx/Documents/ML_log/CNN_visual/02'
        sess.run(tf.global_variables_initializer())
        for i in range(50000):
            trainBatch = [wave[50:178, 200:328].reshape([1,128,128,1])]

            feedData = {g.feed[0]: trainBatch[0], g.feed[1]: np.zeros([1,2]).astype('float32'),
                        g.feed[2]: LR_i, g.feed[3]: True, g.feed[4]:keep}

            fetchVariables = [g.para[0]]

            [result] = sess.run(fetches=fetchVariables, feed_dict=feedData)

            plt.figure(1)
            # imshow(trainBatch[0][0, :, :, 0].T, cmap='seismic')
            # colorbar()
            subplot(111)
            imshow(result[0, :, :, 0].T, cmap = 'seismic')
            colorbar()
            plt.show()
            plt.close('all')

    return None


def reversion():
    LR_i = 1e-3
    batch_i = 1
    add = 0.01
    keep = 1
    kernel_3x3 = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ])
    kernel_5x5 = np.array([
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1]
    ])
    kernel_3x1 = np.array([1,2,1]).reshape([3,1])
    kernel_3x5 = np.dot(np.array([1,2,1]).reshape([3,1]),np.array([1,4,7,4,1]).reshape(1,5)).T
    kernel_3x2 = np.dot(np.array([1,2,1]).reshape([3,1]),np.array([1,1]).reshape(1,2))
    kernel = kernel_3x3
    kernel = kernel / kernel.sum()  # 加权平均
    para_path = 'C:/Users/shiyx/Documents/ML_log/CNN_visual/test01/para_npy/'
    # para_path = 'C:/Users/shiyx/Documents/ML_log/CNN_visual/02/para_npy/'
    g = ReGraph()

    for i in range(5):
        cores_i = np.load(para_path + 'cores_%s.npy' % i)
        g.cores.append(cores_i)
    for i in range(9):
        bn_i = np.load(para_path + 'bn_%s.npy' % i)
        g.bn.append(bn_i)
    for i in range(3):
        bias_i = np.load(para_path + 'bias_%s.npy' % i)
        weight_i = np.load(para_path + 'weight_%s.npy' % i)
        g.bias.append(bias_i)
        g.weight.append(weight_i)

    g.build_graph()
    ## train
    # data = niiS(trainPath, testPath)
    # images = data.trainBatch(1, 0.001, [64, 64])[0]
    with tf.Session(graph=g.graph) as sess:
        pwd = 'C:/Users/shiyx/Documents/ML_log/CNN_visual/test01/samples'
        wave = np.load('C:\\Users\shiyx\Documents\ML_log\CNN_visual\\test01\\train\FCN1_OA3_f10_n10_250000_data.npy')
        sess.run(tf.global_variables_initializer())
        images = np.random.uniform(-3, 3, size=[1, 64, 64, 1]).astype('float32')
        # images = np.reshape(wave[32:96, 32:96].astype('float32'), [1,64,64,1])
        images = np.load(pwd+'/fault_1.npy')

        np.save(pwd+'/test.npy',images)
        init = images
        r = []
        for i in range(10000):

            feedData = {g.feed[0]:images, g.feed[1]: False}
            fetchVariables = g.para

            [gradient, l] = sess.run(fetches=fetchVariables, feed_dict=feedData)
            gradient = gradient[0]
            if i%200==0:
                print(i)
                print(l[0][0])
                if l[0][0]>=0.9:
                    # r.append(l[0][1])
                    # print(i)
                    # continue
                    print(l)
                    plt.figure(1,figsize=[10,3],dpi=200)
                    ax = subplot(131)
                    pcolormesh(init[0, :, :, 0].T, cmap='seismic')
                    ax.invert_yaxis()
                    colorbar()
                    title('init')
                    ax = subplot(132)
                    pcolormesh(images[0, :, :, 0].T, cmap='seismic')
                    ax.invert_yaxis()
                    title('result:%s'%l[0][1])
                    colorbar()
                    ax = subplot(133)
                    pcolormesh(gradient[0, :, :, 0].T, cmap='seismic')
                    ax.invert_yaxis()
                    title('grad')
                    colorbar()
                    # savefig('C:\\Users\shiyx\OneDrive\Python Scripts\CNN_visualize\\report\\noise\\%s.jpg'%l[0][1])
                    show()
                    close('all')
            # decay
            dis = np.abs(images * gradient)
            images = (images + 50*gradient)
            # cut small pixel
            # images[np.where(np.abs(images) <= images.max()/100)]=0
            # cut smal distribute
            images[np.where(dis <= dis.mean()/50)] = 0
            # guass
            images = convolve2d(images[0, :, :, 0], kernel, mode='same').reshape([1, 64, 64, 1])

    return None


def reversion_conv():
    # do reversion for conv layers
    LR_i = 1e-3
    batch_i = 1
    add = 0.01
    keep = 1
    kernel_3x3 = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ])
    kernel_5x5 = np.array([
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1]
    ])
    kernel_3x1 = np.array([1,2,1]).reshape([3,1])
    kernel_3x5 = np.dot(np.array([1,2,1]).reshape([3,1]),np.array([1,4,7,4,1]).reshape(1,5)).T
    kernel_3x2 = np.dot(np.array([1,2,1]).reshape([3,1]),np.array([1,1]).reshape(1,2))
    # decide which kernel to smooth
    kernel = kernel_3x3
    kernel = kernel / kernel.sum()  # 加权平均

    para_path = 'C:/Users/shiyx/Documents/ML_log/CNN_visual/test01/para_npy/'
    # para_path = 'C:/Users/shiyx/Documents/ML_log/CNN_visual/02/para_npy/'
    g = ReGraph_conv()
    shape = g.shape
    chn = g.chn

    # give the graph of stored npy file
    for i in range(5):
        cores_i = np.load(para_path + 'cores_%s.npy' % i)
        g.cores.append(cores_i)
    for i in range(9):
        bn_i = np.load(para_path + 'bn_%s.npy' % i)
        g.bn.append(bn_i)
    for i in range(3):
        bias_i = np.load(para_path + 'bias_%s.npy' % i)
        weight_i = np.load(para_path + 'weight_%s.npy' % i)
        g.bias.append(bias_i)
        g.weight.append(weight_i)
    index = 16
    layer = 3
    g.build_graph(layer=layer)

    ## train
    # data = niiS(trainPath, testPath)
    # images = data.trainBatch(1, 0.001, [64, 64])[0]
    with tf.Session(graph=g.graph) as sess:
        pwd = 'C:/Users/shiyx/Documents/ML_log/CNN_visual/test01/samples'
        # wave = np.load('C:\\Users\shiyx\Documents\ML_log\CNN_visual\\test01\\train\FCN1_OA3_f10_n10_250000_data.npy')
        sess.run(tf.global_variables_initializer())
        # different init_method
        images = np.random.uniform(-3, 3, size=[chn[layer], shape[layer][0], shape[layer][0], 1]).astype('float32')
        # images = np.reshape(wave[32:96, 32:96].astype('float32'), [1,64,64,1])
        # images = np.load(pwd+'/un_1.npy')

        np.save(pwd+'/test.npy',images)
        init = images

        for j in range(201):
            xx = []
            x = []
            for i in range(chn[layer]):
                # do iter for all images of all chns
                image_chn_i = images[i, :, :, :].reshape([1, shape[layer][0], shape[layer][0], 1]).astype('float32')
                feedData = {g.feed[0]:image_chn_i, g.feed[1]: True}
                fetchVariables = [g.para[0][i], g.para[1]]

                [gradient, result] = sess.run(fetches=fetchVariables, feed_dict=feedData)
                print('iter:%s,chn:%s'%(j, i))
                # print(len(gradient))
                # print(gradient[0].shape)
                gradient = gradient[0]
                xx.append(gradient)
                x.append(result[0, i])
                dis = np.abs(image_chn_i * gradient) # distribution of image

                # decay method and step=50
                image_chn_i = (image_chn_i + 50 * gradient)*0.9
                # cut small pixel
                # images[np.where(np.abs(images) <= images.max()/100)]=0

                # cut smal distribute
                image_chn_i[np.where(dis <= dis.mean() / 50)] = 0
                # guass smooth
                images[i, :, :, :] = convolve2d(image_chn_i[0, :, :, 0], kernel, mode='same').reshape(
                    [shape[layer][0], shape[layer][0], 1]).astype('float32')
                # images[i, :, :, :] = image_chn_i[0, :, :, 0].reshape(
                #     [shape[layer][0], shape[layer][0], 1]).astype('float32')

            if j%20 == 0:
                # r.append(l[0][1])
                # print(i)
                # continue
                # print(images[5,:,:,:])
                # print(xx[5])
                w = int(chn[layer] ** 0.5)
                h = chn[layer] // int(chn[layer] ** 0.5) + 1
                plt.figure(1, figsize=[w, h], dpi=150)
                for k in range(chn[layer]):
                    ax = subplot(w, h, k+1)
                    pcolormesh(images[k, :, :, 0].T, cmap='seismic')
                    ax.invert_yaxis()
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.axis('equal')
                    # colorbar()
                    if x[k]<=0.001:
                        title('Neg')
                    else:
                        title('%4.2s'%x[k])

                # plt.subplots_adjust(wspace=0, hspace=0)
                plt.suptitle('layer:%s,iter:%s'%(layer, j))
                savefig('C:\\Users\shiyx\OneDrive\Python Scripts\CNN_visualize\\report\\layer%s\\%s.jpg' % (layer, j))
                # plt.show()
                plt.close('all')


    return None
# deconvnet()
reversion()
# reversion_conv()