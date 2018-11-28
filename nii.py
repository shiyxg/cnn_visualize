# -*- coding: utf-8 -*-
from __future__ import print_function
import nibabel as nb
import numpy as np
import scipy.io as io
import sys
import matplotlib.pyplot as pyplot
import matplotlib.cm as cm

'''
定义一个NII文件的读取类，以帮助CNN等获取训练样本
实现的功能有下：
1, read data&marks
2, pick
3, batch data&marks

需要提前给定文件夹的路径path
然后会在文件夹里面读取两个文件（不存在会报错）
Recorded.nii.gz & FaultMarks.nii.gz
'''


# for rotate,shape=[XL,YL]
def rotate(ori, outputShape, angel):
    inputShape = ori.shape
    output = np.zeros(outputShape)
    index = np.zeros([outputShape[0], outputShape[1], 2])

    sin = np.sin(angel * np.pi / 180)
    cos = np.cos(angel * np.pi / 180)
    center_out = [outputShape[0] // 2, outputShape[1] // 2]
    center_ori = [inputShape[0] // 2, inputShape[1] // 2]
    XL, YL = outputShape
    for i in range(XL):
        for j in range(YL):
            x = (i - center_out[0]) * cos - (j - center_out[1]) * sin + center_ori[0]
            y = (i - center_out[0]) * sin + (j - center_out[1]) * cos + center_ori[1]
            x1 = int(np.floor(x))
            x2 = int(np.ceil(x))
            y1 = int(np.floor(y))
            y2 = int(np.ceil(y))
            if x1 == x2 and y1 != y2:
                num = (y - y1) * ori[x1, y1] + (y2 - y) * ori[x1, y2]
            elif x1 != x2 and y1 == y2:
                num = (x - x1) * ori[x1, y1] + (x2 - x) * ori[x2, y1]
            elif x1 == x2 and y1 == y2:
                num = ori[x1, y1]
            else:
                num = (x - x1) * (y - y1) * ori[x1, y1] + \
                      (x - x2) * (y - y2) * ori[x2, y2] + \
                      (x - x1) * (y2 - y) * ori[x1, y2] + \
                      (x2 - x) * (y - y1) * ori[x2, y1]
            num = ori[x1, y1]
            output[i, j] = num

    return output


class nii():
    def __init__(self, path, expand=32, delOriginData=False, modelIndex=6):
        self.path = path
        self.filename = {'Record': (path + '/Recorded.nii.gz'), 'Marks': (path + '/FaultMarks.nii.gz')}
        self.model = path.split('\\')[modelIndex]
        self.recordFile = nb.load(self.filename['Record'])
        self.marksFile = nb.load(self.filename['Marks'])

        self.oldShape = np.array(self.recordFile.shape)
        self.shape = np.array([self.oldShape[0], self.oldShape[2], self.oldShape[1]])
        self.size = self.oldShape[0] * self.oldShape[1] * self.oldShape[2];
        # self.expand   将三维数据拓展边缘，用零填充，以便后续处理，
        # 但是在pick的时候不用考虑expand
        self.expand = expand

        shape = self.shape
        self.record = np.zeros(shape + expand * 2)
        self.marks = np.zeros(shape + expand * 2)

        record = self.recordFile.get_data().transpose([0, 2, 1])
        marks = self.marksFile.get_data().transpose([0, 2, 1])
        self.faultIndex = np.array(np.where(marks == 1.0))
        self.record[expand:(shape[0] + expand),
        expand:(shape[1] + expand),
        expand:(shape[2] + expand)] = record
        self.marks[expand:(shape[0] + expand),
        expand:(shape[1] + expand),
        expand:(shape[2] + expand)] = marks

        self.Fault_rate = np.count_nonzero(marks) * 1.0 / self.size

        print('Fault/NonFault = ', self.Fault_rate)
        if delOriginData:
            del self.recordFile
            del self.marksFile

    # batch 函数
    # 给定数量Num与shape（2D）（小于expand），随机抽取样本
    # 输出两个矩阵record:[Num,shape[0],shape[1]] & marks：[Num,shape[0],shape[1]]
    def batch(self, batchNum, shape=[32, 32], sampleAxis=13, randomBatch=True, addProb=0.7):

        if randomBatch:
            record_sample = np.zeros([batchNum, shape[0], shape[1]])
            marks_sample = np.zeros([batchNum, shape[0], shape[1]])

            for i in range(batchNum):
                loc_float = np.random.random(3) * self.shape
                index = loc_float.astype('int')
                [a, b] = self.pick(index, shape, sampleAxis)

                # print index
                record_sample[i, :, :] = a
                marks_sample[i, :, :] = b
        else:
            record_sample = np.zeros([batchNum, shape[0], shape[1]])
            marks_sample = np.zeros([batchNum, shape[0], shape[1]])

            for i in range(batchNum):
                ran = np.random.random(1) * 1.0 / addProb

                if ran <= 1:
                    loc_float = np.random.random(1) * self.faultIndex.shape[1]
                    loc_int = loc_float.astype('int')
                    index = self.faultIndex[:, loc_int]
                    index = index.reshape([1, 3])[0]

                else:
                    loc_float = np.random.random(3) * self.shape
                    index = loc_float.astype('int')

                # print index
                [a, b] = self.pick(index, shape, sampleAxis)
                record_sample[i, :, :] = a
                marks_sample[i, :, :] = b

        return [record_sample, marks_sample]

        # pick函数， 给定pick的中心像素的index与pick的shape（2D）（小于expand×2）

    # axis 函数是确定sample的时候取样的轴（x:1.y:2）
    # 自动考虑expand的存在
    def pick(self, index, shape, sampleAxis=13, IMAGE=False):
        index = index + self.expand
        if sampleAxis == 13:
            record_sample = self.record[(index[0] - shape[0] // 2):(index[0] + shape[0] - shape[0] // 2),
                            index[1],
                            (index[2] - shape[1] // 2):(index[2] + shape[1] - shape[1] // 2)]
            marks_sample = self.marks[(index[0] - shape[0] // 2):(index[0] + shape[0] - shape[0] // 2),
                           index[1],
                           (index[2] - shape[1] // 2):(index[2] + shape[1] - shape[1] // 2)]
        elif sampleAxis == 23:
            record_sample = self.record[index[0],
                            (index[1] - shape[0] // 2):(index[1] + shape[0] - shape[0] // 2),
                            (index[2] - shape[1] // 2):(index[2] + shape[1] - shape[1] // 2)]
            marks_sample = self.marks[index[0],
                           (index[1] - shape[0] // 2):(index[1] + shape[0] - shape[0] // 2),
                           (index[2] - shape[1] // 2):(index[2] + shape[1] - shape[1] // 2)]
        return [record_sample, marks_sample]

    def pickLayer(self, layerNum, sampleAxis=13, IMAGE=False):
        [XL, YL, TL] = self.shape
        if sampleAxis == 13:
            a = self.marks[self.expand:(XL + self.expand),
                self.expand + layerNum,
                self.expand:(TL + self.expand)]
            b = self.record[self.expand:(XL + self.expand),
                self.expand + layerNum,
                self.expand:(TL + self.expand)]
        elif sampleAxis == 23:
            a = self.marks[self.expand + layerNum,
                self.expand:(YL + self.expand),
                self.expand:(TL + self.expand)]
            b = self.record[self.expand + layerNum,
                self.expand:(YL + self.expand),
                self.expand:(TL + self.expand)]
        return [b, a]

    def save(self):
        pass

    # return batch(images 256)
    def batchImages(self, batchNum, NormRange=[-3, 3],
                    shape=[32, 32],
                    sampleAxis=13, randomBatch=True, addProb=0.7):

        [record, labels] = self.batch(batchNum, shape,
                                      sampleAxis, randomBatch, addProb=addProb)
        max = NormRange[1]
        min = NormRange[0]

        record = (record - min) / (max - min) * 255.0
        record = np.round(record)

        maxIndex = np.where(record > 255)
        minIndex = np.where(record < 0)

        record[maxIndex] = 255.0;
        record[minIndex] = 0.0

        return [record, labels]

    # return batch Images

    # auto batch num images. random, shift,rotate reverse will finished autoly
    def AutoBatch(self, batchNum, NormRange=[-3, 3],
                  shape=[32, 32],
                  addProb=0.65, ACT=1):
        record_sample = np.zeros([batchNum, ACT, shape[0], shape[1]])
        marks_sample = np.zeros([batchNum, ACT, shape[0], shape[1]])

        for i in range(batchNum):
            ran = np.random.random(1) * 1.0 / addProb

            if ran <= 1:
                loc_float = np.random.random(1) * self.faultIndex.shape[1]
                loc_int = loc_float.astype('int')
                index = self.faultIndex[:, loc_int]
                index = index.reshape([1, 3])[0]

            else:
                loc_float = np.random.random(3) * self.shape
                index = loc_float.astype('int')

            # print(i)
            [a, b] = self.AutoPick(index, shape, ACT=ACT)
            record_sample[i, :, :, :] = a
            marks_sample[i, :, :, :] = b

        record = record_sample.reshape([batchNum * ACT, shape[0], shape[1]])
        labels = marks_sample.reshape([batchNum * ACT, shape[0], shape[1]])

        return [record, labels]

    def AutoPick(self, index, shape, ACT=6):
        shift = np.zeros(3)
        # shift[1:3] = np.random.random(2)*3-1
        shift = shift.astype('int')
        index = index + self.expand + shift
        prob = np.random.random()
        cache = shape
        # shape = np.ceil(np.array(shape)*2**0.5).astype('int')
        # print(shape)
        # print(index)
        if prob < 1.0:
            record_sample = self.record[(index[0] - shape[0] // 2):(index[0] + shape[0] - shape[0] // 2),
                            index[1],
                            (index[2] - shape[1] // 2):(index[2] + shape[1] - shape[1] // 2)]
            marks_sample = self.marks[(index[0] - shape[0] // 2):(index[0] + shape[0] - shape[0] // 2),
                           index[1],
                           (index[2] - shape[1] // 2):(index[2] + shape[1] - shape[1] // 2)
                           ]
        else:
            record_sample = self.record[index[0],
                            (index[1] - shape[0] // 2):(index[1] + shape[0] - shape[0] // 2),
                            (index[2] - shape[1] // 2):(index[2] + shape[1] - shape[1] // 2)
                            ]
            marks_sample = self.marks[index[0],
                           (index[1] - shape[0] // 2):(index[1] + shape[0] - shape[0] // 2),
                           (index[2] - shape[1] // 2):(index[2] + shape[1] - shape[1] // 2)
                           ]
        shape = cache
        record = np.zeros([ACT, shape[0], shape[1]])
        marks = np.zeros([ACT, shape[0], shape[1]])
        record[0, :, :] = record_sample
        marks[0, :, :] = marks_sample
        return [record, marks]


class niiS():
    def __init__(self, trainPaths, testPaths, trainWeight=None, testWeight=None, expand=50, delOriginData=False):
        self.train = []
        self.test = []

        for path in trainPaths:
            self.train.append(nii(path, expand, delOriginData))
        for path in testPaths:
            self.test.append(nii(path, expand, delOriginData))

        self.expand = expand

        if trainWeight == None:
            trainWeight = np.ones(len(self.train))
        if testWeight == None:
            testWeight = np.ones(len(self.test))

        self.trainWeight = trainWeight
        self.testWeight = testWeight

    def trainBatch(self, batchNum, addProb=0.5, shape=[32, 32], ACT=1):
        # distribute the num of batches from each dataset
        ran = np.random.random(len(self.train)) * self.trainWeight
        num_float = ran / ran.sum() * batchNum
        nums = num_float.astype('int')
        nums[-1] = batchNum - nums[0:-1].sum()
        # print(nums)
        # init
        record = np.zeros([batchNum * ACT, shape[0], shape[1]])
        labels = np.zeros([batchNum * ACT, shape[0], shape[1]])

        # batch from each dataset
        for i in range(len(self.train)):
            [record_i, labels_i] = self.train[i].AutoBatch(nums[i], shape=shape, addProb=addProb, ACT=ACT)

            start = nums[0:i].sum() * ACT
            end = nums[i] * ACT + start
            # print([start,end])
            record[start:end] = record_i
            labels[start:end] = labels_i
        label_re = np.zeros([batchNum * ACT, 2])
        label_re[:, 0] = (labels[:, shape[0]//2, shape[0]//2] == 0).astype('float32')
        label_re[:, 1] = (labels[:, shape[0]//2, shape[0]//2] == 1).astype('float32')
        record = np.reshape(record, [batchNum * ACT, shape[0], shape[1], 1])
        return [record, label_re]

    def testBatch(self, batchNum, addProb=0.5, shape=[32, 32], ACT=1):
        # distribute the num of batches from each dataset
        ran = np.random.random(len(self.test)) * self.testWeight
        num_float = ran / ran.sum() * batchNum
        nums = num_float.astype('int')
        nums[-1] = batchNum - nums[0:-1].sum()

        # init
        record = np.zeros([batchNum * ACT, shape[0], shape[1]])
        labels = np.zeros([batchNum * ACT, shape[0], shape[1]])

        # batch from each dataset
        for i in range(len(self.test)):
            [record_i, labels_i] = self.test[i].AutoBatch(nums[i], shape=shape, addProb=addProb, ACT=ACT)

            start = nums[0:i].sum() * ACT
            end = nums[i] * ACT + start
            record[start:end] = record_i
            labels[start:end] = labels_i

        label_re = np.zeros([batchNum * ACT, 2])
        label_re[:, 0] = (labels[:, shape[0] // 2, shape[0] // 2] == 0).astype('float32')
        label_re[:, 1] = (labels[:, shape[0] // 2, shape[0] // 2] == 1).astype('float32')
        record = np.reshape(record, [batchNum * ACT, shape[0], shape[1], 1])
        return [record, label_re]

    '''
    下面的两个函数是用于FCN 网络的batch，每次给一层，防止没有pick到有断层的位置
    '''

    def trainBatchLayer(self, batchNum, shape=[250, 400], sampleAxis=13):
        assert sampleAxis == 13
        modelNum = len(self.train)
        layerNum = np.zeros(modelNum)
        for i in range(modelNum):
            layerNum[i] = self.train[i].shape[1]

        record = np.zeros([batchNum, shape[0], shape[1]])
        marks = np.zeros([batchNum, shape[0], shape[1]])
        for i in range(batchNum):
            while 1:
                modelIndex = int(np.floor(np.random.random() * modelNum))
                layerIndex = int(np.floor(np.random.random() * layerNum[modelIndex]))
                model = self.train[modelIndex]
                data, label = model.pickLayer(layerIndex, sampleAxis=sampleAxis)
                faultPicked = len(np.where(label == 1)[0])
                if faultPicked:
                    max = 3
                    min = -3
                    data = (data - min) / (max - min) * 2 - 1
                    maxIndex = np.where(data > 1)
                    minIndex = np.where(data < -1)
                    data[maxIndex] = 1.0;
                    data[minIndex] = -1.0

                    record[i, :, :] = data
                    marks[i, :, :] = label
                    break

        return [record, marks]

    def testBatchLayer(self, batchNum, shape=[250, 400], sampleAxis=13):
        assert sampleAxis == 13
        modelNum = len(self.test)
        layerNum = np.zeros(modelNum)
        for i in range(modelNum):
            layerNum[i] = self.test[i].shape[1]

        record = np.zeros([batchNum, shape[0], shape[1]])
        marks = np.zeros([batchNum, shape[0], shape[1]])
        for i in range(batchNum):
            while 1:
                modelIndex = int(np.floor(np.random.random() * modelNum))
                layerIndex = int(np.floor(np.random.random() * layerNum[modelIndex]))
                model = self.test[modelIndex]
                data, label = model.pickLayer(layerIndex, sampleAxis=sampleAxis)
                faultPicked = len(np.where(label == 1)[0])
                if faultPicked:
                    max = 3
                    min = -3
                    data = (data - min) / (max - min) * 2 - 1
                    maxIndex = np.where(data > 1)
                    minIndex = np.where(data < -1)
                    data[maxIndex] = 1.0;
                    data[minIndex] = -1.0

                    record[i, :, :] = data
                    marks[i, :, :] = label
                    break

        return [record, marks]


'''


data = niiS(['/home/shi/FaultDetection/data/SYN/FCN1_OA3_f10_n10/Nifti'],['/home/shi/FaultDetection/data/SYN/FCN2_OA3_f40_n10/Nifti'],expand = 0)
a = data.trainBatchLayer(10)
for i in range(10):
    pyplot.imshow(a[0][i,:,:].T+a[1][i,:,:].T)
    pyplot.savefig('/home/shi/CNNlog/FCN/test/train/%g.png'%i)

a = data.testBatchLayer(10)
for i in range(10):
    pyplot.imshow(a[0][i,:,:].T+a[1][i,:,:].T)
    pyplot.savefig('/home/shi/CNNlog/FCN/test/test/%g.png'%i)
'''

'''

a = nii('/home/shi/FaultDetection/data/SYN/FCN1_OA3_f10_n10/Nifti')

r,m = a.pick(np.array([100,100,100]),[200,200])
pyplot.figure()
pyplot.imshow(r.T+2*m.T,cmap = cm.gray)
pyplot.show()
pyplot.figure()
angle = 10
r10 = rotate(r,[125,125],angle)
m10 = rotate(m,[125,125],angle)
pyplot.imshow(r10.T+2*m10.T,cmap = cm.gray)
pyplot.show()
'''

'''
test =['C:\\Users\\Shi\\Documents\\FaultDetection\\data\\Synthetic\\2\\Nifti',
       'C:\\Users\\Shi\\Documents\\FaultDetection\\data\\Synthetic\\3\\Nifti',
       'C:\\Users\\Shi\\Documents\\FaultDetection\\data\\Synthetic\\3\\Nifti',
       'C:\\Users\\Shi\\Documents\\FaultDetection\\data\\Synthetic\\3\\Nifti',
       'C:\\Users\\Shi\\Documents\\FaultDetection\\data\\Synthetic\\3\\Nifti']

train = ['C:\\Users\\Shi\\Documents\\FaultDetection\\data\\Synthetic\\FCN1_OA3_f10_n10\\Nifti',
         'C:\\Users\\Shi\\Documents\\FaultDetection\\data\\Synthetic\\4\\Nifti',
         'C:\\Users\\Shi\\Documents\\FaultDetection\\data\\Synthetic\\FCN1_OA3_f10_n10\\Nifti',
         'C:\\Users\\Shi\\Documents\\FaultDetection\\data\\Synthetic\\3\\Nifti',
         'C:\\Users\\Shi\\Documents\\FaultDetection\\data\\Synthetic\\3\\Nifti']

a = niiS(train,test)

a.trainBatch(100)
a.testBatch(100)

path  = '/home/shi/FaultDetection/data/Synthetic/FCN1_OA3_f10_n10/Nifti'

test = nii(path)
l = 10
shape=[32,32]
[record,label] = test.AutoBatch(l,addProb=1,shape=shape)

for i in range(l):
    a = np.reshape(record[i,:,:],shape)
    b = np.reshape(label[i,:,:], shape)
    #figure = a.T + b.T*10
    pyplot.pcolor(a.T)
    pyplot.savefig('/home/shi/CNNlog/CNN/FD/test/%g record.png'%i)

    pyplot.pcolor(a.T+b.T*3)
    pyplot.savefig('/home/shi/CNNlog/CNN/FD/test/%g all.png'%i)


a = test.pickLayer(125)
b = a[0]
c = a[1]
pyplot.pcolor(b.T)
pyplot.savefig('/home/shi/CNNlog/CNN/FD/test/record.png')

pyplot.pcolor(b.T+c.T*3)
pyplot.savefig('/home/shi/CNNlog/CNN/FD/test/all.png')

pyplot.pcolor(c.T*3)
pyplot.savefig('/home/shi/CNNlog/CNN/FD/test/label.png')

num=1000
add = 1
a = test.AutoBatch(num,addProb = add)
b = a[1][:,14:18,14:18]

index = np.where(b==1.0)
print(len(index[0])*1.0/(num*16))
'''