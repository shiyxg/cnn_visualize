import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from scipy.signal import convolve2d
pwd = 'C:\\Users\shiyx\Documents\ML_log\CNN_visual\\test01\\'
wave = np.load(pwd+'/train/FCN1_OA3_f10_n10_250000_data.npy')
label = np.load(pwd+'/test/FCNO_OA3A2_f20_n20_250000_id.npy')
para_path = 'C:/Users/shiyx/Documents/ML_log/CNN_visual/test01/para_npy/'
cores = []
bn = []
bias = []
weight  = []
for i in range(5):
    cores_i = np.load(para_path + 'cores_%s.npy' % i)
    cores.append(cores_i)
for i in range(9):
    bn_i = np.load(para_path + 'bn_%s.npy' % i)
    bn.append(bn_i)
for i in range(3):
    bias_i = np.load(para_path + 'bias_%s.npy' % i)
    weight_i = np.load(para_path + 'weight_%s.npy' % i)
    bias.append(bias_i)
    weight.append(weight_i)

# wave = wave[32:96, 32:96]
figure()
cmap='seismic'
ax = subplot(1,1,1)
wave[0,0] = np.abs(wave).max()
wave[0,1] = -np.abs(wave).max()
label[np.where(label==0)]=None
pcolormesh(wave.T, cmap=cmap)
colorbar()
pcolormesh(label.T)
ax.invert_yaxis()
ax.plot([32,96], [20,20],color='yellow',linewidth=2)
ax.plot([32,96], [84,84],color='yellow',linewidth=2)
ax.plot([32,32], [20,84],color='yellow',linewidth=2)
ax.plot([96,96], [20,84],color='yellow',linewidth=2)


figure()
ax = subplot(1,3,2)
b = bn[0]
print(bn[1])
wave = wave[50:114, 200:264]
wave = b[1]*(wave-b[2])/b[3]+b[0]
wave[0,0] = np.abs(wave).max()
wave[0,1] = -np.abs(wave).min()
pcolormesh(wave.T,cmap=cmap)
ax.invert_yaxis()
colorbar()
figure(figsize=[8,8],dpi=100)
for i in range(32):
    ax = subplot(5,7,i+1)
    c = convolve2d(wave, cores[0][:,:,0,i], mode='same')
    c[0,0] = np.abs(c).max()
    c[0,1] = -np.abs(c).max()
    pcolormesh(c.T,cmap=cmap)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    print(i)
    c = cores[0][:,:,0,i]/np.abs(cores[0][:,:,0,i]).min()
    print(np.round(c))
show()
