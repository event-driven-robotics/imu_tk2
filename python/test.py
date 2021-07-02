#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 
@author: Leandro de Souza Rosa <leandro.desouzarosa@iit.it>

"""

# %% Preliminaries
import numpy as np
import os
import sys
import glob

repos_path = os.getenv('GIT_REPOS_PATH')
bimvee_path = os.environ.get('BIMVEE_PATH')
wp5_slam_path = os.environ.get('WP5_SLAM_PATH')

imu_tk2_path = '/home/leandro/repos/tools/imu_tk'
imu_tk2_app = imu_tk2_path+'/bin/test_imu_calib'
data_path = '/home/leandro/results/IMUDataDump'

raw_acc_files_regex = '/static_acc[0-9][0-9]'
raw_gyr_files_regex = '/static_gyr[0-9][0-9]'

sys.path.insert(0, repos_path)
sys.path.insert(0, bimvee_path)
sys.path.insert(0, wp5_slam_path)
sys.path.insert(0, imu_tk2_path+'/python')

#%% get raw data files

acc_files = np.sort(glob.glob(data_path+raw_acc_files_regex))
gyr_files = np.sort(glob.glob(data_path+raw_gyr_files_regex))

print(acc_files)
print(gyr_files)

assert acc_files.size == gyr_files.size, 'number of files for calibration should be the same'


#%% plot biases for gyro in different positions

from utils import readRawData
from matplotlib import pyplot as plt
plt.close('all')

fig, axs = plt.subplots(len(acc_files), 1, sharex='all', figsize=(8,10))
fig.suptitle('gyro biases for different static orientations')
for i, (acc, gyr) in enumerate(zip(acc_files, gyr_files)):
    acc_data = readRawData(acc)
    gyr_data = readRawData(gyr)
    
    axs[i].hist(gyr_data[:, 1:4], bins=30, label=['x','y','z'], rwidth = 0.8, alpha = 0.7)
    axs[i].legend();
    axs[i].set_ylabel('gyro')
    axs[i].set_xlabel('# occurences')  

fig.tight_layout()
plt.savefig(data_path+'/static_gyro_biases.png')   

#%% plot accelerometer-gyro correlation

from utils import readRawData
from matplotlib import pyplot as plt
plt.close('all')

fig, axs = plt.subplots(len(acc_files), 3, figsize=(15,15))
fig.suptitle('accelerometer-gyro correlation')
leg = ['x','y','z']

for i, (acc, gyr) in enumerate(zip(acc_files, gyr_files)):
    acc_data = readRawData(acc)
    gyr_data = readRawData(gyr)
    
    for j in range(3):
        axs[i][j].plot(acc_data[:, j+1], gyr_data[:, j+1], '.', label=leg[j])
        axs[i][j].legend();
        axs[i][j].set_ylabel('gyro')
        axs[i][j].set_xlabel('acc')  

#fig.tight_layout()
#plt.savefig(data_path+'/acc_gyro_corr.png')   
