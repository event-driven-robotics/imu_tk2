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
plot_path = data_path+'/plots'

if(not os.path.isdir(plot_path)):
    print('creating folder for plots')
    os.mkdir(plot_path)
    
raw_acc_files_regex = '/static_acc[0-9][0-9]'
raw_gyr_files_regex = '/static_gyr[0-9][0-9]'

sys.path.insert(0, repos_path)
sys.path.insert(0, bimvee_path)
sys.path.insert(0, wp5_slam_path)
sys.path.insert(0, imu_tk2_path+'/python')

g_mag=9.805622
nominal_gyr_scale = 250 * np.pi / (2.0 * 180.0 * 16384.0)
nominal_acc_scale = g_mag/16384.0

#%% get raw data files

acc_files = np.sort(glob.glob(data_path+raw_acc_files_regex))
gyr_files = np.sort(glob.glob(data_path+raw_gyr_files_regex))

print(acc_files)
print(gyr_files)

assert acc_files.size == gyr_files.size, 'number of files for calibration should be the same'


#%% plot biases for gyro in different positions

from utils import readRawData, num2leg
from matplotlib import pyplot as plt

plt.close('all')
lab = ['x','y','z']

fig, axs = plt.subplots(len(acc_files), 1, sharex='all', figsize=(8,10))
fig.suptitle('gyro biases for different static orientations')
for i, (acc, gyr) in enumerate(zip(acc_files, gyr_files)):
    raw_acc = readRawData(acc)
    raw_gyr = readRawData(gyr)
    
    acc_data = raw_acc*nominal_gyr_scale;
    acc_data[:,0] = raw_acc[:,0]
    gyr_data = raw_gyr*nominal_gyr_scale;
    gyr_data[:,0] = raw_gyr[:,0]
    
    legend = []
    for j in range(3):
        legend.append( lab[j] + r' - avg=' + num2leg(gyr_data[:, j+1].mean()) + ' $\pm$ ' + num2leg(gyr_data[:, j+1].std()))
    
    axs[i].hist(gyr_data[:, 1:4], bins=30, label=legend, rwidth = 0.8, alpha = 0.7)
    axs[i].legend();
    axs[i].set_xlabel('Ang. Vel (radians/s)')
    axs[i].set_ylabel('# occurences')  

fig.tight_layout()
plt.savefig(plot_path+'/static_gyro_biases.png')   
