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
import subprocess

repos_path = os.getenv('GIT_REPOS_PATH')
bimvee_path = os.environ.get('BIMVEE_PATH')
wp5_slam_path = os.environ.get('WP5_SLAM_PATH')

imu_tk2_path = '/home/leandro/repos/tools/imu_tk'
imu_tk2_app = imu_tk2_path+'/bin/test_imu_calib'
data_path = '/home/leandro/results/IMUDataDump'
plot_path = data_path+'/plots'

if( not os.path.isdir(plot_path)):
    print('creating folder for plots')
    os.mkdir(plot_path)
    
raw_acc_files_regex = '/acc[0-9][0-9]'
raw_gyr_files_regex = '/gyr[0-9][0-9]'

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


#%% Perform calibrations on the files
suffix = 'base'

for acc, gyr in zip(acc_files, gyr_files):
    acc_params = acc+'.'+suffix
    gyr_params = gyr+'.'+suffix
    
    # skip if these were already performed
    if(os.path.isfile(acc_params) and os.path.isfile(gyr_params)):
        print('already calculated ' + acc_params + ' and ' + gyr_params)
        continue

    print(acc, gyr)

    command = [imu_tk2_app,
               '--acc_file='+acc, 
               '--gyr_file='+gyr, 
               '--suffix='+suffix,
               '--opt_gyr_b=false',
               '--min_acc_b=false',
               '--min_gyr_b=false'
               ]
    print(command)
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

#%% Perform calibrations on the files
suffix = 'optGyrBias'

for acc, gyr in zip(acc_files, gyr_files):
    acc_params = acc+'.'+suffix
    gyr_params = gyr+'.'+suffix
    
    # skip if these were already performed
    if(os.path.isfile(acc_params) and os.path.isfile(gyr_params)):
        print('already calculated ' + acc_params + ' and ' + gyr_params)
        continue

    print(acc, gyr)

    command = [imu_tk2_app,
               '--acc_file='+acc, 
               '--gyr_file='+gyr, 
               '--suffix='+suffix,
               '--min_acc_b=false',
               '--min_gyr_b=false',
               '--opt_gyr_b=true',
               '--max_iter=500'
               ]
    print(command)
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

#%% Perform calibrations on the files
suffix = 'minAccBiases'

for acc, gyr in zip(acc_files, gyr_files):
    acc_params = acc+'.'+suffix
    gyr_params = gyr+'.'+suffix
    
    # skip if these were already performed
    if(os.path.isfile(acc_params) and os.path.isfile(gyr_params)):
        print('already calculated ' + acc_params + ' and ' + gyr_params)
        continue

    print(acc, gyr)

    command = [imu_tk2_app,
               '--acc_file='+acc, 
               '--gyr_file='+gyr, 
               '--suffix='+suffix,
               '--opt_gyr_b=false',
               '--min_acc_b=true',
               '--min_gyr_b=false'
               ]
    print(command)
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#%% Read the parameters and plot them
from utils import readAllCalibParams, plotAllCalibParams

skew = dict()
scale = dict()
bias = dict()
suffixes = ['base', 'optGyrBias', 'minAccBiases'] 

for suffix in suffixes:
    # get the name of all param files in the folder
    acc_params = np.sort(glob.glob(data_path+raw_acc_files_regex+'.'+suffix))
    gyr_params = np.sort(glob.glob(data_path+raw_gyr_files_regex+'.'+suffix))    
    readAllCalibParams(acc_params, gyr_params, suffix, skew, scale, bias)

plotAllCalibParams(skew, scale, bias, plot_path)
#%%
os.system('mv ' + data_path + '/*.png' +' ' + plot_path)