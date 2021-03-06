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
params_path = data_path+'/params'

if( not os.path.isdir(plot_path)):
    print('creating folder for plots')
    os.mkdir(plot_path)
    
if( not os.path.isdir(params_path)):
    print('creating folder for params')
    os.mkdir(params_path)
    
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
    acc_params = params_path + '/' + acc.split('/')[-1] + '.'+suffix
    gyr_params = params_path + '/' + gyr.split('/')[-1] + '.'+suffix

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
               '--max_iter=500',
               '--acc_use_means=false',
               '--use_gyr_G=false'
               ]
    print(command)
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

#%% Perform calibrations on the files
suffix = 'optGyrBias'

for acc, gyr in zip(acc_files, gyr_files):
    acc_params = params_path + '/' + acc.split('/')[-1] + '.'+suffix
    gyr_params = params_path + '/' + gyr.split('/')[-1] + '.'+suffix
    
    # skip if these were already performed
    if(os.path.isfile(acc_params) and os.path.isfile(gyr_params)):
        print('already calculated ' + acc_params + ' and ' + gyr_params)
        continue

    print(acc, gyr)

    command = [imu_tk2_app,
               '--acc_file='+acc, 
               '--gyr_file='+gyr, 
               '--suffix='+suffix,
               '--opt_gyr_b=true',
               '--max_iter=500',
               '--acc_use_means=false',
               '--use_gyr_G=false'
               ]
    print(command)
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#%% Perform calibrations on the files
suffix = 'accMeans_base'

for acc, gyr in zip(acc_files, gyr_files):
    acc_params = params_path + '/' + acc.split('/')[-1] + '.'+suffix
    gyr_params = params_path + '/' + gyr.split('/')[-1] + '.'+suffix
    
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
               '--max_iter=500',
               '--acc_use_means=true',
               '--use_gyr_G=false'
               ]
    print(command)
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

#%% Perform calibrations on the files
suffix = 'accMeans_optGyrBias'

for acc, gyr in zip(acc_files, gyr_files):
    acc_params = params_path + '/' + acc.split('/')[-1] + '.'+suffix
    gyr_params = params_path + '/' + gyr.split('/')[-1] + '.'+suffix
    
    # skip if these were already performed
    if(os.path.isfile(acc_params) and os.path.isfile(gyr_params)):
        print('already calculated ' + acc_params + ' and ' + gyr_params)
        continue

    print(acc, gyr)

    command = [imu_tk2_app,
               '--acc_file='+acc, 
               '--gyr_file='+gyr, 
               '--suffix='+suffix,
               '--opt_gyr_b=true',
               '--max_iter=500',
               '--acc_use_means=true',
               '--use_gyr_G=false'
               ]
    print(command)
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#%% Perform calibrations on the files
suffix = 'gyroG'

for acc, gyr in zip(acc_files, gyr_files):
    acc_params = params_path + '/' + acc.split('/')[-1] + '.'+suffix
    gyr_params = params_path + '/' + gyr.split('/')[-1] + '.'+suffix
    
    # skip if these were already performed
    if(os.path.isfile(acc_params) and os.path.isfile(gyr_params)):
        print('already calculated ' + acc_params + ' and ' + gyr_params)
        continue

    print(acc, gyr)

    command = [imu_tk2_app,
               '--acc_file='+acc, 
               '--gyr_file='+gyr, 
               '--suffix='+suffix,
               '--opt_gyr_b=true',
               '--max_iter=500',
               '--acc_use_means=false',
               '--use_gyr_G=true'
               ]
    print(command)
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
#%%
suffixes = ['base', 'optGyrBias', 'accMeans_base', 'accMeans_optGyrBias', 'gyroG'] 

os.system('mv ' + data_path + '/*.png' +' ' + plot_path)
for suffix in suffixes:
    os.system('mv ' + data_path + '/*.'+ suffix +' ' + params_path)
    
#%% Read the parameters and plot them
from utils import readAllCalibParams, plotAllCalibParams

skew = dict()
scale = dict()
bias = dict()
gmat = dict()


for suffix in suffixes:
    # get the name of all param files in the folder
    acc_params = np.sort(glob.glob(params_path+raw_acc_files_regex+'.'+suffix))
    gyr_params = np.sort(glob.glob(params_path+raw_gyr_files_regex+'.'+suffix))    
    readAllCalibParams(acc_params, gyr_params, suffix, skew, scale, bias, gmat)

plotAllCalibParams(skew, scale, bias, gmat, plot_path)
