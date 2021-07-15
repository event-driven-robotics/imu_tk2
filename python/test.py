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

g_mag = 9.805622
nominal_gyr_scale = 250 * np.pi / (2.0 * 180.0 * 16384.0)
nominal_acc_scale = g_mag/16384.0


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

#%% Read the parameters and plot them
from utils import readAllCalibParams#, plotAllCalibParams

skew = dict()
scale = dict()
bias = dict()
suffixes = ['base', 'optBias', 'minAccBiases'] 

for suffix in suffixes:
    # get the name of all param files in the folder
    # get the name of all param files in the folder
    acc_params_regex = '/acc[0-9][0-9]'+'.'+suffix
    gyr_params_regex = '/gyr[0-9][0-9]'+'.'+suffix
    
    acc_params = np.sort(glob.glob(data_path+acc_params_regex))
    gyr_params = np.sort(glob.glob(data_path+gyr_params_regex))
    
    readAllCalibParams(acc_params, gyr_params, suffix, skew, scale, bias)

#plotAllCalibParams(skew, scale, bias, data_path)

#%% Apply Calibrations on each data
from utils import readRawData, calibrate

#skew, scale, bias = [suffix]['acc'|'gyr']
calib_acc_data = dict()
calib_gyr_data = dict()
acc_data = dict()
gyr_data = dict()

for acc, gyr in zip(acc_files, gyr_files):
    #save the data
    acc_data[acc] = readRawData(acc)
    gyr_data[gyr] = readRawData(gyr)
    
    print('calibrating data from files: ' + acc.split('/')[-1] + ' and ' + gyr.split('/')[-1])
    
    # apply all the calibrations for each suffix
    calib_acc_data[acc] = dict()
    calib_gyr_data[gyr] = dict()
    
    for suffix in suffixes:
        temp_acc = []
        temp_gyr = []
        
        for calib_n in range(len(skew[suffix]['acc'])):
            print('applying ' + str(suffix) + "_" + str(calib_n) + ' calibration')
            sk = skew[suffix]['acc'][calib_n]    
            sc = scale[suffix]['acc'][calib_n]    
            bi = bias[suffix]['acc'][calib_n]    
            #temp_acc.append(str(suffix) + "_" + str(calib_n))
            temp_acc.append( calibrate(acc_data[acc], sk, sc, bi) )

            sk = skew[suffix]['gyr'][calib_n]    
            sc = scale[suffix]['gyr'][calib_n]    
            bi = bias[suffix]['gyr'][calib_n]  
            #temp_acc.append(str(suffix) + "_" + str(calib_n))
            temp_gyr.append(calibrate(gyr_data[gyr], sk, sc, bi) )

        temp_acc = np.array(temp_acc)
        temp_gyr = np.array(temp_gyr)

        calib_acc_data[acc][suffix] = temp_acc
        calib_gyr_data[gyr][suffix] = temp_gyr
    
#%% Plot the norm of the acc for the calibrated data
from matplotlib import pyplot as plt
from utils import num2leg

suffixes = ['base', 'optBias', 'minAccBiases'] 
plt.close('all')

nbins = 30
rwidth = 0.8
alpha = 0.4

for acc_file, gyr_file in zip(calib_acc_data.keys(), calib_gyr_data.keys()):
    print(acc_file, gyr_file)
    
    fig, axs = plt.subplots(1, len(suffixes), figsize=(15,5), sharex='all')
    
    for i, suffix in enumerate(calib_acc_data[acc_file].keys()):
        print(i, suffix)
        
        for j, (calib_acc, calib_gyr) in enumerate(zip(
                calib_acc_data[acc_file][suffix],
                calib_gyr_data[gyr_file][suffix]
                )):
            print('calib run '+str(j))
            norm_acc = np.linalg.norm(calib_acc[:,1:4], axis=1)
            lab = r'avg='+num2leg(norm_acc.mean()) + ' $\pm$ '+num2leg(norm_acc.std())
            axs[i].hist(norm_acc, bins=nbins, alpha=alpha, rwidth=rwidth, label=lab)
        axs[i].legend(prop={'size':8});
        axs[i].set_xlabel('||acc||')
        axs[i].set_ylabel('# occurences')
        axs[i].title.set_text(suffix)
        axs[i].axvline(x=g_mag, c='k', ls='--')
    plt.savefig(plot_path+'/'+acc_file.split('/')[-1]+'_gnorm.png')
    #input('wait')


#%% Plot Calibrated Gyro biases
from matplotlib import pyplot as plt
from utils import num2leg

suffixes = ['base', 'optBias', 'minAccBiases'] 
plt.close('all')

nbins = 30
rwidth = 0.8
alpha = 0.4

for acc_file, gyr_file in zip(calib_acc_data.keys(), calib_gyr_data.keys()):
    print(acc_file, gyr_file)
    
    fig, axs = plt.subplots(3, len(suffixes), figsize=(15,15), sharex='all', sharey='all')
    
    for i, suffix in enumerate(calib_acc_data[acc_file].keys()):
        print(i, suffix)
        
        for j, (calib_acc, calib_gyr) in enumerate(zip(
                calib_acc_data[acc_file][suffix],
                calib_gyr_data[gyr_file][suffix]
                )):
            print('calib run '+str(j))
            
            for k in range(3):
                lab = r'avg='+num2leg(calib_gyr[:,k+1].mean()) +' $\pm$ '+ num2leg(calib_gyr[:,k+1].std())
                
                axs[k][i].hist(calib_gyr[:,k+1], alpha=alpha, label=lab)
                axs[k][i].legend(prop={'size':8});
                axs[k][i].set_ylabel('# occurences')
                axs[k][i].set_xlabel('Ang. Vel. (radians/s)')
                axs[k][i].title.set_text(suffix)
        
        for k in range(3):
            axs[k][i].axvline(x=0, ls='--', c='k')
            
    plt.savefig(plot_path+'/'+acc_file.split('/')[-1]+'_static_gyro_biases.png')
    #input('wait')

#%% Plot orientation obtained from integrating angular velocities for the calibrated data
from matplotlib import pyplot as plt
from utils import integrateOrientations

suffixes = ['base', 'optBias', 'minAccBiases'] 
plt.close('all')

alpha = 0.4

lab = ['x', 'y', 'z']

for acc_file, gyr_file in zip(calib_acc_data.keys(), calib_gyr_data.keys()):
    print(acc_file, gyr_file)
    
    fig, axs = plt.subplots(3, len(suffixes), figsize=(15,15), sharex='all', sharey='row')
    
    for i, suffix in enumerate(calib_acc_data[acc_file].keys()):
        print(i, suffix)
        
        for j, (calib_acc, calib_gyr) in enumerate(zip(
                calib_acc_data[acc_file][suffix],
                calib_gyr_data[gyr_file][suffix]
                )):
            print('calib run '+str(j))
            ori = integrateOrientations(calib_gyr)
            
            for k in range(3):
                axs[k][i].plot(ori[:,0], ori[:,k+1], alpha=alpha, label='calib - '+str(j))
                axs[k][i].set_xlabel('time (s)')
                axs[k][i].set_ylabel('Angle ' + lab[k] + ' (radians)')
        for k in range(3):
            axs[k][i].axhline(y=0, c='k', ls='--', label='reference')
            axs[k][i].legend(prop={'size':8});
                
        axs[0][i].title.set_text(suffix)
    plt.savefig(plot_path+'/'+acc_file.split('/')[-1]+'_orientation.png')
    #input('wait')

#%% Apply gravity compensation with Madgwick filter
from utils import gravity_compensate

T_imu2mdg = np.array( [ [0, 0, 1], [0, -1, 0], [1, 0, 0] ] )
gcomp_acc_data = dict()

# apply gravity comp on uncalibrated data
for acc_file, gyr_file in zip(acc_data.keys(), gyr_data.keys()):
    gcomp_acc_data[acc_file] = gravity_compensate(
            acc_data[acc_file][:,0],
            acc_data[acc_file][:,1:4]*nominal_acc_scale,
            gyr_data[gyr_file][:,1:4]*nominal_gyr_scale,
            g_mag, T_imu2mdg, plot_path)

#%% Apply gravity compensation on calibrated data

suffixes = ['base', 'optBias', 'minAccBiases'] 
T_imu2mdg = np.array( [ [0, 0, 1], [0, -1, 0], [1, 0, 0] ] )
gcomp_calib_acc_data = dict()

for acc_file, gyr_file in zip(calib_acc_data.keys(), calib_gyr_data.keys()):
    print(acc_file, gyr_file)
    gcomp_calib_acc_data[acc_file] = dict()
    
    for i, suffix in enumerate(calib_acc_data[acc_file].keys()):
        print(i, suffix)
        
        gcomp_calib_acc_data[acc_file][suffix] = list()
        for j, (calib_acc, calib_gyr) in enumerate(zip(
                calib_acc_data[acc_file][suffix],
                calib_gyr_data[gyr_file][suffix]
                )):
            print('calib run '+str(j))
            
            gcomp_calib_acc_data[acc_file][suffix].append(
                    gravity_compensate(
                            calib_acc[:,0],
                            calib_acc[:,1:4],
                            calib_gyr[:,1:4],
                            g_mag, T_imu2mdg
                            )
                    )
                    
        gcomp_calib_acc_data[acc_file][suffix] = np.array(gcomp_calib_acc_data[acc_file][suffix])
#%% Plot the gravity compensated accelerations

from matplotlib import pyplot as plt
from utils import integrateOrientations, align_yaxis
from scipy.integrate import cumtrapz

suffixes = ['base', 'optBias', 'minAccBiases'] 
plt.close('all')

alpha = 0.4
lab = ['x', 'y', 'z']
start_time = 10;

for acc_file in gcomp_calib_acc_data.keys():
    print(acc_file)
    
    fig, axs = plt.subplots(3, len(suffixes), 
                            figsize=(20,20), 
                            sharex='all', 
                            sharey='row'
                            )
    
    for i, suffix in enumerate(gcomp_calib_acc_data[acc_file].keys()):
        print(i, suffix)
        
        for j, (calib_acc) in enumerate(gcomp_calib_acc_data[acc_file][suffix]):
            print('calib run '+str(j))
            
            idx = calib_acc[:,0] >= start_time
            x = calib_acc[idx,0]
            y = calib_acc[idx,1:4]
            v = cumtrapz(y, x, axis=0)
            
            for k in range(3):
                axs[k][i].plot(x[1::], v[:,k], alpha=alpha, label='calib - '+str(j))
                axs[k][i].set_xlabel('time (s)')
                axs[k][i].set_ylabel('Vel ' + lab[k] + ' (m/s)')
        
        for k in range(3):
            idx = gcomp_acc_data[acc_file][:,0] >= start_time
            x = gcomp_acc_data[acc_file][idx,0]
            y = gcomp_acc_data[acc_file][idx,1:4]
            v = cumtrapz(y, x, axis=0)
            
            #ax2 = axs[k][i].twinx()
            axs[k][i].plot(x[1::], v[:,k], c='k', ls=':', label='uncalibrated')
            #align_yaxis(axs[k][i], 0, ax2, 0)
            
            axs[k][i].axhline(y=0, c='k', ls='--', label='reference')
            axs[k][i].legend(prop={'size':8});
            
        axs[0][i].title.set_text(suffix)    
    plt.savefig(plot_path+'/'+acc_file.split('/')[-1]+'_velocities.png')
    #break