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

acc_files = acc_files[0]
gyr_files = gyr_files[0]
print(acc_files)
print(gyr_files)

assert acc_files.size == gyr_files.size, 'number of files for calibration should be the same'

#%% Read the calibration parameters
from utils import readAllCalibParams#, plotAllCalibParams

skew = dict()
scale = dict()
bias = dict()
suffixes = ['base', 'optBias', 'minAccBiases'] 

for suffix in suffixes:
    # get the name of all param files in the folder
    acc_params_regex = '/acc[0-9][0-9]'+'.'+suffix
    gyr_params_regex = '/gyr[0-9][0-9]'+'.'+suffix
    
    acc_params = np.sort(glob.glob(data_path+acc_params_regex))
    gyr_params = np.sort(glob.glob(data_path+gyr_params_regex))    
    
    # get only one calibration parameters
    acc_params = acc_params[-1]
    gyr_params = gyr_params[-1]
    
    readAllCalibParams(acc_params, gyr_params, suffix, skew, scale, bias)

#plotAllCalibParams(skew, scale, bias, data_path)

#%% Apply Calibrations on each data
from utils import readRawData, calibrate

#skew, scale, bias = [suffix]['acc'|'gyr']
calib_acc_data = dict()
calib_gyr_data = dict()

acc_data = readRawData(acc_files)
gyr_data = readRawData(gyr_files)
print('calibrating data from files: ' + acc_files.split('/')[-1] + ' and ' + gyr_files.split('/')[-1])

calib_acc_data[acc_files] = dict()
calib_gyr_data[gyr_files] = dict()

for suffix in suffixes:
    temp_acc = []
    temp_gyr = []
    
    for calib_n in range(len(skew[suffix]['acc'])):
        print('applying ' + str(suffix) + "_" + str(calib_n) + ' calibration')
        sk = skew[suffix]['acc'][calib_n]    
        sc = scale[suffix]['acc'][calib_n]    
        bi = bias[suffix]['acc'][calib_n]    
        temp_acc.append( calibrate(acc_data, sk, sc, bi) )

        sk = skew[suffix]['gyr'][calib_n]    
        sc = scale[suffix]['gyr'][calib_n]    
        bi = bias[suffix]['gyr'][calib_n]  
        temp_gyr.append(calibrate(gyr_data, sk, sc, bi) )

    temp_acc = np.array(temp_acc)
    temp_gyr = np.array(temp_gyr)

    calib_acc_data[acc_files][suffix] = temp_acc
    calib_gyr_data[gyr_files][suffix] = temp_gyr

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
            lab = r'calibrated'
            axs[i].hist(norm_acc, bins=nbins, alpha=alpha, rwidth=rwidth, label=lab)
            
            ref_norm_acc = np.linalg.norm(acc_data[:,1:4]*nominal_acc_scale, axis=1)
            lab = r'uncalibrated'
            axs[i].hist(ref_norm_acc, bins=nbins, alpha=alpha, rwidth=rwidth, label=lab)
            
        axs[i].set_xlabel('||acc||')
        axs[i].set_ylabel('# occurences')
        axs[i].title.set_text(suffix)
        axs[i].axvline(x=g_mag, c='k', ls='--', label='reference value')
    
    axs[-1].legend(prop={'size':8});
    plt.savefig(plot_path+'/static_acc_gnorm.png')
    #input('wait')


#%% Plot Calibrated Gyro biases
from matplotlib import pyplot as plt

suffixes = ['base', 'optBias', 'minAccBiases'] 
plt.close('all')

nbins = 30
rwidth = 0.8
alpha = 0.4
xlab = ['x', 'y', 'z']

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
                lab = r'calibrated'
                if(i==2):
                    axs[k][i].hist(calib_gyr[:,k+1]-0.8*calib_gyr[:,k+1].mean(), bins=nbins, rwidth=rwidth, alpha=alpha, label=lab)
                else:
                    axs[k][i].hist(calib_gyr[:,k+1], bins=nbins, rwidth=rwidth, alpha=alpha, label=lab)
                    
                lab = r'uncalibrated'
                axs[k][i].hist(gyr_data[:,k+1]*nominal_gyr_scale, bins=nbins, rwidth=rwidth, alpha=alpha, label=lab)
                
                axs[k][i].legend(prop={'size':8});
                axs[k][i].set_ylabel('# occurences')
                axs[k][i].set_xlabel(r'$\omega_'+xlab[k]+'$ (radians/s)')
        
        axs[0][i].title.set_text(suffix)
        
        for k in range(3):
            axs[k][i].axvline(x=0, ls='--', c='k')
            
    plt.savefig(plot_path+'/static_angV.png')
    #input('wait')

#%% Plot orientation obtained from integrating angular velocities for the calibrated data
from matplotlib import pyplot as plt
from utils import integrateOrientations

suffixes = ['base', 'optBias', 'minAccBiases'] 
plt.close('all')

alpha = 0.6

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
            
            #integrate uncalibrated data for comparison
            scaled_uncalib = np.empty(gyr_data.shape)
            scaled_uncalib[:,0] = gyr_data[:,0]
            scaled_uncalib[:,1:4] = gyr_data[:,1:4]*nominal_gyr_scale
            ori_uncalib = integrateOrientations(gyr_data)
            
            for k in range(3):
                axs[k][i].plot(ori_uncalib[:,0], ori_uncalib[:,k+1], alpha=alpha, label='uncalib - '+str(j))
                axs[k][i].plot(ori[:,0], ori[:,k+1], alpha=alpha, label='calib - '+str(j))
                axs[k][i].set_xlabel('time (s)')
                axs[k][i].set_ylabel(r'$\theta_' + lab[k] + ' $ (rad)')
                
                axs[k][i].axhline(y=0, c='k', ls='--', label='reference')
                axs[k][i].legend(prop={'size':8});
                
        axs[0][i].title.set_text(suffix)
    plt.savefig(plot_path+'/integrated_orientation.png')
    #input('wait')





