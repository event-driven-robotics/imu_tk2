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

sys.path.insert(0, repos_path)
sys.path.insert(0, bimvee_path)
sys.path.insert(0, wp5_slam_path)
sys.path.insert(0, imu_tk2_path+'/python')

if(not os.path.isdir(plot_path)):
    print('creating folder for plots')
    os.mkdir(plot_path)
    
raw_acc_files_regex = '/dynamic_acc[0-9][0-9]'
raw_gyr_files_regex = '/dynamic_gyr[0-9][0-9]'

g_mag = 9.805622
nominal_gyr_scale = 250 * np.pi / (2.0 * 180.0 * 16384.0)
nominal_acc_scale = g_mag/16384.0
suffixes = ['base', 'optGyrBias'] 

start_time = 10
end_time = 100

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
gmat = dict()

for suffix in suffixes:
    # get the name of all param files in the folder
    # get the name of all param files in the folder
    acc_params_regex = '/acc[0-9][0-9]'+'.'+suffix
    gyr_params_regex = '/gyr[0-9][0-9]'+'.'+suffix
    
    acc_params = np.sort(glob.glob(data_path+acc_params_regex))
    gyr_params = np.sort(glob.glob(data_path+gyr_params_regex))

    readAllCalibParams(acc_params, gyr_params, suffix, skew, scale, bias, gmat)

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
            gi = gmat[suffix]['gyr'][calib_n]  
            #temp_acc.append(str(suffix) + "_" + str(calib_n))
            temp_gyr.append(calibrate(gyr_data[gyr], sk, sc, bi, gi, acc_data[acc]) )

        temp_acc = np.array(temp_acc)
        temp_gyr = np.array(temp_gyr)

        calib_acc_data[acc][suffix] = temp_acc
        calib_gyr_data[gyr][suffix] = temp_gyr

#%% Compute the orientation from the angular velocities
from utils import integrateOrientations, cropTime

orientations = dict()

for gyr_file in calib_gyr_data:
    print('integrating orientations for: \n' + gyr_file)
    
    print('Uncalibrated')
    orientations[gyr_file] = dict()
    
    #limit integration time for having some consistency among benchmarks
    orientations[gyr_file]['uncalibrated'] = integrateOrientations(
            cropTime(gyr_data[gyr_file], start_time, end_time),
            nominal_gyr_scale)

    for suffix in suffixes:
        print(suffix)
        orientations[gyr_file][suffix] = []
        
        for j, calib_gyr in enumerate(calib_gyr_data[gyr_file][suffix]):
            print('Calibrated run '+str(j))
            orientations[gyr_file][suffix].append(
                    integrateOrientations(cropTime(calib_gyr, start_time, end_time))
                    )
        
        orientations[gyr_file][suffix] = np.array(orientations[gyr_file][suffix])          

#%% Apply gravity compensation with Madgwick filter
from utils import gravity_compensate

T_imu2mdg = np.array( [ [0, 0, 1], [0, -1, 0], [1, 0, 0] ] )
gcomp_acc_data = dict()

# apply gravity comp on uncalibrated data
for acc_file, gyr_file in zip(acc_data, gyr_data):
    gcomp_acc_data[acc_file] = gravity_compensate(
            acc_data[acc_file][:,0],
            acc_data[acc_file][:,1:4]*nominal_acc_scale,
            gyr_data[gyr_file][:,1:4]*nominal_gyr_scale,
            g_mag, T_imu2mdg, plot_path)

#%% Apply gravity compensation on calibrated data

T_imu2mdg = np.array( [ [0, 0, 1], [0, -1, 0], [1, 0, 0] ] )
gcomp_calib_acc_data = dict()

for acc_file, gyr_file in zip(calib_acc_data, calib_gyr_data):
    print(acc_file, gyr_file)
    gcomp_calib_acc_data[acc_file] = dict()
    
    for suffix in suffixes:
        print(suffix)
        
        gcomp_calib_acc_data[acc_file][suffix] = list()
        for calib_acc, calib_gyr in zip(
                calib_acc_data[acc_file][suffix],
                calib_gyr_data[gyr_file][suffix]
                ):
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
        
#%% Integrate the acc to get the velocities
from utils import integrateVelocities, cropTime

velocities= dict()

for acc_file in gcomp_calib_acc_data:
    print(acc_file)
    velocities[acc_file] = dict()
    
    velocities[acc_file]['uncalibrated'] = integrateVelocities(
            cropTime(gcomp_acc_data[acc_file],start_time, end_time))
    
    for suffix in suffixes:
        print(suffix)
        
        velocities[acc_file][suffix] = []
        
        for j, (calib_acc) in enumerate(gcomp_calib_acc_data[acc_file][suffix]):
            print('calib run '+str(j))
            
            velocities[acc_file][suffix].append(
                    integrateVelocities(cropTime(calib_acc, start_time, end_time)))
                    
        velocities[acc_file][suffix] = np.array(
                velocities[acc_file][suffix])

#%% Caculate the errors of the calibrations
from numpy.linalg import norm
from utils import MakeOrientationContinuous

errors_gyr = dict()
#TODO: make a function to get the error every a time interval
    
for gyr_file in calib_gyr_data:
    print(gyr_file)
    
    print('Uncalibrated')
    errors_gyr[gyr_file] = dict()
    ori = MakeOrientationContinuous(orientations[gyr_file]['uncalibrated'])
    errors_gyr[gyr_file]['uncalibrated'] = norm(ori[-1,1:4])
    
    for suffix in suffixes:
        print(suffix)
        errors_gyr[gyr_file][suffix] = []
        
        for orientation in orientations[gyr_file][suffix]:
            ori = MakeOrientationContinuous(orientation)
            errors_gyr[gyr_file][suffix].append(norm(ori[-1,1:4]))
        
        errors_gyr[gyr_file][suffix] = np.array(errors_gyr[gyr_file][suffix])    
        
        
errors_acc = dict()

for acc_file in gcomp_calib_acc_data:
    print(acc_file)
    errors_acc[acc_file] = dict()
    
    errors_acc[acc_file]['uncalibrated'] = norm(velocities[acc_file]['uncalibrated'][-1,1:4])
    
    for suffix in suffixes:
        print(suffix)
        
        errors_acc[acc_file][suffix] = []
        
        for vel in velocities[acc_file][suffix]:
            errors_acc[acc_file][suffix].append(norm(vel[-1,1:4]))
                    
        errors_acc[acc_file][suffix] = np.array(errors_acc[acc_file][suffix])        
        
#%% Plot a scatter
import re
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D as mlines
from matplotlib.patches import Patch as mpatches

plt.close('all')

alpha = 1
mss = ['s','x']
plt.figure(figsize=(20,20))

for acc_file, gyr_file in zip(errors_acc, errors_gyr):
    print(acc_file, gyr_file)
        
    plt.plot(abs(errors_acc[acc_file]['uncalibrated']),
             abs(errors_gyr[gyr_file]['uncalibrated']),
             marker='d')
    
    #plot the calibrated ones        
    for suffix, ms in zip(suffixes, mss):
        print(suffix)
        
        for ea, eg in zip(errors_acc[acc_file][suffix], errors_gyr[gyr_file][suffix]):
            lab = re.findall(r'\d+', acc_file.split('/')[-1])[0]
            plt.plot(abs(ea), abs(eg), alpha=alpha, marker=ms)
            plt.annotate(lab, (abs(ea), abs(eg)))
            
plt.xlabel('|Orientation error|')
plt.ylabel('|Lin. Velocity error|')

l0 = mlines([],[], marker='d', ls='', label='uncalibrated')
l1 = mlines([],[], marker='s', ls='', label= suffixes[0])
l2 = mlines([],[], marker='x', ls='', label= suffixes[1])
handles1 = [l0,l1,l2]
legend1 = plt.legend(handles=handles1, prop={'size':12});

colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
handles2 = []
for i, file in enumerate(acc_params):
    handles2.append(mpatches(color=colours[i], label='trial '+re.findall(r'\d+', file.split('/')[-1])[0]))
            
plt.legend(handles=handles2, loc=4)
plt.gca().add_artist(legend1)
plt.yscale('log')
plt.xscale('log')
plt.savefig(plot_path+'/errors.png')
          
#%% Per calibration trial average
from utils import setBoxColors
perCalib_acc_error = dict()
perCalib_gyr_error = dict()

trials = ['#'+re.findall(r'\d+', file.split('/')[-1])[0] for file in acc_params]
pos = 3*(np.linspace(1,len(trials),len(trials)))-1
colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.close('all')
fig, ax = plt.subplots(1, 2, figsize=(20,10))#, sharex='all', sharey='all')

for i, suffix in enumerate(suffixes):
    print(i)
    perCalib_acc_error[suffix] = []
    perCalib_gyr_error[suffix] = []
    
    for acc_file, gyr_file in zip(errors_acc, errors_gyr):    
        perCalib_acc_error[suffix].append(errors_acc[acc_file][suffix])
        perCalib_gyr_error[suffix].append(errors_gyr[gyr_file][suffix])
    
    perCalib_acc_error[suffix] = np.array(perCalib_acc_error[suffix])
    perCalib_gyr_error[suffix] = np.array(perCalib_gyr_error[suffix])
    bp = ax[0].boxplot(perCalib_acc_error[suffix], widths = 0.6, positions=pos+i)
    setBoxColors(bp, colours[i])
    bp = ax[1].boxplot(perCalib_gyr_error[suffix], widths = 0.6, positions=pos+i)
    setBoxColors(bp, colours[i])

# plot the uncalibrated data
uncalib_acc_error = []
uncalib_gyr_error = []
for acc_file, gyr_file in zip(errors_acc, errors_gyr):    
    uncalib_acc_error.append(errors_acc[acc_file]['uncalibrated'])
    uncalib_gyr_error.append(errors_gyr[gyr_file]['uncalibrated'])
uncalib_acc_error = np.array(uncalib_acc_error)
uncalib_gyr_error = np.array(uncalib_gyr_error)
    
bp = ax[0].boxplot(uncalib_acc_error, widths = 0.6, positions=[-1])
setBoxColors(bp, 'k')
bp = ax[1].boxplot(uncalib_gyr_error, widths = 0.6, positions=[-1])
setBoxColors(bp, 'k')


plt.sca(ax[0])
plt.xticks( np.concatenate(([-1],pos+0.5)), ['uncalib']+trials)  
ax[0].set_xlabel('calibration trial')   
ax[0].set_ylabel('lin. vel. error')

plt.sca(ax[1])
plt.xticks(np.concatenate(([-1],pos+0.5)), ['uncalib']+trials)     
ax[1].set_xlabel('calibration trial')   
ax[1].set_ylabel('ori. error')     

handles = []
for i, suffix in enumerate(suffixes):
    handles.append(mpatches(color=colours[i], label=suffix))
ax[1].legend(handles=handles)    

fig.savefig(plot_path + '/compare_calibration_trials.pdf')

#%% Per calibration (suffix) average
from utils import setBoxColors
perTest_acc_error = dict()
perTest_gyr_error = dict()

tests = ['#'+re.findall(r'\d+', file.split('/')[-1])[0] for file in acc_data]
pos = 3*(np.linspace(1,len(tests),len(tests)))-1
colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.close('all')
fig, ax = plt.subplots(1, 2, figsize=(20,10))#, sharex='all', sharey='all')

for i, suffix in enumerate(suffixes):
    print(i)
    perTest_acc_error[suffix] = []
    perTest_gyr_error[suffix] = []
    
    for acc_file, gyr_file in zip(errors_acc, errors_gyr):    
        perTest_acc_error[suffix].append(errors_acc[acc_file][suffix])
        perTest_gyr_error[suffix].append(errors_gyr[gyr_file][suffix])
    
    perTest_acc_error[suffix] = np.array(perTest_acc_error[suffix])
    perTest_gyr_error[suffix] = np.array(perTest_gyr_error[suffix])
    bp = ax[0].boxplot(perTest_acc_error[suffix].T, widths = 0.6, positions=pos+i)
    setBoxColors(bp, colours[i])
    bp = ax[1].boxplot(perTest_gyr_error[suffix].T, widths = 0.6, positions=pos+i)
    setBoxColors(bp, colours[i])

# plot the uncalibrated data
uncalib_acc_error = []
uncalib_gyr_error = []
for acc_file, gyr_file in zip(errors_acc, errors_gyr):    
    uncalib_acc_error.append(errors_acc[acc_file]['uncalibrated'])
    uncalib_gyr_error.append(errors_gyr[gyr_file]['uncalibrated'])
uncalib_acc_error = np.array(uncalib_acc_error)
uncalib_gyr_error = np.array(uncalib_gyr_error)
    
bp = ax[0].boxplot(uncalib_acc_error.T, widths = 0.6, positions=[-1])
setBoxColors(bp, 'k')
bp = ax[1].boxplot(uncalib_gyr_error.T, widths = 0.6, positions=[-1])
setBoxColors(bp, 'k')


plt.sca(ax[0])
plt.xticks( np.concatenate(([-1],pos+0.5)), ['uncalib']+tests)  
ax[0].set_xlabel('test #')   
ax[0].set_ylabel('lin. vel. error')

plt.sca(ax[1])
plt.xticks(np.concatenate(([-1],pos+0.5)), ['uncalib']+tests)     
ax[1].set_xlabel('test #')   
ax[1].set_ylabel('ori. error')     

handles = []
for i, suffix in enumerate(suffixes):
    handles.append(mpatches(color=colours[i], label=suffix))
ax[1].legend(handles=handles)    

fig.savefig(plot_path + '/compare_calibration_modes.pdf')