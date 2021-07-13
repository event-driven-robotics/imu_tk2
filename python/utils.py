#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 14:58:19 2021

@author: Leandro de Souza Rosa <leandro.desouzarosa@iit.it>
    
"""

import numpy as np


def readCalibParams(file):
    #print(file)
    with open(file, 'r') as f:
        # read first 3 lines for skew
        temp = []
        for i in range(3):
            #skip empty lines
            line = []
            while(len(line)==0):
                line = next(f).replace('\n', '').split()
                #print('reading :', line)
                
            temp = temp + line
        skew = np.array([np.double(n) for n in temp]).reshape((3,3))
        #print(skew)
        
        # read second 3 lines for scale
        temp = []
        for i in range(3):
            #skip empty lines
            line = []
            while(len(line)==0):
                line = next(f).replace('\n', '').split()
                #print('reading :', line)
            temp = temp + line
        scale = np.array([np.double(n) for n in temp]).reshape((3,3))
        #print(scale)
        
        # read third 3 lines for bias
        temp = []
        for i in range(3):
            #skip empty lines
            line = []
            while(len(line)==0):
                line = next(f).replace('\n', '').split()
                #print('reading :', line)
            temp = temp + line
        bias = np.array([np.double(n) for n in temp]).reshape((3,1))
        #print(bias)
        
    return skew, scale, bias

def readAllCalibParams(acc_params, gyr_params, suffix, skew, scale, bias):
  
    temp_skew_acc = []
    temp_scale_acc = []
    temp_bias_acc = []
    temp_skew_gyr = []
    temp_scale_gyr = []
    temp_bias_gyr = []
    
    assert acc_params.size == gyr_params.size, "Sizes do not match"
    if acc_params.size == 1:
        acc_params = [acc_params]
        gyr_params = [gyr_params]
    
    for acc, gyr in zip(acc_params, gyr_params):
        
        sk, sc, bi = readCalibParams(acc)
        temp_skew_acc.append(sk)
        temp_scale_acc.append(sc)
        temp_bias_acc.append(bi)
        
        sk, sc, bi = readCalibParams(gyr)
        temp_skew_gyr.append(sk)
        temp_scale_gyr.append(sc)
        temp_bias_gyr.append(bi)
    
    skew[suffix] = dict()
    scale[suffix] = dict()
    bias[suffix] = dict()
    
    skew[suffix]['acc'] = np.array(temp_skew_acc)
    scale[suffix]['acc'] = np.array(temp_scale_acc)
    bias[suffix] ['acc'] = np.array(temp_bias_acc)
    skew[suffix]['gyr'] = np.array(temp_skew_gyr)
    scale[suffix]['gyr'] = np.array(temp_scale_gyr)
    bias[suffix]['gyr'] = np.array(temp_bias_gyr)
    

def plotAllCalibParams(skew, scale, bias, path):
    nbins = 20
    rwidth = 0.8
    alpha = 0.7
    
    from matplotlib import pyplot as plt
    plt.close('all')
    for exp in skew:
        for sensor in skew[exp]:
            print(exp, sensor)
            
            fig, axs = plt.subplots(3, 3, figsize=(15,15))
            fig.suptitle(exp+'-'+sensor)
            
            for ax0 in axs: 
                for ax1 in ax0:
                    ax1.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
            
            #plot skews
            axs[0,0].hist(skew[exp][sensor][:,0,1], bins=nbins, rwidth=rwidth, alpha = alpha)
            axs[0,1].hist(skew[exp][sensor][:,0,2], bins=nbins, rwidth=rwidth, alpha = alpha)
            axs[0,2].hist(skew[exp][sensor][:,1,2], bins=nbins, rwidth=rwidth, alpha = alpha)      
            axs[0,0].set_ylabel('# occurences')
            axs[0,0].set_xlabel('skew xy')
            axs[0,1].set_xlabel('skew xz')
            axs[0,2].set_xlabel('skew yz')
            
            #plot scales
            axs[1,0].hist(scale[exp][sensor][:,0,0], bins=nbins, rwidth=rwidth, alpha = alpha)
            axs[1,1].hist(scale[exp][sensor][:,1,1], bins=nbins, rwidth=rwidth, alpha = alpha)
            axs[1,2].hist(scale[exp][sensor][:,2,2], bins=nbins, rwidth=rwidth, alpha = alpha)       
            axs[1,0].set_ylabel('# occurences')
            axs[1,0].set_xlabel('scale x')
            axs[1,1].set_xlabel('scale y')
            axs[1,2].set_xlabel('scale z')
            
            #plot biases
            axs[2,0].hist(bias[exp][sensor][:,0], bins=nbins, rwidth=rwidth, alpha = alpha)
            axs[2,1].hist(bias[exp][sensor][:,1], bins=nbins, rwidth=rwidth, alpha = alpha)
            axs[2,2].hist(bias[exp][sensor][:,2], bins=nbins, rwidth=rwidth, alpha = alpha)       
            axs[2,0].set_ylabel('# occurences')
            axs[2,0].set_xlabel('bias x')
            axs[2,1].set_xlabel('bias y')
            axs[2,2].set_xlabel('bias z')
            #fig.tight_layout()
            
            plt.savefig(path+'/'+exp+'_'+sensor+'.png')
    
def readRawData(file):
    temp = []
    
    with open(file, 'r') as f:
        for line in f:
            temp.append( [np.double(v) for v in line.replace('\n', '').split()] )

    return np.array(temp)
            
        
def calibrate(data, skew, scale, bias):
    n_samples, _ = data.shape
    
    # allocate array for calibrated data and copy timestamp
    calib_data = np.empty(data.shape)
    calib_data[:,0] = data[:,0]
    
    for sample in range(n_samples):
        calib_data[sample, 1:4] = skew@scale@(data[sample, 1:4]-bias.flatten())
    
    return calib_data
        
        
def integrateOrientations(gyr):
    from pyquaternion import Quaternion
    from scipy.spatial.transform import Rotation
    
    # allocate resulting array 
    ret = np.empty(gyr.shape)
    ret[:,0] = gyr[:,0]
    
    # get delta times
    dts = np.diff(gyr[:,0])
    dts = np.concatenate( (dts, [dts[-1]]) )
 
    assert len(dts)==len(gyr)
        
    q = Quaternion()
    
    for i, (dt, w) in enumerate(zip(dts, gyr[:,1:4])):
        q.integrate(w, dt)
        rot = Rotation.from_matrix(q.rotation_matrix)
        #print(w,' - ',  q.angle, ' - ',  q)
        #print(rot.as_euler('xyz') , '\n-------\n')
        ret[i,1:4] = rot.as_euler('xyz')
        
    return ret
        
def num2leg(n, p=3):
    return str(np.round( 10**p * n ) / 10**p )
        
        
        
        
        
        
        
        
        
        
        