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
        
         # read third 3 lines for G
        temp = []
        for i in range(3):
            #skip empty lines
            line = []
            while(len(line)==0):
                
                try:
                    line = next(f).replace('\n', '').split()
                except(StopIteration):
                    #print('EOF')
                    return skew, scale, bias
            temp = temp + line
        g = np.array([np.double(n) for n in temp]).reshape((3,3))
        #print(bias)
        
    return skew, scale, bias, g

def readAllCalibParams(acc_params, gyr_params, suffix, skew, scale, bias, gmat):
  
    temp_skew_acc = []
    temp_scale_acc = []
    temp_bias_acc = []
    temp_skew_gyr = []
    temp_scale_gyr = []
    temp_bias_gyr = []
    #temp_gmat_acc = []
    temp_gmat_gyr = []
    
    assert acc_params.size == gyr_params.size, "Sizes do not match"
    if acc_params.size == 1:
        acc_params = [acc_params]
        gyr_params = [gyr_params]
    
    for acc, gyr in zip(acc_params, gyr_params):
        
        sk, sc, bi = readCalibParams(acc)
        temp_skew_acc.append(sk)
        temp_scale_acc.append(sc)
        temp_bias_acc.append(bi)
        #temp_gmat_acc.append(gi)
        
        sk, sc, bi, gi = readCalibParams(gyr)
        temp_skew_gyr.append(sk)
        temp_scale_gyr.append(sc)
        temp_bias_gyr.append(bi)
        temp_gmat_gyr.append(gi)
        
    skew[suffix] = dict()
    scale[suffix] = dict()
    bias[suffix] = dict()
    gmat[suffix] = dict()
    
    skew[suffix]['acc'] = np.array(temp_skew_acc)
    scale[suffix]['acc'] = np.array(temp_scale_acc)
    bias[suffix] ['acc'] = np.array(temp_bias_acc)
    #gmat[suffix]['acc'] = np.array(temp_gmat_acc)
    
    skew[suffix]['gyr'] = np.array(temp_skew_gyr)
    scale[suffix]['gyr'] = np.array(temp_scale_gyr)
    bias[suffix]['gyr'] = np.array(temp_bias_gyr)
    gmat[suffix]['gyr'] = np.array(temp_gmat_gyr)
    

def plotAllCalibParams(skew, scale, bias, gmat, path):
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
            
        
def calibrate(data, skew, scale, bias, gmat = [], other_data = []):
    n_samples, _ = data.shape
    
    # allocate array for calibrated data and copy timestamp
    calib_data = np.empty(data.shape)
    calib_data[:,0] = data[:,0]
    
    if(gmat == []):
        for sample in range(n_samples):
            calib_data[sample, 1:4] = skew@scale@(data[sample, 1:4] - bias.flatten())
    else:
        for sample in range(n_samples):
            calib_data[sample, 1:4] = scale@(skew@data[sample, 1:4] - bias.flatten()-gmat@other_data[sample, 1:4])

    return calib_data      
        
def integrateOrientations(gyr, scale=1):
    from pyquaternion import Quaternion
    from quaternion import rotation
    
    # allocate resulting array 
    ret = np.empty(gyr.shape)
    ret[:,0] = gyr[:,0]
    
    # get delta times
    dts = np.diff(gyr[:,0])
    dts = np.concatenate( (dts, [dts[-1]]) )
 
    assert len(dts)==len(gyr)
        
    q = Quaternion()
    rots = np.empty((len(dts), 4))
    
    for i, (dt, w) in enumerate(zip(dts, gyr[:,1:4])):
        q.integrate(w*scale, dt)
        rots[i,:] = q.elements
    
    rot = rotation.from_wxyz_quat(rots)
    ret[:,1:4] = rot.as_euler('xyz')
    return ret
        
def num2leg(n, p=3):
    return str(np.round( 10**p * n ) / 10**p )
        
        

def rotation_matrix_from_vectors(vec1, vec2):
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix        
        

def gravity_compensate(ts, acc_data, gyr_data, g, Tmat, plot_path='./', suffix=''):
    import madgwickahrs 
    from matplotlib import pyplot as plt
    from quaternion import rotation
    
    T = rotation.from_matrix(Tmat)

    imu = dict()
    imu['ts'] = ts
    imu['acc'] = T.apply(acc_data)
    imu['angV'] = T.apply(gyr_data)

    kwargs = dict()
    kwargs['gravityMag'] = g
    #kwargs['imuErrorGyr'] = imu['angV'].mean(axis=0)
    kwargs['beta'] = 1
    kwargs['verbose'] = True
    
    mdg = madgwickahrs.MadgwickAHRS(**kwargs)   
    temp = T.inv().apply(mdg.gravityCompensate(imu))
    mdg.reset()
    
    # get just the data within the interval
    gcomp_acc = np.empty(shape=(len(ts), 4))
    gcomp_acc[:,0] = imu['ts']
    gcomp_acc[:,1:4] = temp

    if(suffix != ''):
        #plot for sanity check
        fig = plt.figure()
        x = gcomp_acc[:,0]
        y = gcomp_acc[:,1:4]
        plt.plot(x, y, alpha=0.5)
        plt.plot(x, np.linalg.norm(y, axis=1), alpha=0.5)
        plt.xlabel('time (s)')
        plt.ylabel(r'acc ($m/s^2$)')      
        plt.legend(['x', 'y', 'z', '||acc||'])   
        fig.savefig(plot_path+'/gcomp_acc'+suffix+'.png')
    
    return gcomp_acc

def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)

        
def integrateVelocities(data):
    import numpy as np
    from scipy.integrate import cumtrapz
    
    r = np.empty(data.shape)
    
    #copy timestamp and v0 = 0,0,0
    r[:,0] = data[:,0]
    r[0,1:4] = np.zeros(3)
    
    # integrate the data
    r[1:, 1:4] = cumtrapz(data[:,1:4], data[:,0], axis=0)
      
    return r
    
def setBoxColors(bp, c):
    from matplotlib import pyplot as plt
    plt.setp(bp['boxes'][:], color=c)
    plt.setp(bp['caps'][:], color=c)
    plt.setp(bp['whiskers'][:], color=c)
    plt.setp(bp['medians'][:], color=c)
    plt.setp(bp['fliers'][:], color=c)


def cropTime(data, start, end):
    idx = np.logical_and(data[:,0] >= start, data[:,0] <= end)
    return data[idx,:]


def MakeOrientationContinuous(data):
    import numpy as np
    #allocate and copy initial values
    ret = data.copy()
    
    it = 1
    
    swaps = np.logical_and(ret[1:,1:4] * ret[:-1,1:4] < 0, np.abs(ret[1:,1:4]) > np.pi/2)
    swaps = np.concatenate( ([[False, False, False]], swaps) )
    #print(np.any(swaps), len(np.unique(swaps)))
          
    while(np.any(swaps)):
        for k in range(3):
            swapIdx = np.where(swaps[:,k])[0]
            
            if len(swapIdx) % 2 == 1:
                swapIdx = np.append(swapIdx, (len(data) - 1))
            
            #if k == 0: print('swapIdx: ', swapIdx)
            
            for idx in range(0, len(swapIdx), 2):
                i = swapIdx[idx]
                j = swapIdx[idx+1]
                
                v0 = ret[i-1, k+1]
            
                #if k == 0: print('\n-----------\nbefore:\n', ret[i-1], '\n', ret[i], v0)
                if v0 > 0:
                    ret[i:j,k+1] += 2*np.pi*it
                    #if k == 0: print('+ 2pi')
                else:
                    ret[i:j,k+1] -= 2*np.pi*it
                    #if k == 0: print('- 2pi')
                #if k == 0: print('after :\n', ret[i-1], '\n', ret[i])
        
        ret = ret[:-1,:]
        it += 1
        swaps = np.logical_and(ret[1:,1:4] * ret[:-1,1:4] < 0, np.abs(ret[1:,1:4]) > np.pi/2)
        swaps = np.concatenate( ([[False, False, False]], swaps) )
        #print('##################################################', np.any(swaps), np.unique(swaps))
        #input('wait')
        #break
    return ret














    
        