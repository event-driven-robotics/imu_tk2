# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 15:09:41 2020

@author: Leandro de Souza Rosa <leandro.desouzarosa@iit.it>

Inital code implementation based on https://github.com/morgil/madgwick_py
"""

# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import norm
from quaternion import Quaternion
from tqdm import tqdm
from matplotlib import pyplot as plt

class MadgwickAHRS:
    
    def __init__(self, **kwargs):
        """
        Initialize the class with the given parameters.
        :param quaternion: Initial quaternion
        :param beta: Algorithm gain beta
        :param gravityMag: magntude of the gravity in the enviroment
        :param gyrError: gyroscope error in the three directions [error_axis0,
                                                      error_axis1, error_axis2]
        :param timestamp: initializes the filter timestamp,
                                                      otherwise it is set to 0
        
        :param accelerometer: initial accelerometer measurement for
                               initialization [acc_axis0, acc_axis1, acc_axis2]
        
        # data-specific defaults have been removed from class, for:
        # default values IMU calibration values for SIMULATOR
        # https://github.com/uzh-rpg/rpg_esim/blob/master/event_camera_simulator/esim_trajectory/src/imu_factory.cpp
        # default values IMU calibration values for DAVIS from datasheet
        # https://datasheet.octopart.com/MPU-6150-InvenSense-datasheet-15992095.pdf

        """
        self.quaternion = Quaternion(1, 0, 0, 0)

        self.verbose = kwargs.pop('verbose', False)
        self.vis = kwargs.pop('vis', False)
        self.g = abs(kwargs.pop('gravityMag'))
        
        #just to declare the variable
        self.__gyro_bias = np.zeros(4);
        
        if 'imuErrorGyr' in kwargs:
            self.gyrError = kwargs.pop('imuErrorGyr')
            if self.verbose: print('using gyrError: ', self.gyrError)
        elif 'beta' in kwargs:
            beta = kwargs.pop('beta')
            if self.verbose: print('using beta: ', beta)
            self.gyrError = np.ones(3)*beta/np.sqrt(3/4)
        else:
            raise Exception('\nno gyrError or Beta input')
            
        if 'imu' in kwargs:
            self.ts = kwargs.get('imu')['ts'][0]
            if self.verbose: print('Initializing filter timestamp: ', self.ts)
        else:
            self.ts = 0
            
        self.noGravityAccelerations = np.zeros([1, 3], dtype = np.float64).flatten()
        self.__previousAccelerations = np.zeros([1, 3], dtype = np.float64).flatten()
        
        init = kwargs.pop('init', False)
        
        if (init):
            if 'orientation' in kwargs:
                if self.verbose: print('Initialization with Orientation')
                self.quaternion = Quaternion(kwargs.pop('orientation'))
            
            elif 'imu' in kwargs:
                imu = kwargs.pop('imu')
                timestamps = imu['ts']
                #print('here imu ts: ', imu['ts'][0])
                gyroscope = imu['angV']
                accelerometer = imu['acc']
                log = []
                
                with tqdm(total=len(timestamps), position=0, leave=True) as pbar:
                    for ts, gyr, acc in zip(timestamps, gyroscope, accelerometer):
                        self.update(
                                gyroscope = gyr,
                                accelerometer = acc,
                                timestamp = ts
                                )
                        log.append(self.getOrientation())
                        if self.verbose: pbar.update(1)
                    
                if(self.vis):
                    plt.close('all')
                    plt.plot(log)
                    
                    plt.legend(['qw', 'qx', 'qy', 'qz'])
                    plt.xlabel('iteration')
                    plt.ylabel('quaternion')
                    
                    
                print(' ori: ', self.quaternion.q, ' - norm: ', norm(self.quaternion.q))
                #self.__gravityCompensate(acc);
            
        print(kwargs)
        assert not kwargs
    
    def reset(self):
        self.quaternion = Quaternion(1, 0, 0, 0)
        
    def getAccelerations(self):
        """
        Get gravity compensated acc
        :return: np.array([comp_accX, comp_accY, comp_accZ])
        """
        return self.noGravityAccelerations.copy()
    
    def __gravityCompensate(self, acc_in):
        
        gq = Quaternion([0, 0, 0, 1])
        rg = self.quaternion.conj()*gq*self.quaternion
        self.noGravityAccelerations = acc_in - self.g * rg.q[1:4]
        '''
        print('acc in: ', acc_in, " - norm: ", norm(acc_in))
        print('rot g3: ', self.g*rg.q[1:4], ', norm: ', self.g*norm(rg.q))
        print('acc: ', self.noGravityAccelerations , ', norm: ', 
                                            norm(self.noGravityAccelerations ))
        print('-----')
        '''
        
    def getOrientation(self):
        return self.quaternion.q
    
    def update(self, *args, **kwargs):
        """
        Perform one update gradient with data from a AHRS sensor array. 
        Accelerometer and Magnetometer units do not matter
        :param gyroscope: gyroscope x, y, z in rads per second.
        :param accelerometer: accelerometer x, y, z.
        :param magnetometer: magnetometer x, y, z.
        :return:
        """
        q = self.quaternion
        
        if 'gyroscope' in kwargs:
            gyr = kwargs.get('gyroscope').copy()
        else:
            raise Exception('no gyroscope input, exiting')
            
        if 'accelerometer' in kwargs:
            acc = kwargs.get('accelerometer').copy()
        else:
            raise Exception('no accelerometer input')
            
        if 'timestamp' in kwargs:
            ts = kwargs.get('timestamp')
        else:
            print('no timestamp input')
            return;
            
        #start the filter steps
        acc_in = acc.copy() #save before normalization for gravity compensate

        #print('acc: ', acc)
        if norm(acc) == 0:
            print("accelerometer is zero")
            return;
        acc /= norm(acc)
        
        # Correction
        f = [
                2*(q[1]*q[3] - q[0]*q[2]) - acc[0],
                2*(q[0]*q[1] + q[2]*q[3]) - acc[1],
                2*(0.5 - q[1]**2 - q[2]**2) - acc[2],
            ]
        
        j = [
                [-2*q[2], 2*q[3], -2*q[0], 2*q[1]],
                [2*q[1], 2*q[0], 2*q[3], 2*q[2]],
                [0, -4*q[1], -4*q[2], 0]
            ]
    
        
        f = np.array(f)
        j = np.array(j)
        
        gradient= j.T.dot(f)
        gradient /= norm(gradient)  # normalise gradientmagnitude
        
        dtime = ts - self.ts
        
        error_gyro = np.concatenate( ([0], self.gyrError) )
        
        self.__gyro_bias = np.zeros(4)
        #print('\n\ngyro: ', type(gyr))
        
        if (gyr == [0, 0, 0]).all():
            __temp = 0.5*(q * Quaternion(1, 0, 0, 0))
        else:
            compensated_gyro = (Quaternion(0, gyr[0], gyr[1], gyr[2]) - self.__gyro_bias)
            __temp = 0.5*(q * Quaternion(compensated_gyro))
        
        #beta = 1
        beta = norm((q * Quaternion(error_gyro) ) * 0.5)
        qdot = __temp - beta*gradient.T
        
        # Integrate to yield quaternion
        q += qdot * dtime # calculate the delta time
        self.ts = ts
        self.quaternion = Quaternion(q / norm(q))  # normalise quaternion
        
        # Calculate the acceleration by removing the gravity
        self.__gravityCompensate(acc_in)
        
    def gravityCompensate(self, imu):
        timestamps = imu['ts']
        #print('here imu ts: ', imu['ts'][0])
        gyroscope = imu['angV']
        accelerometer = imu['acc']
        compAcc = []
        with tqdm(total=len(timestamps), position=0, leave=True) as pbar:
            for ts, gyr, acc in zip(timestamps, gyroscope, accelerometer):
                self.update(
                        gyroscope = gyr,
                        accelerometer = acc,
                        timestamp = ts
                        )
                compAcc.append(self.getAccelerations())
                if self.verbose: pbar.update(1)
                
        return np.array(compAcc)