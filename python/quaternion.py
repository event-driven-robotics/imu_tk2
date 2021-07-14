# -*- coding: utf-8 -*-

import numpy as np
import numbers


from scipy.spatial.transform import Rotation as R

class rotation(R):    
        
    @staticmethod
    def from_wxyz_quat(q, **kwargs):
        if isinstance(q, list):
            q = np.array(q)
        
        q2r_order = [1, 2, 3, 0]
        
        if np.shape(q) != (4,):
            nrow, ncol = np.shape(q)
            assert(ncol == 4)
            if nrow > 1:
                q = q[:, q2r_order]
        else:
            q = q[q2r_order]
        
        
        return rotation.from_quat(q, kwargs)
    
    def as_wxyz_quat(self):
        q = self.as_quat()
        
        r2q_order = [3, 0, 1, 2]
        
        if np.shape(q) != (4,):
            nrow, ncol = np.shape(q)
            if nrow > 1:
                q = q[:, r2q_order]
        else:
            q = q[r2q_order]
            
        return q



class Quaternion:
    """
    A simple class implementing basic quaternion arithmetic.
    """
    def __init__(self, w_or_q, x=None, y=None, z=None):
        """
        Initializes a Quaternion object
        :param w_or_q: A scalar representing the real part of the quaternion,
                    another Quaternion object or a
                    four-element array containing the quaternion values
        :param x: The first imaginary part if w_or_q is a scalar
        :param y: The second imaginary part if w_or_q is a scalar
        :param z: The third imaginary part if w_or_q is a scalar
        """
        self._q = np.array([1, 0, 0, 0])

        if x is not None and y is not None and z is not None:
            w = w_or_q
            q = np.array([w, x, y, z])
        elif isinstance(w_or_q, Quaternion):
            q = np.array(w_or_q.q)
        else:
            q = np.array(w_or_q)
            if len(q) != 4:
                raise ValueError(
                    "Expecting a 4-element array or w x y z as parameters")

        self._set_q(q)

    # Quaternion specific interfaces
    def conj(self):
        """
        Returns the conjugate of the quaternion
        :rtype : Quaternion
        :return: the conjugate of the quaternion
        """
        return Quaternion(self._q[0], -self._q[1], -self._q[2], -self._q[3])

    @staticmethod
    def fromEuler(ex, ey, ez):
        """
        Returns a quartenion representing the rotation given in Euler angles
        :param ex: Rotation around X axis
        :param ey: Rotation around y axis
        :param ez: Rotation around z axis
        """
        
        if ex == None or ey == None or ez == None:
            raise Exception('must input ex, ey, and ez')
        
        '''
        http://euclideanspace.com/maths/geometry/rotations/conversions/eulerToQuaternion/index.htm
        if h = heading angle (rotation about y) 	then Qh = quaternion for
        pure heading rotation = cos(h/2) + j sin(h/2) = c1 + j s1
        if a = attitude angle (rotation about z) then Qa = quaternion for pure
        attitude rotation = cos(a/2) + k sin(a/2) = c2 + k s2
        if b = bank angle (rotation about x) then Qb = quaternion for pure
        bank rotation = cos(b/2) + i sin(b/2) = c3 + i s3
        '''
        
        c1 = np.cos(ey/2);
        s1 = np.sin(ey/2);
        c2 = np.cos(ez/2);
        s2 = np.sin(ez/2);
        c3 = np.cos(ex/2);
        s3 = np.sin(ex/2);
        c1c2 = c1*c2;
        s1s2 = s1*s2;
        q = np.zeros(4)
        q[0] = c1c2*c3 - s1s2*s3;
        q[1] = c1c2*s3 + s1s2*c3;
        q[2] = s1*c2*c3 + c1*s2*s3;
        q[3] = c1*s2*c3 - s1*c2*s3;
       
        return Quaternion(q/np.linalg.norm(q))
    
    @staticmethod
    def toQuaternion(rad, x, y, z):
        s = np.sin(rad / 2)
        return Quaternion(np.cos(rad / 2), x*s, y*s, z*s)

    def toEuler(self):
        pitch = np.arcsin(2 * self[1] * self[2] + 2 * self[0] * self[3])
        if np.abs(self[1] * self[2] + self[3] * self[0] - 0.5) < 1e-8:
            roll = 0
            yaw = 2 * np.arctan2(self[1], self[0])
        elif np.abs(self[1] * self[2] + self[3] * self[0] + 0.5) < 1e-8:
            roll = -2 * np.arctan2(self[1], self[0])
            yaw = 0
        else:
            roll = np.arctan2(2 * self[0] * self[1] - 2 * self[2] * self[3],
                              1 - 2 * self[1] ** 2 - 2 * self[3] ** 2)
            yaw = np.arctan2(2 * self[0] * self[2] - 2 * self[1] * self[3],
                             1 - 2 * self[2] ** 2 - 2 * self[3] ** 2)
        return roll, pitch, yaw

    def to_euler123(self):
        roll = np.arctan2(-2*(self[2]*self[3] - self[0]*self[1]),
                          self[0]**2 - self[1]**2 - self[2]**2 + self[3]**2)
        pitch = np.arcsin(2*(self[1]*self[3] + self[0]*self[1]))
        yaw = np.arctan2(-2*(self[1]*self[2] - self[0]*self[3]),
                         self[0]**2 + self[1]**2 - self[2]**2 - self[3]**2)
        return roll, pitch, yaw

    def __mul__(self, other):
        """
        multiply the given quaternion with another quaternion or a scalar
        :param other: a Quaternion object or a number
        :return:
        """
        if isinstance(other, Quaternion):
            w = (self._q[0] * other._q[0] - self._q[1] * other._q[1]
                         - self._q[2] * other._q[2] - self._q[3] * other._q[3])
            x = (self._q[0] * other._q[1] + self._q[1] * other._q[0]
                         + self._q[2] * other._q[3] - self._q[3] * other._q[2])
            y = (self._q[0] * other._q[2] - self._q[1] * other._q[3]
                         + self._q[2] * other._q[0] + self._q[3] * other._q[1])
            z = (self._q[0] * other._q[3] + self._q[1] * other._q[2]
                         - self._q[2] * other._q[1] + self._q[3] * other._q[0])

            return Quaternion(w, x, y, z)
        elif isinstance(other, numbers.Number):
            q = self._q * other
            return Quaternion(q)
        
    def __truediv__(self, other):
        if isinstance(other, numbers.Number):
            q = self._q / other
            return Quaternion(q)
        else:
            raise Exception('Can only divide quaternions by scalars')
            
    def __rmul__(self, other):
        """
        multiply the shit backwards with another scalar
        :param other: a number
        :return:
        """
        if isinstance(other, numbers.Number):
            q = self._q * other
            return Quaternion(q)

    def __add__(self, other):
        """
        add two quaternions element-wise
        or add a scalar to each element of the quaternion
        :param other:
        :return:
        """
        if not isinstance(other, Quaternion):
            if len(other) != 4:
                raise TypeError("""Quaternions must be added to other
                                quaternions or a 4-element array""")
            q = self.q + other
        else:
            q = self.q + other.q

        return Quaternion(q)

    # Implementing other interfaces to ease working with the class

    def _set_q(self, q):
        self._q = q

    def _get_q(self):
        return self._q

    q = property(_get_q, _set_q)

    def __getitem__(self, item):
        return self._q[item]

    def __array__(self):
        return self._q
