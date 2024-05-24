'''
 * ----------------------------------------------------------------------------
 * Copyright 2021, Tzu-Yuan Lin <tzuyaun@umich.edu>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

 * This is a python version of Ross Hartley's invariant EKF code.
 * Original C++ code can be found here: https://github.com/RossHartley/invariant-ekf

 **
 *  @file   LieGroup.py
 *  @author Tzu-Yuan Lin, Chankyo Kim
 *  @brief  Source file for various Lie Group functions 
 *  @date   April 1, 2021
 **
'''

import numpy as np
import scipy.linalg
import sys

TOLERANCE = 1e-10

def skew(v):
    # M = [[0 , -v[2], v[1]],
    #     [v[2], 0, -v[0]],
    #     [-v[1], v[0], 0]]
    
    v2 = v.copy()
    M = np.zeros((3,3))
    M[0,1] = -v2[2]
    M[0,2] = v2[1]
    M[1,0] = v2[2]
    M[1,2] = -v2[0]
    M[2,0] = -v2[1]
    M[2,1] = v2[0]

    return M

def unskew(M):
    v = [M[2,1],M[0,2],M[1,0]]

    return np.array(v)

def factorial(n):
    return 1 if (n==1 or n==0) else factorial(n-1)*n

def Gamma_SO3(w, m):
    # Computes mth integral of the exponential map: 
    # \Gamma_m = \sum_{n=0}^{\infty} \dfrac{1}{(n+m)!} (w^\wedge)^n
    assert m >= 0
    I = np.eye((3))
    theta = np.linalg.norm(w)
    if theta < TOLERANCE:
        return (1.0/factorial(m))*I
    
    A = skew(w)
    theta2 = theta*theta

    if m == 0: # Exp map of SO(3)
        # print(scipy.linalg.expm(A))
        # return I + (np.sin(theta)/theta)*A + ((1-np.cos(theta))/theta2)*A@A
        return scipy.linalg.expm(A)
    elif m == 1: # Left Jacobian of SO(3)
        # eye(3) - A*(1/theta^2) * (R - eye(3) - A)
        # eye(3) + (1-cos(theta))/theta^2 * A + (theta-sin(theta))/theta^3 * A^2
        return I + ((1-np.cos(theta))/theta2)*A + ((theta-np.sin(theta))/(theta2*theta))*A@A
    elif m == 2:
        # 0.5*eye(3) - (1/theta^2) * (R - eye(3) - A - 0.5*A^2);
        # 0.5*eye(3) + (theta-sin(theta))/theta^3 * A + (2*(cos(theta)-1) + theta^2)/(2*theta^4) * A^2  
        return 0.5*I + (theta-np.sin(theta))/(theta2*theta)*A + (theta2 + 2*np.cos(theta)-2)/(2*theta2*theta2)*A@A
    else: 
        R = np.eye(3) + (np.sin(theta)/theta)*A + ((1-np.cos(theta))/theta2)*A@A
        S = np.eye(3)
        Ak = np.eye(3)
        kfactorial = 1
        for k in np.arange(1,m+1):
            kfactorial = kfactorial*(k)
            Ak = Ak@A
            S = (S + (1.0/kfactorial)*Ak)
        
        if m==0:
            return R
        elif m%2:
            return (1.0/kfactorial)*np.eye(3) + (np.power(-1,(m+1)/2)/np.power(theta,m+1))*A @ (R-S)
        else:
            return (1.0/kfactorial)*np.eye(3) + (np.power(-1,m/2)/np.power(theta,m)) * (R - S)

def Exp_SO3(w):
    return Gamma_SO3(w,0)

def LeftJacobian_SO3(w):
    return Gamma_SO3(w,1)

def RightJacobian_SO3(w):
    return Gamma_SO3(-w,1)

def Exp_SEK3(v):
    # assert v.ndim == 1
    K = int((v.size-3)/3)
    X = np.eye(3+K)
    R = None
    Jl = None
    w = v[0:3]
    theta = np.linalg.norm(w)
    I = np.eye(3)
    if theta < TOLERANCE:
        R = I
        Jl = I
    else:
        A = skew(w)
        theta2 = theta*theta
        stheta = np.sin(theta)
        ctheta = np.cos(theta)
        oneMinusCosTheta2 = (1-ctheta)/(theta2)
        A2 = A@A
        R = I + (stheta/theta)*A + oneMinusCosTheta2*A2
        Jl = I + oneMinusCosTheta2*A + ((theta-stheta)/(theta2*theta))*A2

    X[0:3,0:3] = R
    for i in range(K):
        X[0:3,3+i] = (Jl@v[3+3*i:6+3*i]).reshape(3)
    
    return X

def Adjoint_SEK3(X):
    K = np.shape(X)[1]-3
    Adj = np.zeros((3+3*K,3+3*K))
    R = X[0:3,0:3].copy()
    Adj[0:3,0:3] = R
    for i in range(K):
        Adj[3+3*i:6+3*i,3+3*i:6+3*i] = R
        Adj[3+3*i:6+3*i,0:3] = skew(X[0:3,3+i])@R

    return Adj

# def main():
#     v = np.array([0,1,2,4,5,6])
#     print(v)
#     M = skew(v)
#     print(M)
#     print(type(M))    
#     v_2 = unskew(M)
#     print(v_2)
#     print("---------Testing Exp_SEK3---------")
#     X = Exp_SEK3(v)
#     print(X)
#     Adj = Adjoint_SEK3(X)
#     print(Adj)

# if __name__ == '__main__':
#     main()