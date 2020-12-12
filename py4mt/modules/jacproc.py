#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 17:36:08 2020

"""
import numpy as np
import scipy.sparse as scp



# ------------------------------------------------------------------------------

def rsvd(A, rank, n_oversamples=None, n_subspace_iters=None,
         return_range=False):
    """
    =============================================================================
    Randomized SVD. See Halko, Martinsson, Tropp's 2011 SIAM paper:

    "Finding structure with randomness: Probabilistic algorithms for constructing
    approximate matrix decompositions"
    Author: Gregory Gundersen, Princeton, Jan 2019
    =============================================================================
    Randomized SVD (p. 227 of Halko et al).

    :param A:                (m x n) matrix.
    :param rank:             Desired rank approximation.
    :param n_oversamples:    Oversampling parameter for Gaussian random samples.
    :param n_subspace_iters: Number of power iterations.
    :param return_range:     If `True`, return basis for approximate range of A.
    :return:                 U, S, and Vt as in truncated SVD.
    """
    if n_oversamples is None:
        # This is the default used in the paper.
        n_samples = 2 * rank
    else:
        n_samples = rank + n_oversamples

    # Stage A. 
    # print(' stage A')
    Q = find_range(A, n_samples, n_subspace_iters)

    # Stage B.
    # print(' stage B')
    B = Q.T @ A
    # print(np.shape(B))
    # print(' stage B before linalg')
    U_tilde, S, Vt = np.linalg.svd(B)
    # print(' stage B after linalg')
    U = Q @ U_tilde

    # Truncate.
    U, S, Vt = U[:, :rank], S[:rank], Vt[:rank, :]

    # This is useful for computing the actual error of our approximation.
    if return_range:
        return U, S, Vt, Q
    return U, S, Vt

# ------------------------------------------------------------------------------

def find_range(A, n_samples, n_subspace_iters=None):
    """Algorithm 4.1: Randomized range finder (p. 240 of Halko et al).

    Given a matrix A and a number of samples, computes an orthonormal matrix
    that approximates the range of A.

    :param A:                (m x n) matrix.
    :param n_samples:        Number of Gaussian random samples.
    :param n_subspace_iters: Number of subspace iterations.
    :return:                 Orthonormal basis for approximate range of A.
    """
    # print('here we are in range-finder')
    m, n = A.shape
    O = np.random.randn(n, n_samples)
    Y = A @ O

    if n_subspace_iters:
        return subspace_iter(A, Y, n_subspace_iters)
    else:
        return ortho_basis(Y)

# ------------------------------------------------------------------------------

def subspace_iter(A, Y0, n_iters):
    """Algorithm 4.4: Randomized subspace iteration (p. 244 of Halko et al).

    Uses a numerically stable subspace iteration algorithm to down-weight
    smaller singular values.

    :param A:       (m x n) matrix.
    :param Y0:      Initial approximate range of A.
    :param n_iters: Number of subspace iterations.
    :return:        Orthonormalized approximate range of A after power
                    iterations.
    """
    # print('herere we are in subspace-iter')
    Q = ortho_basis(Y0)
    for _ in range(n_iters):
        Z = ortho_basis(A.T @ Q)
        Q = ortho_basis(A @ Z)
    return Q

# ------------------------------------------------------------------------------

def ortho_basis(M):
    """Computes an orthonormal basis for a matrix.

    :param M: (m x n) matrix.
    :return:  An orthonormal basis for M.
    """
    # print('herere we are in ortho')
    Q, _ = np.linalg.qr(M)
    return Q


def sparsifyJac(Jac=None,sparse_thresh =1.E-6,normalized = True, method=None, out = True):
    """
    Sparsifies error_scaled Jacobian from ModEM output
    
    author: vrath
    last changed: Sep 25, 2020
    """
    shj = np.shape(Jac)
    if out:
        nel = shj[0]*shj[1]
        print('sparsifyJac: dimension of original J is %i x %i = %i elements' 
              % (shj[0],shj[1],nel))
        
    Jac    = np.abs(Jac)
    Jmax   = np.amax(Jac)
    thresh = Jmax*sparse_thresh
    Jac[Jac<thresh] = 0.0
    Js= scp.csr_matrix(Jac)
    
    if scp.issparse(Js):
        ns = scp.csr_matrix.count_nonzero(Js)
        print('sparsifyJac: output J is sparse: %r, and has  %i nonzeros, %f percent' %
              (scp.issparse(Js),ns,100.*ns/nel))
        
    if normalized:
        f =1./Jmax
        Js = f*Js
        
    return Js 

def normalizeJac(Jac=None,fn = None, out = True):
    """
    normalizes Jacobian from ModEM output
    
    author: vrath
    last changed: July 25, 2020
    """
    shj = np.shape(Jac)
    shf = np.shape(fn)
    if shf[0] == 1:
        f=1./fn
        Jac = f*Jac
    else:
        erri = np.reshape(1./fn,(shj[0],1))
        Jac = erri[:]*Jac
        
    return Jac

def calculateSens(Jac=None,normalize=True, small = 1.e-14, out = True):
    """
    normalizes Jacobian from ModEM output
    
    author: vrath
    last changed: Sep 25, 2020
    """

    if scp.issparse(Jac):
        J = Jac.todense()
    else:
        J = Jac

    S  = np.sum(np.power(J,2),axis=0)

    if normalize:
        
        Smax=np.amax(S)
        S = S/Smax
        
    if small <= 1.e-14:
        S[S<small] = np.NaN

    return S, Smax

def projectMod(m=None,U=None, small = 1.e-14, out = True):
    
    """
    Nullspace Projection
    (see Munoz & Rath, 2006)
    author: vrath
    last changed: Sep 25, 2020
    """
    b = np.dot(U.T,m)
    # print(m.shape)
    # print(b.shape)
    # print(U.shape)
    
    mp = m - np.dot(U,b)
    
    return mp


def transMod(m=None,M=None, small = 1.e-14, out = True):
    
    """
    Transform Model 
    M shoulld be something like C_m^-1/2
    ( see egg Kelbert 2012, Egbert & kelbert 2014)
    author: vrath
    last changed:  Oct 12, 2020
    """
    
   
    transm = np.dot(M,m)
    
    return transm