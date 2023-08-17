#!/usr/bin/env python3

# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
# ---

"""
Reads ModEM's Jacobian, does fancy things.

@author: vrath   Feb 2021

"""

# Import required modules

import os
import sys

# import struct
import time
from datetime import datetime
import warnings
from sys import exit as error
import gc

import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl
import scipy.sparse as scs
import netCDF4 as nc


mypath = ["/home/vrath/Py4MT/py4mt/modules/", "/home/vrath/Py4MT/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)


import jacproc as jac
import modem as mod
import util as utl
from version import versionstrg

gc.enable()


rng = np.random.default_rng()
nan = np.nan  # float("NaN")
version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")

PY4MT_DATA = os.environ["PY4MT_DATA"]

rhoair = 1.e17

normalize_err = True
sparsify = True
sparse_thresh = 1.e-12
blank = np.nan
# Ubaye case"
# WorkDir = r"/home/vrath/work/MT_Data/Ubaye/UB22_jac_best/"
# MFile   = WorkDir +r"Ub22_ZoffPT_02_NLCG_014.rho"
# MPad=[12, 12 , 12, 12, 0, 36]
# # JFiles = [WorkDir+r"Ub22_Zoff.jac", ]
# # DFiles = [WorkDir+r"Ub22_Zoff.dat", ]

# # JFiles = [WorkDir+r"Ub22_P.jac", ]
# # DFiles = [WorkDir+r"Ub22_P.dat", ]

# # JFiles = [WorkDir+r"Ub22_T.jac", ]
# # DFiles = [WorkDir+r"Ub22_T.dat", ]

# JFiles = [WorkDir+r"Ub22_T.jac", WorkDir+r"Ub22_P.jac", WorkDir+r"Ub22_Zoff.jac", ]
# DFiles = [WorkDir+r"Ub22_T.dat", WorkDir+r"Ub22_P.dat", WorkDir+r"Ub22_Zoff.dat", ]


# KRAFLA 
# WorkDir = r"/media/vrath/BlackOne/MT_Data/Krafla/Krafla1/"
# MFile   = WorkDir +r"Krafla.rho"
# MPad=[13, 13 , 13, 13, 0, 36]
# JFiles = []
# DFiles = []
# files = os.listdir(WorkDir)
# for entry in files:
#     # print(entry)
#     if entry.endswith(".jac"):
#         JFiles.append(entry)
#         name, ext = os.path.splitext(entry)
#         DFiles.append(name+".dat")

# UBINAS
WorkDir = PY4MT_DATA+"/Peru/Ubinas/UbiJac/"

MFile   = WorkDir + "UBI_best.rho"
MPad=[14, 14 , 14, 14, 0, 71]

JFiles = [WorkDir + "UBI_best.jac", ]
DFiles = [WorkDir + "UBI_best_jac.dat", ]


NN = np.size(JFiles)
if np.size(DFiles) != np.size(JFiles):
    error("Data file number not equal Jac file number! Exit.")
nF = np.size(DFiles)


total = 0.0


start = time.time()
dx, dy, dz, rho, reference, _, vcell = mod.read_model(MFile, trans="linear", volumes=True)
dims = np.shape(rho)

aircells = np.where(rho>rhoair/100)

jacmask = jac.set_mask(rho=rho, pad=MPad, flat = True, out=True)
jdims= np.shape(jacmask)
j0 = jacmask.reshape(dims)
j0[aircells] = blank
jacmask = j0.reshape(jdims)


elapsed = time.time() - start
total = total + elapsed
print(" Used %7.4f s for reading model from %s " % (elapsed, MFile))

mxVal = 1e-30
mxLst = []
# nF = 1
for f in np.arange(nF):

    name, ext = os.path.splitext(JFiles[f])
    start =time.time()
    print("\nReading Data from "+DFiles[f])
    Data, Site, Freq, Comp, Head = mod.read_data_jac(DFiles[f])
    elapsed = time.time() - start
    print(" Used %7.4f s for reading Data from %s " % (elapsed, DFiles[f]))
    total = total + elapsed

    start = time.time()
    print("Reading Jacobian from "+JFiles[f])
    Jac, Info = mod.read_jac(JFiles[f])
    elapsed = time.time() - start
    print(" Used %7.4f s for reading Jacobian from %s " % (elapsed, JFiles[f]))
    total = total + elapsed

    nstr = ""
    if normalize_err:
        nstr = nstr+"_nerr"
        start = time.time()
        dsh = np.shape(Data)
        err = np.reshape(Data[:, 7], (dsh[0], 1))
        mx0 = np.nanmax(np.abs(Jac*jacmask))
        Jac = jac.normalize_jac(Jac, err)
        elapsed = time.time() - start
        print(" Used %7.4f s for normalizing Jacobian from %s " % (elapsed, JFiles[f]))

    mx = np.nanmax(np.abs(Jac*jacmask))
    print(JFiles[f]+" maximum value is "+str(mx))
    if f==0:
        mxwhere = np.where(np.abs(Jac*jacmask) > mx/1000)
        test0 = np.zeros((1,np.size(rho)))
        test1 = Jac[71,:]*jacmask
        index = np.where(np.abs(test1) > np.nanmax(test1)/1000.)
        test0[0,index[0]]=1.
        test0 =test0.reshape(dims)
        fn = WorkDir+"test_largevals.rho"
        mod.write_model(fn, dx, dy, dz, test0, reference, trans="linear", air=aircells)
        fn = WorkDir+"test_sensline.rho"
        test1 = test1.reshape(dims)
        mod.write_model(fn, dx, dy, dz, test1, reference, trans="linear", air=aircells)
        test2 = jacmask.reshape(dims)
        mod.write_model(fn, dx, dy, dz, test2, reference, trans="linear", air=aircells)



    mxLst.append(mx)
    mxVal = np.amax([mxVal,mx])

print(" Merged Maximum value is "+str(mxVal))



for f in np.arange(nF):

    name, ext = os.path.splitext(JFiles[f])
    start =time.time()
    print("\nReading Data from "+DFiles[f])
    Data, Site, Freq, Comp, Head = mod.read_data_jac(WorkDir+DFiles[f])
    elapsed = time.time() - start
    print(" Used %7.4f s for reading Data from %s " % (elapsed, DFiles[f]))
    total = total + elapsed

    start = time.time()
    print("Reading Jacobian from "+JFiles[f])
    Jac, Info = mod.read_jac(WorkDir+JFiles[f])
    elapsed = time.time() - start
    print(" Used %7.4f s for reading Jacobian from %s " % (elapsed, JFiles[f]))
    total = total + elapsed

    nstr = ""
    if normalize_err:
        nstr = nstr+"_nerr"
        start = time.time()
        dsh = np.shape(Data)
        err = np.reshape(Data[:, 7], (dsh[0], 1))
        mx0 = np.nanmax(np.abs(Jac))
        Jac = jac.normalize_jac(Jac, err)
        elapsed = time.time() - start
        print(" Used %7.4f s for normalizing Jacobian from %s " % (elapsed, JFiles[f]))

    mx = np.nanmax(np.abs(Jac*jacmask))
    mxLst.append(mx)
    mxVal = np.amax([mxVal,mx])


    sstr=""
    if sparsify:
        sstr="_sp"+str(round(np.log10(sparse_thresh)))
        start = time.time()
        Jacs, _ = jac.sparsify_jac(Jac,sparse_thresh=mxVal*sparse_thresh)
        elapsed = time.time() - start
        total = total + elapsed
        print(" Used %7.4f s for sparsifying Jacobian %s " % (elapsed, JFiles[f]))
        NPZFile = name+nstr+sstr+"_JacStack_jacs.npz"
        scs.save_npz(NPZFile, Jacs)
        NPZFile = name+nstr+sstr+"_JacStack_info.npz"
        np.savez(NPZFile, Data=Data, Site=Site, Freq=Freq, Comp=Comp)

        if f==0:
            JacStack = Jacs.copy()
            DataStack = Data.copy()
            SiteStack = Site.copy()
            FreqStack = Freq.copy()
            CompStack = Comp.copy()
        else:
            JacStack = scs.vstack([JacStack, Jacs])
            DataStack = np.append(DataStack, Data, axis=0)
            SiteStack = np.append(SiteStack, Site, axis=0)
            FreqStack = np.append(FreqStack, Freq, axis=0)
            CompStack = np.append(CompStack, Comp, axis=0)


NPZFile = WorkDir+"Krafla1"+nstr+sstr+"_JacStack_jacs.npz"
scs.save_npz(NPZFile, JacStack)
NPZFile = WorkDir+"Krafla1"+nstr+sstr+"_JacStack_info.npz"
np.savez(NPZFile, Data=DataStack, Site=SiteStack, Freq=FreqStack,Comp=CompStack)
elapsed = time.time() - start
total = total + elapsed
print(" Used %7.4f s for writing sparsified Jacobian to %s " % (elapsed, name+nstr+sstr+"_JacStack"))
print("\n\nUsed %7.4f s for processing Jacobian." % (total))
