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
Reads ModEM'smodel files, does cellwise math on thenm.
Ellipsoids or boxes

@author: vr Feb 2021

"""
import os
import sys
from sys import exit as error
import time
from datetime import datetime
import warnings

import numpy as np
import netCDF4 as nc
import scipy.ndimage as spn
import scipy.linalg as spl

import vtk
import pyvista as pv
import PVGeo as pvg


PY4MT_ROOT = os.environ["PY4MT_ROOT"]
mypath = [PY4MT_ROOT+"/py4mt/modules/", PY4MT_ROOT+"/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)


import modem as mod
import util as utl
from version import versionstrg

warnings.simplefilter(action="ignore", category=FutureWarning)

Strng, _ = versionstrg()
now = datetime.now()
print("\n\n"+Strng)
print("Generate Random sets of bodies"+"\n"+"".join("Date " + now.strftime("%m/%d/%Y, %H:%M:%S")))
print("\n\n")



rhoair = 1.0e17


ModFile_in = r"/home/vrath/work/MT/Annecy/ImageProc/In/ANN20_02_PT_NLCG_016"
ModFile_out = r"/home/vrath/work/MT/Annecy/ImageProc/Out/ANN20_02_PT_NLCG_016_insert"

geocenter = [45.938251, 6.084900]
utm_x, utm_y = utl.proj_latlon_to_utm(geocenter[0], geocenter[1], utm_zone=32631)
utmcenter = [utm_x, utm_y, 0.0]

ell = ["ell", "rep", 0.0, 0.0, 0.0, 3000.0, 10000, 2000.0, 1000.0, 0.0, 0.0, 30.0]
box = ["box", "rep", 0.0, 0.0, 0.0, 1000.0, 2000.0, 1000.0, 1000.0, 0.0, 0.0, 30.0]
bodies = [ell, box]
nb = np.shape(bodies)
# smoother=['gaussian',0.5]
smoother = ["uniform", 3]
total = 0
start = time.time()

dx, dy, dz, rho, reference = mod.read_model(ModFile_in + ".rho", out=True)
# writeMod(ModFile_out+'.rho', dx, dy, dz, rho,reference,out = True)
elapsed = time.time() - start
total = total + elapsed
print(" Used %7.4f s for reading model from %s " % (elapsed, ModFile_in + ".rho"))

air = rho > rhoair / 100.0

rho = mod.prepare_mod(rho, rhoair=rhoair)

for ibody in range(nb[0]):
    body = bodies[ibody]
    rhonew = mod.insert_body(dx, dy, dz, rho, body, smooth=smoother)
    rhonew[air] = rhoair
    Modout = ModFile_out + "_" + body[0] + str(ibody) + "_" + smoother[0] + ".rho"
    mod.write_model(Modout, dx, dy, dz, rhonew, reference, out=True)

    elapsed = time.time() - start
    print(" Used %7.4f s for processing/writing model to %s " % (elapsed, Modout))
    print("\n")


total = total + elapsed
