#!/usr/bin/env python3

"""
This script constructs a list of edifiles in a given directory, and produces
plots for all of them.

@author: sb & vr oct 2019
adapte

"""

# Import required modules

import os
import sys
import numpy as np
from mtpy.core.mt import MT

import matplotlib as mpl
import matplotlib.pyplot as plt

PY4MT_DATA = os.environ["PY4MT_DATA"]
PY4MT_ROOT = os.environ["PY4MT_ROOT"]

mypath = [PY4MT_ROOT+"/py4mt/modules/", PY4MT_ROOT+"/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)

import util
from version import versionstrg


version, _ = versionstrg()
titstrng = util.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")

# Graphical paramter. Determine the plot formats produced,
# and the required resolution:

PlotFmt = [".png"]
dpi = 400
PdfC = True
if not ".pdf" in PlotFmt:
    PdfC = False
    print("No PDF catalog because no pdf output!")
PdfCName  = "Opf2023_data.pdf"


# What should be plotted?
# 1 = yx and xy; 2 = all 4 components
# 3 = off diagonal + determinant

PlotType = 2
no_err = False
strng="_Z"+str(PlotType)
# Plot tipper?
# "y" or "n", followed by "r","i", or "ri", for real part, imaginary part, or both, respectively.
plot_t = "n" #yri" #""yri"
print (plot_t[0])
if plot_t[0]=="y":
    strng = strng+"T"+plot_t[1:]
# Plot phase tensor?
# "y" or "n"
plot_p = "y"
if plot_p=="y":
    strng = strng+"P"

strng="_data1"


PerLimits = (0.0001, 1.)  # AMT
# PerLimits = (0.001,100000.) #BBMT
# PerLimits = (0.00003,10000.) #AMT+BBMT
RhoLimits = (10., 10000.)
PhiLimits = (-180., 180.)
Tiplimits = (-.5, 0.5)


# Define the path to your EDI-files:
EdiDir = r"/home/vrath/rjmcmc_mt/work/edi/"
print(" Edifiles read from: %s" % EdiDir)
    
# Define the  path for saving  plots:
PltDir = EdiDir +"/plots/"
print(" Plots written to: %s" % PltDir)
if not os.path.isdir(PltDir):
    print(" File: %s does not exist, but will be created" % PltDir)
    os.mkdir(PltDir)

# No changes required after this line!

# Construct list of EDI-files:


edi_files = []
files = os.listdir(EdiDir)
for entry in files:
    # print(entry)
    if entry.endswith(".edi") and not entry.startswith("."):
        edi_files.append(entry)
edi_files = sorted(edi_files)

if PdfC:
    pdf_list= []
    for filename in edi_files:
        name, ext = os.path.splitext(filename)
        pdf_list.append(PltDir+name+strng+".pdf")
# Create an MT object

for filename in edi_files:
    name, ext = os.path.splitext(filename)
    file_i = EdiDir + filename
    
    mt_obj = MT()
    mt_obj.read(file_i)
    
    lat = mt_obj.station_metadata.location.latitude
    lon = mt_obj.station_metadata.location.longitude
    elev = mt_obj.station_metadata.location.elevation
    print(" site %s at :  % 10.6f % 10.6f % 8.1f" % (name, lat, lon, elev ))


# • mt.station_metadata.time_period.start
    pmin, pmax  = PerLimits

    # mt.station_metadata.time_period.end = 
       
    p = mt_obj.Z.period
    ps = np.shape(p)

    # z = mt.Z.z
    # t = mt.Tipper.tipper
    # print(np.shape(t))
    # print()
    
    # DatLimits = PerLimits
    # fmin = 1./DatLimits[1]
    # fmax = 1./DatLimits[0]

    # for ii in np.arange(fs[0]):

    #     if (abs(z[ii,:,:]).any()>1.e30):    z[ii,:,:] = 1.e30
    #     if (abs(z[ii,:,:]).any()<1.e-30):   z[ii,:,:] = 1.e-30
    #     if (abs(t[ii,:]).any()>1.e30):      t[ii,:] = 1.e30
    #     if (abs(t[ii,:]).any()<1.e-30):     t[ii,:] = 1.e-30


    # if no_err is True:
    #     # mt.Z.z_err = 0.0001*np.ones_like( freq_list = mt.Z.freqnp.real(mt.Z.z))
    #     # mt.Tipper.tipper_err = 0.0001*np.ones_like(np.real(mt.Tipper.tipper))
    #     mt.Z.z_err = 0.001 * np.real(mt.Z.z)
    #     mt.Tipper.tipper_err = 0.001 * np.real(mt.Tipper.tipper)
        

    zplot = mt_obj.plot_mt_response(plot_num = 2,fig_num = 2,
        res_limits = (1., 1000.),x_imits = (3.e-5, 300.))  
    # zplot.plot_num = 2
    # zplot.fig_num = 2 
    # zplot.res_limits = (1., 1000.)
    # zplot.x_imits = (3.e-5, 300.)
    # zplot.redraw_plot()
        
    for F in PlotFmt:
        zplot.save_plot(PltDir+name+strng+"_z"+F, fig_dpi=400)
 
    
    pplot= mt_obj.plot_phase_tensor()
    for F in PlotFmt:
          pplot.save_plot(PltDir+name+strng+"_pt"+F, fig_dpi=400)
    
        
    # plot.x_limits(pmin, pmax)
 # |  make_pt_cb(self, ax)
 # |  
 # |  set_period_limits(self, period)
 # |      set period limits
 # |      
 # |      :return: DESCRIPTION
 # |      :rtype: TYPE
 # |  
 # |  set_phase_limits(self, phase, mode='od')
 # |  
 # |  set_resistivity_limits(self, resistivity, mode='od', scale='log')
 # |      set resistivity limits
    
    
    # plot_num=plot_z,
    #                                    plot_tipper=plot_t,
    #                                    plot_pt=plot_p,
    #                                    x_limits=PerLimits,
    #                                    # res_limits=RhoLimits,
    #                                    # phase_limits=PhiLimits,
    #                                    # tipper_limits=Tiplimits,
    #                                    fig_dpi=400,
    #                                    xy_ls="",yx_ls="", det_ls="",
    #                                    # ellipse_colorby="skew",
    #                                    # ellipse_range = [-10.,10.,2.]
    #                                    )

# Finally save figure
# if PdfC:
#     util.make_pdf_catalog(PltDir, PdfList=pdf_list, FileName=PltDir+PdfCName)
