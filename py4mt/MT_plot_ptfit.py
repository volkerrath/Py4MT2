# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
# ---

import os
import sys
import warnings
import time

from sys import exit as error
from datetime import datetime

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
import pyproj as proj

mypath = ["/home/vrath/Py4MT/py4mt/modules/",
          "/home/vrath/Py4MT/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)

import mtplots as mtp

# import gdal
# import scipy as sc
# import vtk
# import pyvista as pv
# import pyvistaqt as pvqt
# import discretize
# import tarfile
# import pylab as pl
# from time import sleep

import mtplots as mtp
import modem as mod
import util as utl
from version import versionstrg

Strng, _ = versionstrg()
now = datetime.now()
print("\n\n"+Strng)
print("Plot Phase Tensor fit"+"\n"+"".join("Date " + now.strftime("%m/%d/%Y, %H:%M:%S")))
print("\n\n")

warnings.simplefilter(action="ignore", category=FutureWarning)


WorkDir =  r"/home/vrath/work/MT/Annecy/ANN25a_best/"
PredFile = r"/home/vrath/work/MT/Annecy/ANN25a_best/Ann25c_ZPT_200_Alpha01_NLCG_017"
ObsvFile = r"/home/vrath/work/MT/Annecy/ANN25a_best/Ann25_ZPTb"
PlotDir = WorkDir + 'Plots/'

print(' Plots written to: %s' % PlotDir)
if not os.path.isdir(PlotDir):
    print(' File: %s does not exist, but will be created' % PlotDir)
    os.mkdir(PlotDir)


PerLimits = (0.0001, 3.)
PhTLimitsXX = (-5., 5.)
PhTLimitsXY = (-1., 1.)
ShowErrors = True
ShowRMS = True


PlotFile = "Annecy_PhT_final"
PlotFormat = [".pdf", ".png", ".svg"]
PdfCatalog = True
if not ".pdf" in PlotFormat:
    error(" No pdfs generated. No catalog possible!")
    PdfCatalog = False
PdfCName = PlotFile


"""

"""
EPSG = 5015

start = time.time()

FF = ObsvFile
SiteObs, CompObs, DataObs, HeadObs = mod.read_data(FF+".dat")
obs_dat = DataObs[:, 6]
obs_err = DataObs[:, 7]
obs_per = DataObs[:, 0]
obs_cmp = CompObs
obs_sit = SiteObs
lat = DataObs[:,1]
lon = DataObs[:,2]
x = DataObs[:,3]
y = DataObs[:,4]
z = DataObs[:,5]

FF = PredFile
SiteCal, CompCal, DataCal, HeadCal = mod.read_data(FF+".dat")
cal_dat= DataCal[:, 6]
cal_per= DataCal[:, 0]
cal_cmp= CompCal
cal_sit = SiteCal

# Determine graphical parameter.
# print(plt.style.available)
plt.style.use("seaborn-paper")
mpl.rcParams["figure.dpi"] = 400
mpl.rcParams["axes.linewidth"] = 0.5
Fontsize = 10
Labelsize = Fontsize
Linewidth= 2
Markersize = 4
Grey = 0.7
Lcycle =Lcycle = (cycler("linestyle", ["-", "--", ":", "-."])
          * cycler("color", ["r", "g", "b", "y"]))

Sites = np.unique(SiteObs)

for s in Sites:
    print("Plotting site: "+s)
    site = (obs_sit==s)
    site_lon = lon[site][0]
    site_lat = lat[site][0]
    site_utmx, site_utmy = utl.proj_latlon_to_utm(site_lat, site_lon, utm_zone=EPSG)
    site_utmx = int(np.round(site_utmx))
    site_utmy = int(np.round(site_utmy))
    site_elev = z[site][0]

    cmp ="PTXX"
    cmpo = np.where((obs_cmp==cmp) & (obs_sit==s))
    PhTxxo = obs_dat[cmpo]
    PhTxxe = obs_err[cmpo]
    Perxxo = obs_per[cmpo]
    cmpc = np.where((cal_cmp==cmp) & (cal_sit==s))
    PhTxxc = cal_dat[cmpc]
    Perxxc = cal_per[cmpc]
    if ShowRMS:
        RnormPhTxx, ResPhTxx = utl.calc_resnorm(PhTxxo, PhTxxc, PhTxxe)
        nRMSPhTxx, _ = utl.calc_rms(PhTxxo, PhTxxc, 1.0/PhTxxe)


    cmp ="PTXY"
    cmpo = np.where((obs_cmp==cmp) & (obs_sit==s))
    PhTxyo = obs_dat[cmpo]
    PhTxye = obs_err[cmpo]
    Perxyo = obs_per[cmpo]
    cmpc = np.where((cal_cmp==cmp) & (cal_sit==s))
    PhTxyc = cal_dat[cmpc]
    Perxyc = cal_per[cmpc]
    if ShowRMS:
        RnormPhTxy, ResPhTxy = utl.calc_resnorm(PhTxyo, PhTxyc, PhTxye)
        nRMSPhTxy, _ = utl.calc_rms(PhTxyo, PhTxyc, 1.0/PhTxye)

    cmp ="PTYX"
    cmpo = np.where((obs_cmp==cmp) & (obs_sit==s))
    PhTyxo = obs_dat[cmpo]
    PhTyxe = obs_err[cmpo]
    Peryxo = obs_per[cmpo]
    cmpc = np.where((cal_cmp==cmp) & (cal_sit==s))
    PhTyxc = cal_dat[cmpc]
    Peryxc = cal_per[cmpc]
    if ShowRMS:
        RnormPhTyx, ResPhTyx = utl.calc_resnorm(PhTyxo, PhTxyc, PhTyxe)
        nRMSPhTyx, _ = utl.calc_rms(PhTyxo, PhTyxc, 1.0/PhTyxe)

    cmp ="PTYY"
    cmpo = np.where((obs_cmp==cmp) & (obs_sit==s))
    PhTyyo = obs_dat[cmpo]
    PhTyye = obs_err[cmpo]
    Peryyo = obs_per[cmpo]
    cmpc = np.where((cal_cmp==cmp) & (cal_sit==s))
    PhTyyc = cal_dat[cmpc]
    Peryyc = cal_per[cmpc]
    if ShowRMS:
        RnormPhTyy, ResPhTyy = utl.calc_resnorm(PhTyyo, PhTyyc, PhTyye)
        nRMSPhTyy, _ = utl.calc_rms(PhTyyo, PhTyyc, 1.0/PhTyye)

    cm = 1/2.54  # centimeters in inches
    fig, axes = plt.subplots(2,2, figsize = (16*cm, 12*cm))
    fig.suptitle(r"Site: "+s
                 +"\nLat: "+str(site_lat)+"   Lon: "+str(site_lon)
                 +"\nUTMX: "+str(site_utmx)+"   UTMY: "+str(site_utmy)
                 +" (EPSG="+str(EPSG)+")  \nElev: "+ str(abs(site_elev))+" m",
                 ha="left", x=0.1,fontsize=Fontsize-1)

    axes[0,0].plot(Perxxc, PhTxxc, "-r", linewidth =Linewidth)
    if ShowErrors:
        axes[0,0].errorbar(Perxxo,PhTxxo, yerr=PhTxxe,
                                linestyle="",
                                marker="o",
                                color="b",
                                linewidth=Linewidth,
                                markersize=Markersize)
    else:
        axes[0,0].plot(Perxxo, PhTxxo,
                       color="b",
                       marker="o",
                       linestyle="",
                       linewidth =Linewidth,
                       markersize=Markersize)


    axes[0,0].set_xscale("log")
    axes[0,0].set_xlim(PerLimits)
    if PhTLimitsXX != ():
        axes[0,0].set_ylim(PhTLimitsXX)
    axes[0,0].legend(["predicted", "observed"])
    axes[0,0].xaxis.set_ticklabels([])
    axes[0,0].tick_params(labelsize=Labelsize-1)
    axes[0,0].set_ylabel("PhTXX", fontsize=Fontsize)
    axes[0,0].grid("major", "both", linestyle=":", linewidth=0.5)
    if ShowRMS:
        nRMSr = np.around(nRMSPhTxx,1)
        StrRMS = "nRMS = "+str(nRMSr)
        axes[0,0].text(0.05, 0.05,StrRMS,
                           transform=axes[0,0].transAxes,
                           fontsize = Fontsize-2,
                           ha="left", va="bottom",
                           bbox={"pad": 2, "facecolor": "white", "edgecolor": "white" ,"alpha": 0.8} )


    axes[0,1].plot(Perxyc, PhTxyc, "-r", linewidth =Linewidth)
    if ShowErrors:
        axes[0,1].errorbar(Perxyo,PhTxyo, yerr=PhTxye,
                        linestyle="",
                        marker="o",
                        color="b",
                        linewidth=Linewidth,
                        markersize=Markersize)
    else:
        axes[0,1].plot(Perxyo, PhTxyo,
                       color="b",
                       marker="o",
                       linestyle="",
                       linewidth =Linewidth,
                       markersize=Markersize)
    axes[0,1].set_xscale("log")
    axes[0,1].set_xlim(PerLimits)
    if PhTLimitsXY != ():
        axes[0,1].set_ylim(PhTLimitsXY)
    axes[0,1].legend(["predicted", "observed"])
    axes[0,1].tick_params(labelsize=Labelsize-1)
    axes[0,1].set_ylabel("PhTXY", fontsize=Fontsize)
    axes[0,1].xaxis.set_ticklabels([])
    axes[0,1].tick_params(bottom="off", labelbottom="off")
    axes[0,1].grid("major", "both", linestyle=":", linewidth=0.5)
    if ShowRMS:
        nRMSr = np.around(nRMSPhTxy,1)
        StrRMS = "nRMS = "+str(nRMSr)
        axes[0,1].text(0.05, 0.05,StrRMS,
                           transform=axes[0,1].transAxes,
                           fontsize = Fontsize-2,
                           ha="left", va="bottom",
                           bbox={"pad": 2, "facecolor": "white", "edgecolor": "white" ,"alpha": 0.8} )


    axes[1,0].plot(Peryxc, PhTyxc, "-r", linewidth =Linewidth)
    if ShowErrors:
        axes[1,0].errorbar(Peryxo,PhTyxo, yerr=PhTyxe,
                                linestyle="",
                                marker="o",
                                color="b",
                                linewidth=Linewidth,
                                markersize=Markersize)
    else:
        axes[1,0].plot(Peryxo, PhTyxo,
                       color="b",
                       marker="o",
                       linestyle="",
                       linewidth =Linewidth,
                       markersize=Markersize)

    axes[1,0].set_xscale("log")
    axes[1,0].set_xlim(PerLimits)
    if PhTLimitsXY != ():
        axes[1,0].set_ylim(PhTLimitsXY)
    axes[1,0].legend(["predicted", "observed"])
    axes[1,0].tick_params(labelsize=Labelsize-1)
    axes[1,0].set_xlabel("Period (s)", fontsize=Fontsize)
    axes[1,0].set_ylabel("PhTYX", fontsize=Fontsize)
    axes[1,0].grid("major", "both", linestyle=":", linewidth=0.5)
    if ShowRMS:
        nRMSr = np.around(nRMSPhTyx,1)
        StrRMS = "nRMS = "+str(nRMSr)
        axes[1,0].text(0.05, 0.05,StrRMS,
                           transform=axes[1,0].transAxes,
                           fontsize = Fontsize-2,
                           ha="left", va="bottom",
                           bbox={"pad": 2, "facecolor": "white", "edgecolor": "white" ,"alpha": 0.8} )

    axes[1,1].plot(Peryyc, PhTyyc, "-r", linewidth =Linewidth)
    if ShowErrors:
       axes[1,1].errorbar(Peryyo,PhTyyo, yerr=PhTyye,
                                linestyle="",
                                marker="o",
                                color="b",
                                linewidth=Linewidth,
                                markersize=Markersize)
    else:
        axes[1,1].plot(Peryyo, PhTyyo,
                       color="b",
                       marker="o",
                       linestyle="",
                       linewidth =Linewidth,
                       markersize=Markersize)
    axes[1,1].set_xscale("log")
    axes[1,1].set_xlim(PerLimits)
    if PhTLimitsXX != ():
        axes[1,1].set_ylim(PhTLimitsXX)
    axes[1,1].legend(["predicted", "observed"])
    axes[1,1].tick_params(labelsize=Labelsize-1)
    axes[1,1].set_xlabel("Period (s)", fontsize=Fontsize)
    axes[1,1].set_ylabel("PhTYY", fontsize=Fontsize)
    axes[1,1].grid("major", "both", linestyle=":", linewidth=0.5)
    if ShowRMS:
        nRMSr = np.around(nRMSPhTyy,1)
        StrRMS = "nRMS = "+str(nRMSr)
        axes[1,1].text(0.05, 0.05,StrRMS,
                           transform=axes[1,1].transAxes,
                           fontsize = Fontsize-2,
                           ha="left", va="bottom",
                           bbox={"pad": 2, "facecolor": "white", "edgecolor": "white" ,"alpha": 0.8} )

    fig.tight_layout()

    for F in PlotFormat:
        plt.savefig(PlotDir+PlotFile+"_"+s+F, dpi=400)


    plt.show()
    plt.close(fig)

if PdfCatalog:
    utl.make_pdf_catalog(PlotDir, PdfCName)
