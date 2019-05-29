import os
import astropy.units as u
from astropy.io import fits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba.decorators import jit
import matplotlib.cm as cm
from scipy.optimize import curve_fit
import timeit
import seaborn as sns
import random
from glue.app.qt.application import GlueApplication as qglue
import math
import parameters
import time


H0=1
c=1
R0=1
pi4=math.pi/4.0
sqrt2=math.sqrt(2)

@jit
def alphaZ(x):
    alpha = pi4 - math.asin(1/sqrt2/(1+x))
    return alpha

@jit
def z_Out_Of_Alpha(alpha):
    z = 1.0/math.sin(pi4-alpha)/sqrt2-1.0
    return z

@jit
def alpha_Out_Of_d_HU(d_HU):
    alpha =  pi4-math.asin((1-d_HU)/sqrt2)
    return alpha

@jit    
def z_Out_Of_d_HU(d_HU):
    alpha= alpha_Out_Of_d_HU(d_HU)
    z = z_Out_Of_Alpha(alpha)
    return z

@jit
def d_HU_epoch(R0,z):
    alpha = alphaZ(z)
    d_HU=R0*(1- math.cos(alpha)+math.sin(alpha))
    return d_HU


def read_test_pyfits(filename, colname):
    with fits.open(filename, memmap=True) as hdul:
        data = (hdul[1].data[colname])
        return data.copy()


def read_nobs_pyfits(filename):
    with fits.open(filename, memmap=True) as hdul:
        data = (hdul[1].data)
        return np.shape(data)[0], hdul[1].columns.names


@jit
def get_BOSS_data(gal):
    nObs, cols = read_nobs_pyfits(gal)
    colnames=[]
    for x in cols:
        if x in ['ID', 'RA', 'DEC', 'Z', 'NZ', 'BOSS_SPECOBJ_ID',
                                         'BOSS_TARGET1', 'BOSS_TARGET2', 'EBOSS_TARGET0', 'ZOFFSET', 'TARGETOBJID',
                                         'OBJID', 'PLUG_RA', 'PLUG_DEC', 'Z']:
            colnames.append(x)
    ncols = len(colnames)
    myGalaxy = pd.DataFrame(data=np.zeros([nObs, ncols]), columns=colnames)
    for rowname in myGalaxy.columns:
        myGalaxy[rowname] = read_test_pyfits(gal, rowname).byteswap().newbyteorder()
    return fix_BOSS_data(myGalaxy)


@jit
def fix_BOSS_data(myGalaxy):
    print(myGalaxy.columns)
    pi4 = np.pi / 4.0
    sqrt2 = np.sqrt(2)
    npi=  np.pi/180.0 
    myGalaxy.DEC = myGalaxy.DEC.round(1)
    myGalaxy.RA = myGalaxy.RA.round(1)
    myGalaxy['CosRA'] = np.cos(myGalaxy.RA * npi)
    myGalaxy['SinRA'] = np.sin(myGalaxy.RA * npi)
    myGalaxy['CosDEC'] = np.cos(myGalaxy.DEC * npi)
    myGalaxy['SinDEC'] = np.sin(myGalaxy.DEC * npi)
    myGalaxy.Z = myGalaxy.Z.abs()
    myGalaxy['distance0'] = np.abs(zDistance(myGalaxy.Z))
    myGalaxy['distance'] = 0.0
    myGalaxy['density'] = 0.0
    myGalaxy['Me'] = myGalaxy['NZ']
    n = 3
    myGalaxy['alpha'] = np.round(pi4 - np.arcsin(1 / sqrt2 / (1 + np.abs(myGalaxy.Z))), n)
    myGalaxy['x'] = np.round(myGalaxy.alpha * myGalaxy.CosDEC * myGalaxy.CosRA, n)
    myGalaxy['y'] = np.round(myGalaxy.alpha * myGalaxy.CosDEC * myGalaxy.SinRA, n)
    myGalaxy['z'] = np.round(myGalaxy.alpha * myGalaxy.SinDEC, n)
    return myGalaxy

@jit
def fix_BOSS_data_noBinning(myGalaxy):
    print(myGalaxy.columns)
    pi4 = np.pi / 4.0
    sqrt2 = np.sqrt(2)
    npi=  np.pi/180.0 
    myGalaxy.DEC = myGalaxy.DEC.round(3)
    myGalaxy.RA = myGalaxy.RA.round(1)
    myGalaxy['CosRA'] = np.cos(myGalaxy.RA * npi)
    myGalaxy['SinRA'] = np.sin(myGalaxy.RA * npi)
    myGalaxy['CosDEC'] = np.cos(myGalaxy.DEC * npi)
    myGalaxy['SinDEC'] = np.sin(myGalaxy.DEC * npi)
    myGalaxy.Z = myGalaxy.Z.abs()
    myGalaxy['distance0'] = np.abs(zDistance(myGalaxy.Z))
    myGalaxy['distance'] = 0.0
    myGalaxy['density'] = 0.0
    myGalaxy['Me'] = myGalaxy['NZ']
#     n = 3
    n=5
#     myGalaxy['alpha'] = np.round(pi4 - np.arcsin(1 / sqrt2 / (1 + np.abs(myGalaxy.Z))), n)
    myGalaxy['alpha'] = np.abs(myGalaxy.Z)
    myGalaxy['x'] = np.round(myGalaxy.alpha * myGalaxy.CosDEC * myGalaxy.CosRA, n)
    myGalaxy['y'] = np.round(myGalaxy.alpha * myGalaxy.CosDEC * myGalaxy.SinRA, n)
    myGalaxy['z'] = np.round(myGalaxy.alpha * myGalaxy.SinDEC, n)
    return myGalaxy

@jit
def fix_Obj_data(myGalaxy):
    print(myGalaxy.columns)
    pi4 = np.pi / 4.0
    sqrt2 = np.sqrt(2)
    myGalaxy['Z'] = myGalaxy.Z_NOQSO
    myGalaxy.Z = myGalaxy.Z.abs()
    myGalaxy['distance0'] = np.abs(zDistance(myGalaxy.Z))
    myGalaxy['distance'] = 0.0
    myGalaxy['density'] = 0.0
    myGalaxy['Me'] = myGalaxy['Z_NOQSO']
    n = 4
    myGalaxy['alpha'] = np.round(pi4 - np.arcsin(1 / sqrt2 / (1 + np.abs(myGalaxy.Z))), n)
    myGalaxy['x'] = np.round(myGalaxy.alpha * myGalaxy.CX, n)
    myGalaxy['y'] = np.round(myGalaxy.alpha * myGalaxy.CY, n)
    myGalaxy['z'] = np.round(myGalaxy.alpha * myGalaxy.CZ, n)
    return myGalaxy


@jit
def zDistance(Z):
    pi4 = np.pi / 4.0
    sqrt2 = np.sqrt(2)
    return (pi4 - np.arcsin(1 / sqrt2 / (1 + np.abs(Z))))

@jit
def rDistance(Z):
    a = 13.58
    rDist=[]
    for x in Z:
        rDist.append(a / (1 - 4 / np.pi * zDistance(x)) ** 1.5 - a)
    # [a / (1 - 4 / np.pi * zDistance(x)) ** 1.5 - a for x in Z]
    return rDist


@jit
def get_distances(myGalaxy, indsGroup):
    autocorr = pd.Series()
    for i in indsGroup:
        d12 = myGalaxy[['x', 'y', 'z']].iloc[i] - myGalaxy[['x', 'y', 'z']]
        d12 = d12 * d12
        myGalaxy.distance = 0
        myGalaxy.density = 0
        myGalaxy.distance = np.round(np.sqrt((d12).sum(axis=1)), 3)
        res = myGalaxy.distance.value_counts()
        res = res * myGalaxy.groupby(['distance'])['Me'].sum()
        autocorr = autocorr.add(res, fill_value=0.0)
    return autocorr / (len(indsGroup))


@jit
def TwoPointCorr(mySet, chunckGalaxies, correctMe=False):
    gals = ['galaxy_DR12v5_CMASS_North.fits', 'galaxy_DR12v5_LOWZ_North.fits',
            'galaxy_DR12v5_CMASS_South.fits', 'galaxy_DR12v5_LOWZ_South.fits']
    a = mySet[0]
    b = mySet[1]
    myGalaxy = pd.concat([get_BOSS_data(parameters.sdssAddress + gals[a]), get_BOSS_data(parameters.sdssAddress + gals[b])])
    myGalaxy = myGalaxy.sort_values(by=['Z'])
    myGalaxy.reset_index(drop=True, inplace=True)
    start_time = timeit.default_timer()
    n = 5
    positions = np.linspace(0, 80, num=n, ) / 1000.0
    inds = []
    delta = 0.001
    for pos in positions:
        mask1 = myGalaxy.distance0 <= (pos + delta)
        mask2 = myGalaxy.distance0 >= (pos - delta)
        indsGroup = myGalaxy[mask1 & mask2].index.tolist()
        print(pos, len(indsGroup))
        if (len(indsGroup) != 0):
            inds.append((pos, indsGroup))

    autocorr = {}
    for pos, indsGroup in inds:
        limit = len(indsGroup)
        if limit > chunckGalaxies:
            random.shuffle(indsGroup)
            limit = chunckGalaxies
        posA = np.round(myGalaxy.iloc[indsGroup[0:limit]].distance0.mean(), 3)
        autocorr[posA] = get_distances(myGalaxy, indsGroup[0:limit])
        if correctMe:
            print('I am being corrected!!')
            autocorr[posA] = autocorr[posA] / (autocorr[posA].index + posA) ** 2
        elapsed = timeit.default_timer() - start_time
        print(pos, elapsed, 'myDataSet=', a, b)
    df = pd.DataFrame.from_dict(data=autocorr)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(method='bfill')
    for x in df.columns[0:]:
        shift=0
        if correctMe:
            print('I am being corrected!!')
            shift=np.int(700*x) 
        print(shift)
        df[[x]]= df[[x]].shift(shift)/df[[x]].iloc[10:].max()
    autocorr = pd.DataFrame.from_dict(data=autocorr, orient='index')
    return df, autocorr


@jit
def zDistribution(mySet, chunckGalaxies):
    gals = ['galaxy_DR12v5_CMASS_North.fits', 'galaxy_DR12v5_LOWZ_North.fits',
            'galaxy_DR12v5_CMASS_South.fits', 'galaxy_DR12v5_LOWZ_South.fits']
    a = mySet[0]
    b = mySet[1]
    myGalaxy = pd.concat([get_BOSS_data(parameters.sdssAddress + gals[a]), get_BOSS_data(parameters.sdssAddress + gals[b])])
    myGalaxy = myGalaxy.sort_values(by=['Z'])
    myGalaxy.reset_index(drop=True, inplace=True)
    start_time = timeit.default_timer()
    # Use distance to store rounded Z
    myGalaxy.distance = np.round(myGalaxy.Z, 3)
    zDensity = myGalaxy.groupby(['distance'])['Me'].count()
    # Use distance to store rounded distance0
    myGalaxy.distance = np.round(myGalaxy.distance0, 3)
    dDensity = myGalaxy.groupby(['distance'])['Me'].count()
    # Do not consider NZ since there cannot be a bias on the choice of NZ
    # dDensity=dDensity*myGalaxy.groupby(['distance'])['Me'].sum()
    dDensity = dDensity / np.max(dDensity.values)
    zDensity = zDensity / np.max(zDensity.values)
    return zDensity, dDensity

