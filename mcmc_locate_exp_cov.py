#---------------------------------------------------

import numpy as np

def npa_get_starting_model(rng, xmin, xmax, ymin, ymax, zmin, zmax, tmin, tmax, vpmin, vpmax):
    
    p=rng.random()
    X0=xmin+p*(xmax-xmin)
    p=rng.random()
    Y0=ymin+p*(ymax-ymin)
    p=rng.random()
    Z0=zmin+p*(zmax-zmin)
    p=rng.random()
    T0=tmin+p*(tmax-tmin)
    p=rng.random()
    VP0=vpmin+p*(vpmax-vpmin)
    
    return X0, Y0, Z0, T0, VP0

#---------------------------------------------------

from math import sqrt

def npa_LOCATE_syn(nstat_sele, stlo, stla, conv, X0, Y0, Z0, T0, VP0):
    
    SYN_DATA_P = []
    istat=0
    while istat < nstat_sele:
        x1=stlo[istat]
        y1=stla[istat]
        z1=0.0
        dist= sqrt( (conv*(X0-x1))**2 + (conv*(Y0-y1))**2 + (Z0-z1)**2 )
        theo_P = T0 + (1.0/VP0) * dist
        #print(istat,dist,T0,VP0,theo_P)
        SYN_DATA_P.append(theo_P)
        istat += 1

        
    return SYN_DATA_P

#---------------------------------------------------

SYN_DATA_P = []

def npa_log_likelihood(nstat_sele, SYN_DATA_P , Ppick, icov):
    
    lppd=0
    id0=0
    while id0 < nstat_sele:
        
        p=0.0
        istat_min=id0-1
        istat_max=id0+1
        if id0 == 0:
            istat_min=0
        if id0 == nstat_sele-1:
            istat_max=nstat_sele-1
           
        jd0=istat_min
        while jd0 <= istat_max:
            
            a=icov[id0,jd0]
            b=(SYN_DATA_P[jd0] - Ppick[jd0])
            p = p + a*b
            jd0+=1

        lppd = lppd + p*((SYN_DATA_P[id0] - Ppick[id0]))
        
        id0+=1
        
    return lppd


#---------------------------------------------------

import numpy as np

def npa_candidate_model(rng, X0, Y0, Z0, T0, VP0, xmin, xmax, ymin, ymax, zmin, zmax, tmin, tmax, vpmin, vpmax):
    
    
    X_cand=X0
    Y_cand=Y0
    Z_cand=Z0
    T_cand=T0
    VP_cand=VP0
    
    p=rng.random()
    
    if p < 0.20:
# Pertub X
        X_cand=npa_uniform_rand_walk(rng, X0,xmin,xmax)

    if p >= 0.20 and p < 0.40:
# Perturb Y
        Y_cand=npa_uniform_rand_walk(rng, Y0,ymin,ymax)

    if p >= 0.40 and p < 0.60:
# Perturb Z
        Z_cand=npa_uniform_rand_walk(rng, Z0,zmin,zmax)

    if p >= 0.60 and p < 0.80:
# Perturb VP
        VP_cand=npa_uniform_rand_walk(rng, VP0,vpmin,vpmax)
            
    if p >= 0.80:
# Perturb T0        
        T_cand=npa_uniform_rand_walk(rng, T0,tmin,tmax)
        
    return X_cand, Y_cand, Z_cand, T_cand, VP_cand


#---------------------------------------------------


def npa_uniform_rand_walk(rng, x,xmin,xmax):

    x_new=x
    
    sc=0.1
    halfwidth=0.5*sc*(xmax-xmin)
    xcandhigh=x+halfwidth
    if xcandhigh > xmax:
        xcandhigh=xmax
    xcandlow=x-halfwidth
    if xcandlow <xmin:
        xcandlow=xmin
    maxdev=xcandhigh-xcandlow
    dev=rng.random()
    xcand=xcandlow+dev*maxdev
    xhigh=xcand+halfwidth
    if xhigh > xmax:
        xhigh=xmax
    xlow=xcand-halfwidth
    if xlow < xmin:
        xlow=xmin
    probacc=(xcandhigh-xcandlow)/(xhigh-xlow)
    dev=rng.random()
    if probacc > dev:
        x_new=xcand

    return x_new


#---------------------------------------------------

import math
import numpy as np

def npa_metropolis(rng, lppd, lppd0):
    
    p=rng.random() 
    
    a=lppd-lppd0
    if a > 50.0:
        a = 50.0
    if a < -50.0:
        a = -50.0
                
    alpha = math.exp( -0.5 * a )
    
    if alpha >= 1.0:
        
        Accepted = 1
      
    if alpha < 1.0:
        
        if p <= alpha:
            
            Accepted = -1
            
        else:
        
            Accepted = 0
            
    return Accepted


#---------------------------------------------------


