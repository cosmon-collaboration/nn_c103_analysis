#!/usr/bin/env python3
import matplotlib.pyplot as plt
import os
import numpy as np
import gvar as gv
import vegas
import h5py as h5
import lsqfit
import scipy.constants as const
import copy
import smatrix

# Ken:  Why does HBARC have the electron charge in it???
HBARC   = const.hbar * const.c / const.e * 1e15 / 1e6
print(f"old HBARC = {HBARC}") # seems to work (what are the orig units??)
# documentation on scipy.constants is poor - no units spec
hc   = 197.32698045930246 # MeV * fm, From Mathematica
HBARC = hc
print(f"hc = {hc}")
a_fm    = 0.0849  # lattice spacing in fm
scale   = HBARC / a_fm

# don't try to fit mpi - it is a known value
mpilatg = gv.gvar('0.310810(95)')  # 1/a
mpilat = mpilatg.mean
mpi = mpilat / a_fm    # In inverse fm for mult with r in fm
mpimev = hc * mpilat / a_fm 
print(f"mpimev = {mpimev} MeV, mpi = {mpi} fm^-1")

mnucleonmev=1400.0  # Nucleon mass in MeV

# Ken:  why are we overriding to 1?    
# Andre's Answer:   hack to work in lattice units instead of MeV fm
# a_fm = 1.
# scale = 1

fitgo=True
maxt = 18
t_V_fit=maxt # time slice to plot
trange = range(maxt-1,maxt+1)

maxfm = 1.5
maxlu = maxfm / a_fm

# Top level keys
# R_pn_TRIP V0_pn_TRIP 
# V_hal_pn_TRIP   -   This must be the data
# Vdt_pn_TRIP Vdtdt_pn_TRIP 
# r_a  - must be locations  - multiples a
#
# for a given x, find the set of samples close enough to make an
# important contribution to the density at x.
def getLocalDensityXset(x):
    return densityXset[densitya * (densityXset - x)**2 < 8.0];

# Should make a class and instantiate it
def setDensity():
    global densitya,densityXset,densityXmax,densityXmin,densityXbinsize,densityCenters,densityScale
    # i_r_fit = [i_r for i_r,r in enumerate(r_a)]
    densityXset = r_a
    densityXmax = densityXset.max()
    densityXmin = densityXset.min()
    densityXbinsize = 0.05 / a_fm # in lattice units
    densitya = -4.0 / (densityXbinsize**2) # spread of samples
    densityCenters = [ getLocalDensityXset(x) for x in np.arange(densityXmin + densityXbinsize/2, densityXmax + densityXbinsize, densityXbinsize)]
    print(f"density: x range {densityXmin} lu to {densityXmax} lu")
    densityScale = 1.0
    # maxlu is the end of the range used for fitting
    d = density((maxlu + densityXmin)*0.5)
    densityScale = 1/d

# compute relative density of samples using Gaussian smearing
# densityCenters has samples close enough to matter for the bin
def density(x):
    xb = int(np.floor((x - densityXmin) / densityXbinsize))
    if xb < 0:
        return 0.0
    tmp = densityCenters[xb] - x
    return densityScale * sum(np.exp(densitya * (tmp*tmp)))

# Extract data to fit as hash table with key f"pn_trip_t{t}" where t is the time
data = {}
with h5.File('c103_V_hal_pn_TRIP.h5') as f5:
    r_a = f5['r_a'][()]
    setDensity()
    pn_trip = f5['V_hal_pn_TRIP'][()]
    # for t in range(pn_trip.shape[1]):
    # for t in range(2,19):
    for t in trange:
        # string formatting for hash table data
        # data['pn_trip_t%d' %t] = pn_trip[:,t]
        print(f"Adding data for pn_trip_t{t}")
        tript = pn_trip[:,t]
        # tript = np.array([ gv.gvar(tript[i].mean, tript[i].sdev * density(r_a[i])) for i in range(len(tript))])
        data[f"pn_trip_t{t}"] = tript # pn_trip[:,t]

    #print("data = ", data)
    # Get density for each 
    sdevscale = np.sqrt(np.array([density(r_a[i]) for i in range(len(r_a))]))
    data_gv = gv.dataset.avg_data(data, bstrap=True)
    for t in trange:
        tag = f"pn_trip_t{t}"
        dv = data_gv[tag]
        data_gv[tag] = np.array([ gv.gvar(dv[i].mean, dv[i].sdev * sdevscale[i]) for i in range(len(dv))])
    
    print("data_gv = ", data_gv)
    #quit()


def plotDensity():
    plt.style.use('_mpl-gallery')
    plt.ion() # Enable interactive mode
    fig,ax = plt.subplots()
    y = np.array([density(x) for x in densityXset])
    ax.plot(densityXset, y, linewidth=2.0)
    ax.set(xlim=(0, 20), xticks=np.arange(1, 20), ylim=(0, 5), yticks=np.arange(0, 5))
    plt.show()

#plotDensity()
#quit()

def V_gsum(x, p):
    w = p['w'] # gaussian width
    a0 = p['a0']
    a1 = p['a1']
    a2 = p['a2']
    return np.exp(- x^2 / a) * (a0 + a1 * x + a2 * x^2)
    

# Av18 had c = 2.1 fm^-2 for physical pion mass in the Yukawa regulator
# This is compared to  mpi/hc of 0.535448162 fm^-1 at the physical point.
# We can use the ratio  c/(mpi/hc)^2 ~= 7.32 or sqrt(c)/(mpi/hc) ~= 2.7
# This is a function of the ratio of pion mass to the next heaviest meson.  
# 700 MeV pion we should probably use about a factor of 2 because the ratio
# won't be as large.  (Do we have data sitting around for the "sigma" mass?)
#
# Globals
#  mpi       Actually mpi / hc  with units fm^-1
#
# Parameters
#  'tfpiNN2':    Yukawa prefactor is tfpiNN2 * mpimev, not including sigma.sigma or tau.tau
#  'c':       regulator cutoff parameter.   physical value is 2.1, we will have a larger (closer in) value
#  Short range parts - maybe we just use Av18 values as priors means since as heavier mesons shift less than pion.
#  'wc':     Woods-Saxon constant   (not sure about prior)
#  Constrainted automatically 'wl':     Woods-Saxon linear     (not sure about prior)
#  'wq':     Woods-Saxon quadratic  (not sure about prior)
#  'r0r':    sqrt of radius for Wood-Saxon  (physical r0r^2 = 0.5 fm)
#  'a':      slope for Wood-Saxon   (physical 0.2 fm)
#
#  Still need to implement condition in Av18, that V_wsopep'(r)|_{r=0}  in 1S0
# This constraint kills off the first derivative of the potential at the origin.
# It is implemented from the contraint by fixing 'wl' as function of ‘wc’.
# See eqn 24 in Av18 paper.
#
V_opep_c = 2.1
V_opep_tfpiNN2 = 0.0
def V_opep_reg(r, p):
    c = p.get('c', V_opep_c)
    return (1 - np.exp(-c*r*r))**4

def V_wsopep(r, p):
    a = p['a']
    r0 = p['r0r']
    r0 *= r0;
    ws = p['ws']
    wq = p['wq']
    # support these being in the fit or hardwired
    if 'c' in p:
        c = p['c']
    else:
        c = V_opep_c
    if 'tfpiNN2' in p:
        tfpiNN2 = p['tfpiNN2']
    else:
        tfpiNN2 = V_opep_tfpiNN2

    z = mpi * r
    # prefactor is tfpiNN2 * mpimev * (tau.tau) * (sigma.sigma)
    # -3 for (tau.tau)*(sigma.sigma)  T=0, S=1   (sigma.sigma) = 4S - 3
    ya = tfpiNN2 * mpimev * (-3.0)
    # stick in tiny value in denominator to git rid of suppressed divide by 0
    t1 = ya * (np.exp(-z) / (z+1e-10)) * V_opep_reg(r, p)
    # compute wl that gives V_wsopep'(0, p) == 0
    wl =  (1 + np.exp(-r0/a))*(3*c*hc*tfpiNN2 + ws * (1.0/np.cosh(0.5 * r0/a)**2)/(4.0*a))
    w = 1.0 / (1.0 + np.exp((r - r0) / a))  # Woods-Saxon
    t2 = (ws + wl * r + wq * r*r) * w   # mult woods-saxon by quadratic like Av18
    return t1 + t2

# Wrapper to convert potential V_wsopep(r, p) in MeV to fm^-2
def W_wsopep(r, p):
    return (mnucleonmev / HBARC**2)*V_wsopep(r, p)

def V_opep(r, p):
    tfpiNN2 = p['tfpiNN2']
    if 'c' in p:
        c = p['c']
    else:
        c = V_opep_c

    z = mpi * r
    # prefactor is tfpiNN2 * mpimev * (tau.tau) * (sigma.sigma)
    # -3 for (tau.tau)*(sigma.sigma)  T=0, S=1   (sigma.sigma) = 4S - 3
    ya = tfpiNN2 * mpi * hc * (-3.0)
    # stick in tiny value in denominator to git rid of suppressed divide by 0
    t1 = ya * (np.exp(-z) / (z+1e-10)) * V_opep_reg(r, p)
    return t1

opep_p = {
    # 'tfpiNN2' : gv.gvar(0.024 * 2.5**2, 0.100),
    'tfpiNN2' : gv.gvar(1.2 * 0.024 * (722.4/134.0)**2, 0.200),
    # 'c' : gv.gvar(2.66, 0.2)
    'c' : gv.gvar(3.50, 0.2)
    # 'c' : gv.gvar(2.1, 0.2)
}
wsopep_p = {
    'a' : gv.gvar(0.10, 0.05),
    'r0r' : gv.gvar(0.05, 0.1),
    #'ws'  : gv.gvar(2500.0, 500.0),
    #'wq'  : gv.gvar(1000.0, 100.0),
    'ws'  : gv.gvar(3500.0, 500.0),
    # 'wq'  : gv.gvar(400.0, 100.0)
    'wq'  : gv.gvar(800.0, 100.0)
    # Use Andre's value for mpi
    # 'mpi' : gv.gvar(722.3933898298679/hc, 20.0/hc),

    # Use physical point value 0.024, compensate for mpi/f_pi ratio change
    # contains   mpi^2 (g_A / f_\pi)^2
    # At physical point mpi/f_\pi ~ 1.6, in this data mpi/f_\pi = 4.0
    # With 700 MeV pions,    mpi/f_\pi ~ 2.5  (Check this again, Andre uses F_\pi = f_\pi/\sqrt{2})
    # 'tfpiNN2' : gv.gvar(0.024 * (722.4/134.0)**2, 0.500)
    # 'tfpiNN2' : gv.gvar(0.024 * (722.4/134.0)**2, 0.500)
    # 'tfpiNN2' : gv.gvar(0.024 * (722.4/134.0)**2, 0.500)
    # 'tfpiNN2' : gv.gvar(0.024 * 2.5**2, 0.100)
}

wsopep_phys = {
    'a' : 0.2,
    'r0r' : np.sqrt(0.5),
    'ws'  : 2605.2682,
    'wq'  : 441.973,
    'mpi' : 0.68403,
    # make cr twice mpi but sloppy.  Cut off pion when sigma and others kick in
    'cr' : np.sqrt(2.1),
    'tfpiNN2' : 0.024
}

#for tr in [0.1, 0.5, 1.0]:
#    print(f"wsopep({tr}) = {V_wsopep(tr, wsopep_phys)}")
#quit()


def V_wsopep_plot(t, p, extend = False):
    plt.ion() # Enable interactive mode
    fig = plt.figure() # creates a figure
    ax = plt.axes([.12,.12,.87,.87]) # can't find docs yet
    # evaluate associated function, in this case V_wsopep on r data with fit values
    # Becaues of gvars this will yield mean and uncertainty as the result v_fit
    if extend:
        xset = np.arange(0.01, densityXmax, 0.05)
    else:
        xset = np.arange(densityXmin, densityXmax, 0.05)

    rset = xset * a_fm
    v_fit = np.array([V_wsopep(r, p) for r in rset])
    # extract mean and deviation for configured model
    y  = np.array([k.mean for k in v_fit])
    dy = np.array([k.sdev for k in v_fit])
    # fill in line with uncertainty
    ax.fill_between(rset, y-dy, y+dy, color='k', alpha=.33)

    y  = [k.mean*scale for k in data_gv[f"pn_trip_t{t}"]]
    dy = [k.sdev*scale for k in data_gv[f"pn_trip_t{t}"]]
    ax.errorbar(r_a * a_fm, y, yerr=dy, linestyle='None',
                mfc='None', alpha=.3, #marker='o', 
                label=r'$t=%d$' %t, color='r')

    ax.axhline(0, color='k')
    plt.ylim(-100.0, 2500.0)
    plt.xlim(0.0, 1.6)
    plt.title(f"Potential at time {t}")
    plt.xlabel("Radial distance [fm]")
    plt.ylabel("Potential [MeV]")
    ax.text(0.5, 1000.0, f"potential at $time={t}$", fontsize=16)
    plt.ioff()
    #   plt.show()

def feffrng(p):
    plist = np.array([smatrix.get_phase_single(0, k, W_wsopep, p, 2.0) for k in klist])
    kcotd = np.array([ k*(1.0/np.tan(d)) for k,d in zip(klist, plist)])
    ercoef = np.polynomial.polynomial.Polynomial.fit(k2list, kcotd, 3, symbol="k2")
    return ercoef.coef
    
def get_effrangeexpansion(gp):
    expval = vegas.PDFIntegrator(gp)
    result = expval(feffrng, neval=20, nitn=5)
    return result

def report_phase(t, gp):
    print("phase shift gen from distribution: ", gp)
    global elist, klist, k2list
    elist = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]);
    klist = np.array([np.sqrt(mnucleonmev*e)/HBARC for e in elist])
    k2list = np.array([k*k for k in klist])
    result = get_effrangeexpansion(gp)
    print("result = ", result)
    return
    for i in range(2):
        p = gv.sample(gp)
        print("phase shift gen with sample: ", p)
        plist = np.array([smatrix.get_phase_single(0, k, W_wsopep, p, 3.0) for k in klist])
        print(f"sample{i}: phases({t}) = ", plist)
        kcotd = np.array([ k*(1.0/np.tan(d)) for k,d in zip(klist, plist)])
        print(f"kcotd{i}:  ", kcotd)
        ercoef = np.polynomial.polynomial.Polynomial.fit(k2list, kcotd, 3, symbol="k2")
        print("ercoef = ", ercoef)
        a = -1.0 / ercoef.coef[0]
        reff = ercoef.coef[1]
        z = reff / a
        if z < 0.0 or z > 1.0:
            print(f"No bound state as (reff / a) = {z} < 0 or > 1")
        else:
            be = (1.0 / reff) * (1.0 - np.sqrt(1.0 - 2.0 * z))
            print(f"Binding energy is {be}")

#
# Strategy:   
#    Fit opep to long range decay region from 0.5 fm to 1.5 fm
#    With tfpiNN2 and regulator parameter c now fixed
#    Fit wsopep from 0 to 1.5
#
def fit_wsopep():
    fit_results = {}
    # Advice from Andre is that discretization errors show up as small r.
    # Use r_min = 2.
    # First lets fit the opep from
    if 'c' in opep_p:
        r_min_opep = 0.5 / a_fm # to start of roll off to be sensitive to c
    else:
        r_min_opep = 0.5 / a_fm
    r_max_opep = maxlu
    print(f"opep fit range {r_min_opep} to {r_max_opep}")
    r_min_wsopep = 0.0 / a_fm
    r_max_wsopep = maxlu
    print(f"wsopep fit range {r_min_wsopep} to {r_max_wsopep}")
    for t in trange:
        t_c = t
        print(f" r_minmax = {r_min_opep}, {r_max_opep}")
        print(f"Fitting opep from r={r_min_opep*a_fm} to r={r_max_opep*a_fm}")
        i_r_fit = [i_r for i_r,r in enumerate(r_a) if r > r_min_opep and r < r_max_opep]
        fit_x = r_a[i_r_fit] * a_fm
        # print("fit_x = ", fit_x)
        # t_V_fit currently set to 10,   need to loop and fit
        fit_y = data_gv[f"pn_trip_t{t}"][i_r_fit] * scale
        # print("fit_y = ", fit_y)
        p = opep_p
        p0 = dict()
        for key,v in p.items():
            p0[key] = v.mean
        opep_fit = lsqfit.nonlinear_fit(udata=(fit_x,fit_y), prior=p, p0=p0, fcn=V_opep)
        # opep_fit = lsqfit.nonlinear_fit(udata=(fit_x,fit_y), prior=p, fcn=V_opep)
        print(f"t = {t}")
        print(opep_fit)
        fit_results[t] = opep_fit
        print("Fit Results: ", opep_fit.p)
        # hack globals to configure wsopep with opep parameters
        global V_opep_c, V_opep_tfpiNN2
        if 'c' in opep_fit.p:
            V_opep_c = opep_fit.p['c'].mean
        V_opep_tfpiNN2 = opep_fit.p['tfpiNN2'].mean
        print(f"c={V_opep_c}, tfpiNN2={V_opep_tfpiNN2}")
        print(f"Fitting wsopep from r={r_min_wsopep*a_fm} to r={r_max_wsopep*a_fm}")
        i_r_fit = [i_r for i_r,r in enumerate(r_a) if r > r_min_wsopep and r < r_max_wsopep]
        fit_x = r_a[i_r_fit] * a_fm
        # t_V_fit currently set to 10,   need to loop and fit
        fit_y = data_gv[f"pn_trip_t{t}"][i_r_fit] * scale
        p = wsopep_p
        p0 = dict()
        for key,v in p.items():
            p0[key] = v.mean
        wsopep_fit = lsqfit.nonlinear_fit(udata=(fit_x,fit_y), prior=p, p0=p0, fcn=V_wsopep)
        fit_results[t] = wsopep_fit
        print("wsfit = ", wsopep_fit)
        tp = dict()
        for key,v in wsopep_fit.p.items():
            tp[key] = v
        for key,v in opep_fit.p.items():
            tp[key] = v
        fit_results[t] = [t, tp, wsopep_fit, opep_fit]  # save everything
        print("twsfit = ", tp)
        V_wsopep_plot(t, fit_results[t][1], True)
        report_phase(t, fit_results[t][1])

    # V_wsopep_plot(t_V_fit, tp, True)
    return
    p = wsopep_p
    p0 = dict()
    for key,v in p.items():
        p0[key] = v.mean
    # tmp_fit = lsqfit.nonlinear_fit(udata=(fit_x,fit_y), prior=p, p0=p0, fcn=V_wsopep)
    tmp_fit = lsqfit.nonlinear_fit(udata=(fit_x,fit_y), prior=p, fcn=V_wsopep)
    print("fit : ", tmp_fit)
    fit_results[t_V_fit] = tmp_fit
    V_wsopep_plot(r_min, t_V_fit, fit_results, True)

fit_wsopep()
quit()
