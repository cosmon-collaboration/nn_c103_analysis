#
# Written by Ken McElvain 2022
# Translated from Mathematica version.
#
import math
import cmath
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import hankel1, hankel2, spherical_yn, spherical_jn

# incoming base wave
def spherical_hn1(n,z,derivative=False):
    """ Spherical Hankel Function of the First Kind """
    return spherical_jn(n,z,derivative=False)+1j*spherical_yn(n,z,derivative=False)

def spherical_hn2(n,z,derivative=False):
    """ Spherical Hankel Function of the Second Kind """
    return spherical_jn(n,z,derivative=False)-1j*spherical_yn(n,z,derivative=False)

# incoming free wave in channel L with wave number k
def hminus(L, k, Rp):
    kRp = k * Rp
    hn2 = spherical_hn2(L, kRp)
    return -1.0j * kRp * hn2

# outgoing free wave in channel L with wave number k
def hplus(L, k, Rp):
    kRp = k * Rp
    return 1.0j * kRp * spherical_hn1(L, kRp)

# package up the differential equation from README.md
def model(Rp, s, L, k, vf, vd):
    hm = hminus(L, k, Rp)
    hp = hplus(L, k, Rp)
    rhs = (s * np.conj(hm) - np.conj(hp)) * vf(Rp, vd) * (hm - hp * s)
    rhs *= (1 / (2.0j * k))
    return rhs

# Could define dfun  d rhs/ d Rp
# I don't have knowledge of vf'(Rp)

#
# Get S-matrix for potential vf
# L is the angular momentum channel
# k is the wave number
# vf(r, vd) is the potential function 
# vd is the config data for vf
# vR is the range at which vf[vR] is zero enough
#
#  S = Exp[2 i d], where d is the phase shift
#
# Put the Schrodinger equation in this form:
# \partial_x^2 \psi + k^2 \psi = V(r)
# This will likely mean that vf is   2 \mu /(hbar*c)^2  W(r) where
# W is the potential in MeV and k is in fm^-1
#
def get_smatrix_single(L, k, vf, vd, vR):
    # Start close to 0.  At 0 one could have 0/0 issues
    # start far enough away from 0 to avoid numerical issues.
    rrange = (1e-12, vR)
    s0 = [1.0+0.0j] # initial S-matrix is just 1
    # atol and rtol Tolerance didn't work, must be some error estimate failure
    # However, brute forcing the max step size does work.
    sol = solve_ivp(model, rrange, s0, args=(L, k, vf, vd), method="DOP853", atol = 1e-8, rtol = 1e-8, max_step=vR/100.0)
    # print(sol.t)
    y = sol.y[0]
    last = y[len(y)-1] # len(2) will be 2, entry 1 will be r=vR
    return last

def get_phase_single(L, k, vf, vd, vR):
    s = get_smatrix_single(L, k, vf, vd, vR)
    rt = cmath.polar(s) # convert to r exp(j*theta)
    # r should be 1.0, we want the angle theta
    return rt[1] / 2.0
