"""Help function for fitting HAL potential data."""
import numpy as np
from scipy.special import gamma as gammaf
import scipy.constants as const
from scipy.optimize import minimize
from scipy.special import spherical_jn, spherical_yn, hermite
import gvar as gv
import lsqfit
import plotly.graph_objects as go
import pandas as pd
from tqdm.notebook import tqdm
from lsqfitgui.plot.uncertainty import plot_gvar
from lsqfitgui.plot.fit import plot_fit, plot_residuals

HBARC = const.hbar * const.c / const.e * 1e15 / 1e6

M_PI0 = 714 / HBARC
M_N0 = 1612 / HBARC

class Extrap:
    """ class for performing infinite t extrapolations """
    
    def __init__(self, nstates=2, tmin=10, tmax=20):
        self.nstates = nstates
        self.tmin = tmin
        self.tmax = tmax

    def get_priors_t(self,V0):
        prior = gv.BufferDict()
        prior["V_0"]=gv.gvar(V0.mean,V0.sdev*2)

        for i in range(1,self.nstates):
            prior[f"E_{i}"]=gv.gvar(np.log(2*i*3.0),1.0)
            prior[f"A_{i}"]=gv.gvar(1.0,1.0)
        return prior

    def model_t(self,t,en,A):
        return A*np.exp(-np.exp(en)*t)

    def fit_t(self,t, p):
        pm = p["V_0"] + 0*t
        
        for i in range(1,self.nstates):
            pm = pm + self.model_t(t,p[f"E_{i}"],p[f"A_{i}"])
        return pm

def stability_plot(dat,rpot,afm,tmin,tmax,nstates,cmax):
    """ plot infinite t extrapolated potential at a given r = rpot and n = nstates terms in fit function as a function of tmin; tmin and nstates should be given as lists
"""
    fit = dict()
    for tm in tmin:
        for ns in nstates:
            fitter_t = Extrap(ns,tm,tmax)
            x = np.arange(tm,tmax)*afm
            y = gv.dataset.avg_data([[dat[rpot,tt,cc] for tt in range(tm,tmax)] for cc in np.arange(cmax)],bstrap=True)
            prior_t = fitter_t.get_priors_t(gv.gvar(np.mean([[dat[rpot,tt,cc] for tt in range(tm,tmax)] for cc in np.arange(cmax)]),np.std([[dat[rpot,tt,cc] for tt in range(tm,tmax)] for cc in np.arange(cmax)])))
            fit[tm,ns] = lsqfit.nonlinear_fit((x, y), fcn=fitter_t.fit_t, prior=prior_t)
    fig = go.Figure()
    syms = ['square-open','circle-open','circle']
    clrs = ['red','green','blue','orange','cyan','purple','gray']
    for ns in nstates:
        plot = plot_gvar([tm+0.1*(ns-1) for tm in tmin], [fit[tm,ns].p["V_0"] for tm in tmin], kind="errorbars",  scatter_kwargs={"name": f"nstates={ns}","marker":dict(
            symbol=syms[2-1],
            size=8,
            line=dict(width=2),
            color=clrs[ns]
            )}
                         )
        fig.add_trace(plot.data[0])
    fig_all = go.Figure(data = fig.data)
    return fig_all
        
class Fit:
    """ class for fitting V(r) versus r """
    
    def __init__(self, model="gaussian", pow_c=1, pow_lr=1):
        self.model = model
        self.pow_c = pow_c
        self.pow_lr = pow_lr
        self.rmax = 4.128
        
    def get_priors(self,p):

        prior = gv.BufferDict()
        
        for i in range(self.pow_lr):
            prior[f"c_{i}"] = p[f"c_{i}"]
            prior[f"a_{i}"] = p[f"a_{i}"]
            prior[f"m_{i}"] = p[f"m_{i}"]
        for i in range(self.pow_c):
            if self.model == "gaussian":
                prior[f"b_{i}"] = p[f"b_{i}"]
                prior[f"gamma1_{i}"] = p[f"gamma1_{i}"]
            if self.model == "harmosc":
                prior[f"b_{i}"] = p[f"b_{i}"]
                prior[f"gamma1_0"] = p[f"gamma1_0"]
            if self.model == "harmosc1D":
                prior[f"b_{i}"] = p[f"b_{i}"]
                prior[f"gamma1_{i}"] = p[f"gamma1_{i}"]
            if self.model == "herm_g2":
                prior[f"b_{i}"] = p[f"b_{i}"]
                prior[f"gamma1_0"] = p[f"gamma1_0"]
                prior[f"gamma2_{i}"] = p[f"gamma2_{i}"]
            if self.model == "osc":
                prior[f"b_{i}"] = p[f"b_{i}"]
                prior[f"gamma1_{i}"] = p[f"gamma1_0"]
                prior[f"gamma2_{i}"] = p[f"gamma2_{i}"]
                prior[f"ph_{i}"] = p[f"ph_0"]
        return prior

    def gaussian(self, x, gamma, b):
        """Normal function centered around zero."""
        return b * np.exp(-((np.exp(gamma))**2 * x ** 2))
    
    def ope(self, x, m, a, c):
        """Implements OPE in S-wave coordinate space times r.
        Fit parameters:
        m: pion mass
        a: coefficient in front of exponential                                                                              
    """
        out = ((1-self.gaussian(x,c,1))**2)*(-np.exp(a) * np.exp(-np.exp(m) * x)/ x )
        return out

    def lagL0(self,q,n):
        q2 = q*q
        q4 = q2*q2
        if n == 1:
            rslt = 1
        elif n == 2:
            rslt = (3.0/2.0) - q
        elif n == 3:
            rslt = (1.0/8.0)*(15.0 - 20*q + 4*q2)
        elif n == 4:
            rslt = (1/48.0)*(105.0 - 210.0*q + 84.0*q2 - 8*q2*q)
        elif n == 5:
            rslt = (1.0/384.0)*(945.0 - 2520.0*q + 1512.0*q2 - 288.0*q2*q + 16*q4)
        else:
            raise ValueError('n is too large')
        return rslt
    
    def herm_w(self, x, n):
        if n == 0:
            return 1
        elif n == 1:
            return x
        elif n == 2:
            return x**2 - 1
        else:
            return 0

    def harmosc(self, x, gamma, b, n):
        return b*(np.exp(gamma))**(-3/2) * np.sqrt(2*gammaf(n+2)/gammaf(n+2+1/2))*np.exp(-(x**2 / (2*np.exp(gamma)**2))) * self.lagL0(x**2/(np.exp(gamma)**2),n+1)

    def harmosc1D(self, x, gamma, b, n):
        return b * np.exp(-(np.exp(gamma)**2) * x**2) * self.herm_w(np.exp(gamma)*x,n)

    """ Alternate fitting functions considered """
    
    def herm_g2(self, x, gamma1, gamma2, b, n):
        herm_w = hermite(n)
        return b * np.exp(-(np.exp(gamma1)**2) * x**2) * self.herm_w(np.exp(gamma2)*x,n)

    def osc(self, x, gamma1, gamma2, ph, b):
        return b * np.cos(gamma2*x + ph) * np.exp(-(np.exp(gamma1)**1)*x**1)

    def potential(self, x, p):
        """Potential as OPE + gaussians which sets v(0) to V0"""
        out = np.sum([self.ope(x, p[f"m_{i}"], p[f"a_{i}"],  p[f"c_{i}"]) / x for i in range(self.pow_lr)], axis=0)

        if self.model == "gaussian":
            out = out + np.sum([self.gaussian(x,  p[f"gamma1_{i}"],  p[f"b_{i}"]) for i in range(self.pow_c)], axis=0)

        elif self.model == "osc":
            out += np.sum([self.osc(x, p[f"gamma1_{i}"], p[f"gamma2_{i}"], p[f"ph_{i}"], p[f"b_{i}"]) for i in range(self.pow_c)], axis=0)

        elif self.model == "harmosc":
            out +=  np.sum([self.harmosc(x, p[f"gamma1_0"], p[f"b_{i}"] ,i) for i in range(self.pow_c)], axis=0)

        elif self.model == "harmosc1D":
            out +=  np.sum([self.harmosc1D(x, p[f"gamma1_{i}"], p[f"b_{i}"] ,i) for i in range(self.pow_c)], axis=0)
            
        elif self.model == "herm_g2":
            out +=  np.sum([self.herm_g2(x, p[f"gamma1_0"], p[f"gamma2_{i}"], p[f"b_{i}"],i) for i in range(self.pow_c)], axis=0)
        return out

class SEQN:
    """Schrödinger equation integrator kernel."""

    mN0 = 1612 / HBARC / 2

    def __init__(self, e0, p, fitter, v_extra=None):
        """Intializes the kernel
        Parameters:
        e0: Energy solution of SEQN (in inverse Fermi)
        p: Parameters of the potential
        v_extra: Constant shift to potential. Can be used to increase uncertainty.                                

        """
        self.e0 = e0
        self.p = p
        self.fitter = fitter
        self.potential = self.fitter.potential
        self.k0 = (
            np.sqrt(2 * self.e0 * self.mN0)
            if e0 > 0
            else np.sqrt(2 * (self.e0 + 0j) * self.mN0)
        )
        self.v_extra = v_extra if v_extra is not None else 0

    def __call__(self, rr, psi):
        """Returns first and second derivative of wave function at rr
        psi: wave function and first derivative at rr
        """
        return np.array(
            [
                psi[1],
                2
                * self.mN0
                * (self.potential(rr,self.p) + self.v_extra - self.e0)
                * psi[0],
            ]
        )

class AsymptoticWV:
    """Kernel for asymptotic wave function (S-Wave).

    A * sin(k * r + delta)
    """

    def __init__(self, k0, rr):
        self.k0 = k0
        self.rr = rr

    def __call__(self, p):
        """
        Paramters:
            A: amplitude
            delta: phase shift
        """
        return self.wave_function((p["A"], p["delta"]))

    def wave_function(self, p_args):
        """Flat call.

        Assumes p_args[0] = p["A"] and p_args[1] = p["delta"]
        """
        return p_args[0] * np.sin(self.k0 * self.rr + p_args[1])

def _get_phase(seqn, rr, psi, r_min=4, mpi=M_PI0, prior=None):
    """Get effective range expansion for SEQN kernel, rr array and wave function."""
    x_fit = rr[rr > r_min]
    y_fit = psi[rr > r_min]
    fcn = AsymptoticWV(seqn.k0, x_fit)

    if isinstance(y_fit[0], float):
        prior = (prior["A"], prior["delta"]) if prior else (1, 1.6)

        def _fcn(p):
            return np.sum((fcn.wave_function(p) - y_fit) ** 2)

        fit = minimize(_fcn, prior)
        delta = fit.x[1]

    elif isinstance(y_fit[0], gv._gvarcore.GVar):
        prior = prior or gv.BufferDict(A="1(0.5)", delta="1.6(1.6)")
        fit = lsqfit.nonlinear_fit(data=y_fit, fcn=fcn, prior=prior)
        delta = fit.p["delta"]
    else:
        raise TypeError("Unknown type for fitting asymptotic wave.")

    return (seqn.k0 / mpi) ** 2, 1 / np.tan(delta) * seqn.k0 / mpi


def get_phase(
        fitter, e0, rr, p, tol=1.0e-8, r_min=4, mpi=M_PI0, prior_asymptotic=None, v_extra=None
):
    """Return wave function and effective range expansion.

    Paramters:
        e0: positive energy to solve Schrödinger EQN for
        rr: r-values to evalute wave function at
        p: potential fit parameters
    """
    assert e0 > 0
    assert rr.max() > r_min

    seqn = SEQN(e0=e0, p=p, fitter=fitter, v_extra=v_extra)
    integrator = gv.ode.Integrator(deriv=seqn, tol=tol)

    psi_initial = np.array([0, 1])
    psi = np.concatenate([[psi_initial[0]], integrator(psi_initial, interval=rr).T[0]])
    psi /= gv.mean(psi).max()

    q_over_mpi_sqrd, q_over_mpi_cot_delta = _get_phase(
        seqn, rr, psi, r_min=r_min, mpi=mpi, prior=prior_asymptotic
    )

    return psi, q_over_mpi_sqrd, q_over_mpi_cot_delta

def get_phase_df(fitter, fit, ranges, tol=1.0e-8, r_min=1, mpi=M_PI0, prior_asymptotic=None):

    wave_functions = []
    q_over_mpi_cot_deltas = []  # divided by mpi
    q_over_mpi_sqrds = []

    for e0 in tqdm(ranges["e0"]):
        psi, q_over_mpi_sqrd, q_over_mpi_cot_delta = get_phase(fitter, e0, ranges["r"], fit.p,tol,r_min,mpi)
        wave_functions.append(psi)

        q_over_mpi_cot_deltas.append(q_over_mpi_cot_delta)
        q_over_mpi_sqrds.append(q_over_mpi_sqrd)
        
    phase_df = pd.DataFrame(
        np.transpose([q_over_mpi_cot_deltas, q_over_mpi_sqrds]),
        columns=["k cot(δ) / mπ", "(k / mπ)^2"],
    )
    return phase_df

def get_phase_list(fitter, fit, ranges, tol=1.0e-8, r_min=1, mpi=M_PI0, prior_asymptotic=None):

    wave_functions = []
    q_over_mpi_cot_deltas = []  # divided by mpi                                                                             
    q_over_mpi_sqrds = []

    for e0 in tqdm(ranges["e0"]):
        psi, q_over_mpi_sqrd, q_over_mpi_cot_delta = get_phase(fitter, e0, ranges["r"], fit,tol,r_min,mpi)
        wave_functions.append(psi)

        q_over_mpi_cot_deltas.append(q_over_mpi_cot_delta)
        q_over_mpi_sqrds.append(q_over_mpi_sqrd)

    phase_df = pd.DataFrame(
        np.transpose([q_over_mpi_cot_deltas, q_over_mpi_sqrds]),
        columns=["k cot(δ) / mπ", "(k / mπ)^2"],
    )
    return phase_df


