import math
import cmath
#import jax
#import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt 
import scipy
#import seaborn as sns
import time
import itertools

from scipy import optimize    
from scipy.optimize import root_scalar
from scipy.optimize import brentq        
from scipy import integrate 
from scipy.optimize import fsolve
from scipy.optimize import minimize
#from sympy import *
import h5py as h5
from QC.zeta import Z
import QC.h5_reader as h5r

import nn_parameters as params
#from nn_fit import Fit

# obj = Fit()
p = params.params()
states = p["masterkey"]

h5_file_path = "./data/nn_deuteron_spectrum.h5"

h5r.print_h5_keys(h5_file_path)
nbs = 0

L = 48 
level_dict = { 'PSQ0': {'T1g' : [0,2]},
              'PSQ1': {'A2' : [0,1],'E' : [0,1]},
              'PSQ2': {'B1' : [0,1],'B2' : [0,1]},
              'PSQ3': {'A2' : [0,2],'E' : [0,1]}}


def load_data():
    fit_results = h5.File(h5_file_path,'r')
    print("fit_results",fit_results)
    # load the nucleon mass
    mN = fit_results["mN"]
    
    # load all the NN and N data
    data = {}
    data['mN']  = mN
    print(states)
    for state in states:
        print("state",state)
        data[str(state)] = {}
        for k in fit_results:
            if k != 'mN':
                print("k",k)
                irrep, psq, level = k.split('_')
                Psq = int(psq[-1])
                #Psq, irrep, n = k[0][0] #state form
                data[str(state)]['dE_NN'] = fit_results[k]['dE_NN']
                data[str(state)]['Psq']   = int(Psq)
                data[str(state)]['E_N1']  = fit_results[k]['E_N1']
                data[str(state)]['psq_1'] = fit_results[k]['Psq_N1']
                data[str(state)]['E_N2']  = fit_results[k]['E_N2']
                data[str(state)]['psq_2'] = fit_results[k]['Psq_N2']
    return data
dat = load_data()

def make_qcotd(data):

    # create qcotd
    qcotd     = {}
    qcotd_raw = {}
    #Nbs = data['mN'].shape[0] - 1 #remove boot0 to determine Nbs
    Nbs = 0
    print("nbs",Nbs)
    for state in states:
        qcotd[state] = {}
        qcotd_raw[state] = {}
        if p['continuum_disp']:
            EN1 = np.sqrt(data['mN']**2 + data[state]['psq_1']*(2*np.pi / self.L)**2)
            EN2 = np.sqrt(data['mN']**2 + data[state]['psq_2']*(2*np.pi / self.L)**2)
        else:
            EN1 = data[state]['E_N1']
            EN2 = data[state]['E_N2']
        dENN = data[state]['dE_NN']

        # populate individual irrep data
        E_NN   = dENN + EN1 + EN2
        E_cmSq = E_NN**2 - data[state]['Psq'] * (2 * np.pi / self.L)**2
        E_cm   = np.sqrt(E_cmSq)
        qsq    = E_cmSq / 4 - data['mN']**2

        qcotd_raw[state]['EcmSq'] = E_cmSq
        qcotd_raw[state]['qsq']   = qsq

        Psq   = data[state]['Psq']
        irrep = state[1]
        # boxQ  = BMat.BoxQuantization(self.momRay[Psq], Psq, 
        #                             irrep, chanList, 
        #                             [0,], self.Kinv, True)
        # boxQ.setMassesOverRef(0, 1, 1)
        #self.qcotd_raw[state]['qcotd_lab'] = np.zeros(self.Nbs +1)
        qcotd_raw[state]['qcotd_cm']  = np.zeros(self.Nbs +1)
        for bs in range(Nbs +1):
            ma  = data['mN'][bs]/data['mN'][bs]
            mb  = data['mN'][bs]/data['mN'][bs]
            mref = data['mN'][bs]
            #boxQ.setRefMassL(data['mN'][bs]*self..L)
            #self.qcotd_raw[state]['qcotd_lab'][bs] = boxQ.getBoxMatrixFromElab(E_NN[bs] / data['mN'][bs]).real
            qcotd_raw[state]['qcotd_cm'][bs]  = qc_kin.qcotd( E_cm[bs]/data['mN'][bs], L,Psq,1,1,mref)
            #boxQ.getBoxMatrixFromEcm(E_cm[bs] / data['mN'][bs]).real

    

        self.qcotd[irrep_grp]['EcmSq'] = E_cmSq
        self.qcotd[irrep_grp]['qsq']   = qsq

        Psq   = int(irrep_grp[0])
        # we can use either irrep in the irrep_grp to define the BoxMatirx
        irrep = self.irrep_grps[irrep_grp][0][1][1]
        # boxQ  = BMat.BoxQuantization(self.momRay[Psq], Psq, 
        #                             irrep, chanList, 
        #                             [0,], self.Kinv, True)
        #boxQ.setMassesOverRef(0, 1, 1)
        #self.qcotd[irrep_grp]['qcotd_lab'] = np.zeros(self.Nbs+1)
        self.qcotd[irrep_grp]['qcotd_cm']  = np.zeros(self.Nbs+1)
        for bs in range(self.Nbs +1):
            ma  = data['mN'][bs]/data['mN'][bs]
            mb  = data['mN'][bs]/data['mN'][bs]
            mref = data['mN'][bs]
            #boxQ.setRefMassL(data['mN'][bs]*self.args.L)
            #self.qcotd[irrep_grp]['qcotd_lab'][bs] = boxQ.getBoxMatrixFromElab(E_NN[bs] / data['mN'][bs]).real
            self.qcotd[irrep_grp]['qcotd_cm'][bs]  = qc_kin.qcotd( E_cm[bs]/data['mN'][bs], self.L,Psq,1,1,mref)
            #self.qcotd[irrep_grp]['boxQ'] = lambda x: qc_kin.qcotd( x, self.L,Psq,1,1,data['mN'][0])
            #boxQ.getBoxMatrixFromEcm(E_cm[bs] / data['mN'][bs]).real
        # set boxQ to boot0 and save
        #boxQ.setRefMassL(data['mN'][0]*self.args.L)
        #self.qcotd[irrep_grp]['boxQ'] = qc_kin.qcotd( E_cm[0]/data['mN'][0], self.L,Psq,1,1,data['mN'][0])


# level_dict = { 'PSQ0': {'T1g' : [0,1,2,3,4]},
#               'PSQ1': {'A2' : [0,1,2,3,4],'E' : [0,1,2,3,4]},
#               'PSQ2': {'B1' : [0,1,2,3,4],'B2' : [0,1,2,3,4]},
#               'PSQ3': {'A2' : [0,1,2,3,4],'E' : [0,1,2,3,4]}}

#print(ecm_data)
# first, build E_cm / mN
# def ecm_data_mN(ecm_data, mN_data):
#     ecm_mN = {}
#     for psq in ecm_data.keys():
#         ecm_mN[psq] = {}
#         for irrep in ecm_data[psq].keys():
#             ecm_mN[psq][irrep] = {}
#             for level in ecm_data[psq][irrep].keys():
#                 ecm_mN[psq][irrep][level] = ecm_data[psq][irrep][level] / mN_data
#     return ecm_mN

def covariance_matrix_constructor(data_dict):
    data_list = []
    for psq in level_dict.keys():
        for irrep in level_dict[psq].keys():
            for ecm in level_dict[psq][irrep]:
                data_list.append(data_dict[psq][irrep][f'ecm_{ecm}'][1:])
    data_vector = np.array(data_list)
    covariance = np.cov(data_vector)
    return covariance
    
# ecm_mN_data = ecm_data_mN(ecm_data, mN_data)

def plot_energies_vs_irreps(data):
     # Marker styles for ecm_keys
    markers = {
        'ecm_0': 'o', 'ecm_1': 's', 'ecm_2': 'D', 'ecm_3': '^', 'ecm_4': 'v'
    }

    # Colors for irreps
    irrep_colors = itertools.cycle(plt.cm.tab10.colors)

    # Map for irreps to colors
    ecm_color_map = {}

    # Initialize x-axis positions and energy values
    x_mapping = {}
    x_labels = []
    x_pos = 0

    # Spacing parameters
    psq_gap = 1  # Gap between PSQs
    irrep_gap = 0.25  # Gap between irreps within the same PSQ

    plt.figure(figsize=(10, 6.134))
    # mpi = 0.310810
    # t_channel_cut = mpi**2 / (4*mN_data[0]) 
    # t_channel_cut = t_channel_cut/(mN_data[0])
    # print(t_channel_cut)
    # Loop over PSQ groups
    for psq_index, (psq, irreps_dict) in enumerate(data.items()):
        psq_base = psq_index * psq_gap  # Base position for this PSQ

        for irrep, levels in irreps_dict.items():
            # Assign colors to irrep

            # Assign x-position for the irrep within the current PSQ
            irrep_label = f"{irrep} ({psq[-1]})"
            if irrep_label not in x_mapping:
                x_mapping[irrep_label] = psq_base + len(x_mapping) * irrep_gap
                x_labels.append(irrep_label)

            x_pos = x_mapping[irrep_label]

            for ecm_key, values in levels.items():
                if ecm_key not in ecm_color_map:
                    ecm_color_map[ecm_key] = next(irrep_colors)
                color = ecm_color_map[ecm_key]
                marker = markers.get(ecm_key, 'x')  # Default marker for unknown keys
                # Calculate central value and error
                central_value = values[0]
                error = np.std(values[1:]) if len(values) > 1 else 0.0
                #print(error)

                plt.errorbar(
                    x_pos,
                    central_value,
                    yerr=error,
                    fmt=marker,
                    color=color,
                    label=f"{ecm_key}" if psq_index == 0 and irrep == list(irreps_dict.keys())[0] else None,
                    alpha=0.7,
                    capsize=3  # Size of the error bar caps
                )

    # Customize the plot
    plt.xticks(
        ticks=[x_mapping[label] for label in x_labels],
        labels=x_labels,
        rotation=45
    )
    plt.xlabel("")
    plt.ylabel("$E_{\mathrm{cm}} / m_N$")
    plt.title("")
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1) )#, title="Markers by ecm_key")
    plt.grid(True, linestyle='--', alpha=0.5)
    #plt.axhline(y=t_channel_cut, color='green', linestyle='--', label='t-channel cut')
    plt.tight_layout()  # Adjust layout for better readability
    plt.savefig("Images/Plots/Energy_vs_irreps.pdf")


def momentum_state(i):
        if i == 0:
            return np.array([i,i,i])
        elif i == 1:
            return np.array([0,0,i])
        elif i == 2:
            return np.array([i-1,i-1,0])
        elif i == 3:
            return np.array([i/3,i/3,i/3])
        elif i == 4:
            return np.array([0,0,i/2])
        else: 
            # Raise an exception for invalid input
            raise ValueError("Invalid value for 'i'. 'i' must be 0, 1, 2, 3, or 4.")

def q2(ecm,ma,mb):
        #ecm is the energy data
        #mpi, mS = self.data.sigma_pi_masses()
        q2 = ecm**2 / 4 - (ma**2 + mb**2) / 2 + ((ma**2 - mb**2)**2) / (4*ecm**2)
        return q2

def clm(q2,l,m,d,L):
    #k2 = q2(ecm,ma,mb)
    msplit = 1 #if ma == mb else 1 + (ma**2 - mb**2 )/ecm**2
    pre = math.sqrt(4*math.pi)/(L**3)
    pre *= pow((2*math.pi)/L , l-2)
    return pre*Z(q2 * (L/(2*math.pi))**2, gamma=1,l=l,m=m,d=d,m_split=msplit)


def kcm2_data(ecm_data, mN_data):
    kcm2 = {}
    for psq in ecm_data.keys():
        kcm2[psq] = {}
        for irrep in ecm_data[psq].keys():
            kcm2[psq][irrep] = {}
            for level in ecm_data[psq][irrep].keys():
                kcm2[psq][irrep][level] = q2(ecm_data[psq][irrep][level], mN_data, mN_data)
    return kcm2


#print(k2_data)

#print(ecm_mN_data['PSQ0']['T1g']['ecm_0'][0])
#print(k2(ecm_data ['PSQ0']['T1g']['ecm_0'][0], mN_data[0])/(mN_data[0]**2))
#print(q2(ecm_data ['PSQ0']['T1g']['ecm_0'][0], mN_data[0], mN_data[0])/(mN_data[0]**2))

def ere(q2,a,b):
    # a,b = params
    return -(1/a) + 0.5*b*q2

def deter(k2,psq,irrep,params):
    a, b, h = params 
    eps = h * k2 if psq != 'PSQ0' or psq != 'PSQ3' else 0
    d = momentum_state(int(psq[-1]))
    if psq == 'PSQ0':
        return ere(k2,a,b) - 4*math.pi*clm(k2,0,0,d,L)
    elif psq == 'PSQ1':
        if irrep == 'A2':
            return ere(k2,a,b) - 4*math.pi*clm(k2,0,0,d,L) + (1/math.sqrt(5)) * ((4*math.pi)/(k2)) * clm(k2,2,0,d,L) * ( math.sqrt(2)* np.sin(2*eps) - np.sin(eps)**2 )
        elif irrep == 'E':
            return ere(k2,a,b) - 4*math.pi*clm(k2,0,0,d,L) - (1/(2*math.sqrt(5))) * ((4*math.pi)/(k2)) * clm(k2,2,0,d,L) * ( math.sqrt(2)* np.sin(2*eps) - np.sin(eps)**2 )
    elif psq == 'PSQ2':
        if irrep == 'B1':
            return ere(k2,a,b) - 4*math.pi*clm(k2,0,0,d,L) + (1/math.sqrt(5)) * ((4*math.pi)/(k2)) * clm(k2,2,0,d,L) * ( math.sqrt(2)* np.sin(2*eps) - np.sin(eps)**2 )
        elif irrep == 'B2':
            return ere(k2,a,b) - 4*math.pi*clm(k2,0,0,d,L) - (1/(2*math.sqrt(5))) * ((4*math.pi)/(k2)) * clm(k2,2,0,d,L) * ( math.sqrt(2)* np.sin(2*eps) - np.sin(eps)**2 )
    elif psq == 'PSQ3':
        return ere(k2,a,b) - 4*math.pi*clm(k2,0,0,d,L)



# Example usage
# a = covariance_matrix_constructor(ecm_mN_data)
# print(np.cov(a))

            
def QC1(psq,irrep,q2_range, params): # works, tale ~ 10 s for PSQ =0 , ~ 20 s for PSQ=1, why different? 
    roots = []
    tol = 1e-4
    root_counter = 0
    func = lambda k2: np.real(deter(k2,psq,irrep,params))
    for z in range(len(q2_range)-1):
        #print(q2_range[z])
        z1 = func(q2_range[z])
        z2 = func(q2_range[z+1])
        #print(z1,z2)
        #print(((q2_range[z] + q2_range[z+1])/2))
        if np.sign(z1) != np.sign(z2) and np.abs(z1-z2) < 1:
            #print("sign change",q2_range[z])
            if root_counter in level_dict[psq][irrep]:
                #print(root_counter)
                res = fsolve(func, q2_range[z] - .001)[0]
                # Check if the result is not close to any existing root
                is_new_root = all(not np.isclose(res, root, atol=tol) for root in roots)
                if is_new_root:
                    roots.append(res)
                #roots.append(res)
                root_counter += 1
            else:
                root_counter += 1
            # elif all(not np.isclose(res, root, atol=tol) for root in roots):
            #     roots.append(res)
                #roots.append(res)
            #elif all(abs(res - root) > tol for root in roots):   
        if len(roots) == len(level_dict[psq][irrep]):
            break
    return np.array(sorted(roots))

#print(QC1('PSQ0','T1g',np.linspace(-0.01,0.05,200),[-22,0.4,.001]))

def plot_func(psq, irrep, params, q2_range=(-0.01,0.05), num_points=200):
    """
    Plot the behavior of func(q2) over a specified range.

    Args:
        psq (str): The psq parameter for the deter function.
        irrep (str): The irrep parameter for the deter function.
        params (list): Parameters for the deter function.
        q2_range (tuple): Range of q2 values to plot (min, max).
        num_points (int): Number of points to evaluate in the range.

    Returns:
        None
    """
    #func = lambda k2: deter(k2, psq, irrep, params)

    # Generate q2 values and evaluate func(q2)
    q2_values = np.linspace(q2_range[0], q2_range[1], num_points)
    func_values = [np.real(deter(k,psq,irrep,params)) for k in q2_values]
    q2_where_0 = QC1(psq,irrep,q2_values ,params)
    #print(func_values)
    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.plot(q2_values, func_values,color='blue')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('q2')
    plt.ylabel('determinant')
    plt.title(f'Behavior of deter for psq={psq}, irrep={irrep}')
    # Adjust y-axis limits
    #y_min, y_max = min(func_values), max(func_values)
    plt.ylim(-.1,.1)
    #plt.xlim(-.01,0.05)
    for root in q2_where_0:
        plt.plot(root,0, "*", color='green')
    #plt.plot(q2_where_0 , 0,"*",color='red')
     # Display the parameters on the plot
    param_text = f"a={params[0]}, b={params[1]}, h={params[2]}"
    plt.text(0.05, 0.95, param_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", alpha=0.5))

    #plt.legend()
    plt.grid()
    plt.savefig(f"Images/Plots/Deter_{psq}_{irrep}_{params}.pdf")

#plot_func('PSQ0','T1g',[-22,0.4,.001],q2_range=(-0.01,0.05), num_points=200)
#print(QC1('PSQ2','B1',[-22,0.4,.001]))
def plot_all_determinant_conditions( params, q2_range=(-0.01, 0.05), num_points=300 ):
    """
    Plot the behavior of deter(q2) for all psq and irrep combinations.

    Args:
        params (list): Parameters for the deter function.
        q2_range (tuple): Range of q2 values to plot (min, max).
        num_points (int): Number of points to evaluate in the range.

    Returns:
        None
    """
    for psq in level_dict.keys():
        for irrep in level_dict[psq].keys():
            plot_func(psq, irrep, params, q2_range=q2_range, num_points=num_points)
#plot_all_determinant_conditions([-22,0.4,.001],q2_range=(-0.01,0.1), num_points=300)

def chi2(params,data_dict,index,q2_range):  # WT results: 
    # i = 0 has to be g1u, i=1 g1,i=2 g2,i=3 g3
    # cov = np.cov(data)
    #A00,A11,A01,B01 = x
    res = np.empty(0)
    energy_data = data_dict#
    #energy = energy[:,self.index] #bootstrap, 0 is mean value
    for psq in level_dict.keys():
        for irrep in level_dict[psq].keys():
            sols = QC1(psq,irrep,q2_range, params)
            lvl_count = 0
            for ecm in level_dict[psq][irrep]:
                #print(data_dict[psq][irrep][f'ecm_{ecm}'][index])
                #ecm_mN_sol = 2*np.sqrt( sol + mN_data[index]**2)
                res = np.append(res, data_dict[psq][irrep][f'ecm_{ecm}'][index] - sols[lvl_count])
                #print(res)
                lvl_count += 1

    cov = covariance_matrix_constructor(data_dict)
    value = res@np.linalg.inv(cov)@res
    return value  

def minimize_chi2(data_dict,index,q2_range):
    # Initial guess for the parameters
    x0 = [-10, 0.1, 0.001]
    # Define the objective function
    obj_func = lambda x: chi2(x, data_dict, index,q2_range)
    # Minimize the objective function
    result = minimize(obj_func, x0, method='Nelder-Mead')
    return result

#print(chi2([-22,0.4,.001],k2_data,0,np.linspace(-0.01,0.1,200)))  
print(minimize_chi2(k2_data,0,np.linspace(-0.01,0.1,200)))



