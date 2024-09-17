import numpy as np

import math
from QC.zeta import Z

#~~~~~~~~~~~~~~~~~~~~~~~`
# energy-dependent kinematic functions
# required for analysis
# ~~~~~
#  convert PSQ from hdf5 to momentum vector
def momentum_state(psq):
        # d = [0,0,0]
        if psq == 0:
            return np.array([0,0,0])
        # d = [0,0,1]
        elif psq == 1:
            return np.array([0,0,1])
        # d = [1,1,0]
        elif psq == 2:
            return np.array([1,1,0])
        # d = [1,1,1]
        elif psq == 3:
            return np.array([1,1,1])
        # d = [0,0,2]
        elif psq == 4:
            return np.array([0,0,2])
        else: 
            # Raise an exception for invalid input
            raise ValueError("Invalid value for 'i'. 'i' must be 0, 1, 2, or 3.")
# q^2 is COM momentum

def kallen(x,y,z):
    return None

def q2(ecm,ma,mb): #assume all ref inputs
    #ecm is the energy data
    # assumed to be (e_cm/m_ref)
    #ma,mb comes from the energies in the channel
    q2 = ecm**2 / 4 - (ma**2 + mb**2) / 2 + ((ma**2 - mb**2)**2) / (4*ecm**2)
    return q2

# msplit required for Luscher: 
# measures the shift of the rvector of unequal relativistic masses
def msplit(ecm,ma,mb): #if ma,mb are degenerate its 1
    if ma == mb:
        return 1 
    else:
        return 1 + ((ma**2 - mb**2)/ecm**2 ) 


# ~~~~~~~~~~~~~~~~~~~~~~~`
# lorentz factor
# gamma = E/E^*`
# ref is reference mass from data set
# d is PSQ of the state
def gamma(ecm,d,L,ref):
    d_vec = momentum_state(d) 
    L_ref = L*ref
    E = math.sqrt(ecm**2 + (((2*math.pi)/L_ref)**2)*np.dot(d_vec,d_vec))
    #print("E=",E)
    return E/ ecm #np.abs(ecm)

# Leuscher Zeta Function
# using numerical approach including numerical integrals
def qcotd(ecm,L,psq,ma,mb,ref):
        L_ref = L*ref
        d_vec = momentum_state(psq) #0,1,2,3
        c = 2 / (gamma(ecm,psq,L,ref)*L_ref*math.sqrt(math.pi))
        #print("c=",c)
        # print('ecm=', ecm)
        # print("gamma = ",self.gamma(ecm,psq,ref))
        #print( psq )
        #print( ma )
        return c*Z(q2(ecm,ma,mb)*((L_ref/(2*math.pi))**2),gamma=gamma(ecm,psq,L,ref),l=0,m=0,d=d_vec,m_split=msplit(ecm,ma,mb),precision=1e-11).real

def clm(ecm,L,psq,l,m,ma,mb,ref): #kinematic function related to Lüscher zeta function
    # for non-relativistic NN 
    L_ref = L*ref
    d_vec = momentum_state(psq)
    prefactor = math.sqrt(4 * math.pi)/ L_ref
    prefactor *= pow((2*math.pi/L_ref),int(l-2))
    return prefactor * Z( q2(ecm,ma,mb)*((L_ref/(2*math.pi))**2), gamma = 1 , l=l, m=m, d = d_vec,m_split=msplit(ecm,ma,mb),precision=1e-11)


    