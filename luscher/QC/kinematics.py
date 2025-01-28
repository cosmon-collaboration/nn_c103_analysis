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

def ere(self, x, *p):
        ''' x = qcm**2 / MN**2
            qcotd = p[0] + p[1] * x + p[2] * x**2 + ...
        '''
        qcotd = 0.
        for n in range(len(p)):
            qcotd += p[n] * x**n
        return qcotd

def deter(k2,psq,irrep,params):
    a, b, h = params 
    eps = h * k2 if psq != 'PSQ0' or psq != 'PSQ3' else 0
    d = momentum_state(int(psq[-1]))
    if psq == 0:
        return ere(k2,a,b) - 4*math.pi*clm(k2,0,0,d,L)
    elif psq == 1:
        if irrep == 'A2':
            return ere(k2,a,b) - 4*math.pi*clm(k2,0,0,d,L) + (1/math.sqrt(5)) * ((4*math.pi)/(k2)) * clm(k2,2,0,d,L) * ( math.sqrt(2)* np.sin(2*eps) - np.sin(eps)**2 )
        elif irrep == 'E':
            return ere(k2,a,b) - 4*math.pi*clm(k2,0,0,d,L) - (1/(2*math.sqrt(5))) * ((4*math.pi)/(k2)) * clm(k2,2,0,d,L) * ( math.sqrt(2)* np.sin(2*eps) - np.sin(eps)**2 )
    elif psq == 2:
        if irrep == 'B1':
            return ere(k2,a,b) - 4*math.pi*clm(k2,0,0,d,L) + (1/math.sqrt(5)) * ((4*math.pi)/(k2)) * clm(k2,2,0,d,L) * ( math.sqrt(2)* np.sin(2*eps) - np.sin(eps)**2 )
        elif irrep == 'B2':
            return ere(k2,a,b) - 4*math.pi*clm(k2,0,0,d,L) - (1/(2*math.sqrt(5))) * ((4*math.pi)/(k2)) * clm(k2,2,0,d,L) * ( math.sqrt(2)* np.sin(2*eps) - np.sin(eps)**2 )
    elif psq == 3:
        return ere(k2,a,b) - 4*math.pi*clm(k2,0,0,d,L)

           
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

def BB(self,ecm,a0,a1,b0,b1,c0,c1,eps):
    R = np.array([[np.cos(eps) , np.sin(eps)],
                    [-np.sin(eps) , np.cos(eps) ]])
    
    f0 = (1/ecm)*(a0+b0*self.delta(ecm,self.mpi_ref[self.index],self.mS_ref[self.index]))/(1 + c0*self.delta(ecm,self.mpi_ref[self.index],self.mS_ref[self.index]))
    f1 = (1/ecm)*(a1+b1*self.delta(ecm,self.mpi_ref[self.index],self.mS_ref[self.index]))/(1 + c1*self.delta(ecm,self.mpi_ref[self.index],self.mS_ref[self.index]))

    F = np.array([[f0,0],[0,f1]])

    K = R@F@np.linalg.inv(R)
    return np.linalg.inv(K)

def det_func(self,ecm,psq,key,params):
    #Kinv_f = self.k_param(ecm,key,params)
    #Kinv_f = lambda ecm: self.ERE(ecm,A00,A11,A01,B00,B11,B01)
    if isinstance(ecm, np.ndarray): # this is required for functionality with fsolve
        ecm = ecm[0]
        Kinv =  self.k_param(ecm,key,params)
        Sp = self.qcotd(ecm,psq,self.mS_ref[self.index],self.mpi_ref[self.index])
        Nk = self.qcotd(ecm,psq,self.mN_ref[self.index],self.mk_ref[self.index])
        F = np.array([[Sp , 0.0],
                        [0.0 , Nk]])
        det = np.linalg.det(Kinv - F)
        return det / (1 + np.abs(det))
    else:
        Kinv =  self.k_param(ecm,key,params)
        Sp = self.qcotd(ecm,psq,self.mS_ref[self.index],self.mpi_ref[self.index])
        Nk = self.qcotd(ecm,psq,self.mN_ref[self.index],self.mk_ref[self.index])
        F = np.array([[Sp , 0.0],
                        [0.0 , Nk]])
        det = np.linalg.det(Kinv - F)
        return det / (1 + np.abs(det))
# fernando did BB form fit


    