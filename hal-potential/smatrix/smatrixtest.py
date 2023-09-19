import math
import cmath
import numpy as np
import matplotlib.pyplot as plt
import smatrix

#
# Example of two phase equivalent (in L=0) potentials
# Eckart potentials from Bargman PhysRev.75.301.pdf (1948
# They are good test cases here because they have identical analytic phase shifts.
# The are already posed in the  
#  \partial_x^2 \psi(x) + k^2 \psi(x) = V(x) \psi(x)
# so V needs no multiplication by $2 \mu / \hbar^2$
#
lam=1.0  # bound state at -(1/4) lam^2
def Veck(r, sigma, beta, xlambda):
    e = math.exp(-xlambda*r)
    bex = beta*e + 1.0
    be2 = bex*bex
    return -sigma*xlambda*xlambda*beta*e/be2
def mu(beta, xlambda):
    return xlambda * (beta - 1.0)/(beta + 1.0)
def Veckf1(k, beta, xlambda):
    return (2 * k + 1.0j * xlambda)/(2 * k - 1.0j * 2 * xlambda)
def V2(r, data):
    return Veck(r, 2.0, 3.0, 2.0*lam)
def V1(r, data):
    return Veck(r, 6.0, 1.0, lam)
# Analytic computatation of S matrix for V1 and V2
def v1S(k):
    return Veckf1(k, 1.0, lam) / Veckf1(-k, 1, lam)

def plotV1V2():
    color1 = (0.5, 0.4, 0.3)
    color2 = (0.2, 0.2, 1.0)
    ra = np.arange(0.0, 8.0, 0.1)
    # print(ra)
    V1a = [V1(r, 0) for r in ra]
    V2a = [V2(r, 0) for r in ra]
    plt.xlabel('r')
    plt.grid(True)
    #plt.legend(loc='best')
    plt.plot(ra, V1a)
    plt.plot(ra, V2a)
    plt.show()


# v = Veck(1.0, 2.0, 3.0, 2.0)
# print(f"v={v}")

plotV1V2()

def dotest():
    print("Testing analytic S-matrix for special potentials V1 and V2")
    print("against the numerical calculation")
    print("sa is the analytic result, sv is the variable phase method result")
    for testk in [0.1, 0.5, 1.0, 3.0]:
        sv = smatrix.get_smatrix_single(0, testk, V2, None, 10.0)
        sa = v1S(testk)
        print(f"V2: k={testk}, sa={sa}, sv={sv}")

        # V1 dies off slower, give it bigger vR
        sv = smatrix.get_smatrix_single(0, testk, V1, None, 20.0)
        # sa is the same for V1, they are phase equivalent
        print(f"V1: k={testk}, sa={sa}, sv={sv}")

dotest()

