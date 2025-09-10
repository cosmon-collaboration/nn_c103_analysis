#
# This file contains code to perform an L=0 projection
# of periodic lattice data.   
# 1) Roll the origin (0,0,0) to a lattice point near the center.
# 2) Set up a 3D interpolation of the data.  The interpolation
#    function used does not understand periodic data so it will
#    be valid in a region inside of the edges of the volume by a few lattice points.
# 3) Perform spherical integrations against a Ylm at radii = (0, 0.5, 1.0, 1.5, ...)lu 
#    out to just short of the faces.  For this purpose we will use Y_00 
# 4) Set up 1D interpolator for the radial function.
# 5) Use the Ylm and radial function to fill in values on the lattice inside a ball
#    that does not touch the faces.
# 6) Roll the resulting lattice back to the original origin position
#
import copy
import numpy as np
import scipy
from scipy.special import genlaguerre
from scipy.special import gamma
from scipy.special import sph_harm_y
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import dblquad
from scipy.integrate import lebedev_rule

# The order tells us that this rule can integrate Y^{order}_m(\hat{r}) exactly.
lebedev_order = 9  # can remove L=4,6,8 at this order

#
# This is a reference integrand for use with dblquad
# to compare to the lebedev integration.  It is comparitively
# very slow.
#
def integrand(phi, theta, r, L, m, origin, interp):
    """
    Integrand for extracting radial component in L,m channel
    Note that interp starts at (0,0,0) in the corner of the space
    We want r=0 to be at the origin

    Warning: dblquad reverses the arguments, putting the inner integral variable first.
    Warning2: dblquad can integrate scalar floating types, not complex.  If we want
            to extract higher L pieces we may want to create
            integrand_real, and integrand_imag and integrate them separately or find
            a different integrator.
    """
    st = np.sin(theta)
    ylm = sph_harm_y(L, m, theta, phi).real
    z = r * np.cos(theta)
    rr = r * st
    x = rr * np.cos(phi)
    y = rr * np.sin(phi)
    v = np.array([x,y,z]) + origin
    d = interp(v)[0].real
    rslt = st * ylm * d  # don't include r**2 here, we would just divide it out later
    return rslt

def integrate_lebedevylm(yrule, rLmoi):
    '''
    Integrate with included ylm weight for points in lebedev rule.
    :param rule:  is a lebedev rule, we get once and pass here
    :param rLmoi:  (r, L, m, origin, interp)
    :return:     The integral - sum of weight_i f(pt_i)
    '''
    r, L, m, origin, interp = rLmoi
    pts, yweights = yrule  # pts on unit sphere
    npts = len(yweights) # number of weights
    sum = np.complex128(0.0)
    for i in range(npts):
        v = r * pts[:,i] + origin
        sum += yweights[i] * interp(v)[0]
    return [sum.real, 0.0]

def lebedevylm_rule(order, L, m):
    '''
    Make a rule with weights adjusted by Ylm.   Weights now complex
    :param order: Order of lebedev rule
    :param L:     Total angular momentum
    :param m:     z projection of angular momentum
    :return:      Rule with weights adjusted by Ylm.
    '''
    pts, weights = lebedev_rule(order)
    npts = len(weights)
    cweights = np.zeros((npts,), dtype=complex)
    for i in range(len(weights)):
        pt = pts[:, i]  # unit vector, so r==1
        theta = np.arccos(pt[2])  # z to polar angle
        phi = np.arctan2(pt[1], pt[0])
        ylm = sph_harm_y(L, m, theta, phi)
        cweights[i] = weights[i] * ylm  # combine weights and ylm to complex weight
    return [pts, cweights] # just make top level a list

def extractLm(origin, data, samplespacing, L, m, maxr, integrator = "lebedev", lebOrder=lebedev_order):
    """
    Extract table of [r, f_{L,m}(r)] from the lattice data
    :param origin: The center in lattice units (float, can be between points)
    :param data: The array of data points
    :param samplespacing: in lattice units, from 0.0 to edge
    :param L: angular momentum quantum number
    :param m: z projection of angular momentum
    :param maxr: maximum radial position to extract radial function
    :return: Table of [r, value]
    """
    if integrator == "lebedev" or integrator == "both":
        # A lebedev rule of an order L will integrate a Y_{L,m} or lower exactly
        # The number of points will be larger than the order.
        yrule = lebedevylm_rule(lebOrder, L, m)
    else:
        yrule = None
    n = data.shape[0]
    x = [float(i) for i in range(n)] # explicit list coordinate values along axis
    # build interpolator for data.   We will avoid edges.  Could add halos
    # The default method 'linear' is too choppy and the integration convergence
    # is terrible.
    interp = RegularGridInterpolator((x,x,x), data.real, method = 'quintic')
    fdata = []
    r = 0.0
    ethresh = 1e-6
    while r < maxr:
        if r == 0.0:
            # The 2.0 * sqrt(pi) cancels the Y_{0,0}, since we aren't integrating in this special case.
            # L > 0 radial functions go to 0 at origin (eff potential -> \infty)
            rslt = 0.0 if L > 0 else np.float64(interp(origin)[0].real * (2.0 * np.sqrt(np.pi)))
        else:
            rLmoi = (r, L, m, origin, interp) # extra args for integrand (r, L, m, origin, interp)
            if integrator == "lebedev":
                rslt = integrate_lebedevylm(yrule, rLmoi)
            else:
                rslt = dblquad(integrand, 0.0, np.pi, lambda theta: 0.0, lambda theta: 2*np.pi , args=rLmoi )
            if rslt[1] > max(ethresh, np.abs(rslt[0]) * ethresh): 
                raise ValueError(f"Error return from integral is too large in result = {rslt}")
            rslt = rslt[0]
        item = [r, rslt]
        fdata.append(item)
        # print(item)
        r += samplespacing # move to next radial distance
    return fdata

#
# These functions are used for testing the L=0 projection
#
def honorm(nodal, L, b):
    """
    Harmonic oscillator state normalization that is independent of r
    Separate out normalization so it can be removed from loops
    :param nodal: nodal number of HO state  n=1,2,...
    :param L: angular momentum quantum number
    :param b: length scale of HO
    :return:  overall normalization factor independent of r
    """
    return b**(-3/2) * np.sqrt(2 * gamma(nodal)/ gamma(nodal + L + 1/2))

def ho(nodal, L, b, r):
    """
    Unnormalized harmonic oscillator radial function
    :param nodal:   nodal number of HO state  n=1,2,...
    :param L: angular momentum quantum number
    :param b: length scale of HO
    :param r: radial distance
    :return:
    """
    rb = r / b
    rb2 = rb * rb
    return rb**L * np.exp(-0.5 * rb2) * genlaguerre(nodal-1, L+1/2)(rb2)

def cart2sph(v):
    """
    Convert cartesian coordinates to spherical coordinates
    :param v: is array of [x,y,z]
    :return: r,theta,phi for spherical position
    """
    r = np.linalg.norm(v)
    # should special case r==0
    phi = np.arctan2(v[1], v[0])
    theta = np.arccos(v[2] / r) if r != 0.0 else 0.0
    return r, theta, phi

#
# Generates mix of different L HO states so we can
# test the L=0 projection.
#
def maketestdata(n, origin, hodata1, hodata2, b):
    """
    Our test data will be a mix of two HO wave functions
    :param n:  number of lattice points in each direction
    :param origin:  location in grid of r=0, can be between lattice points
    :param hodata1:  HO wave function spec (weight, nodal, L, m)
    :param hodata2:  HO wave function spec (weight, nodal, L, m)
    :param b: length scale of HO  function
    :return:  lattice data with summed HO states
    """
    print("Creating test data", flush=True)
    data = np.zeros( (n,n,n), dtype=np.complex64 )
    w1, nodal1, L1, m1 = hodata1
    w2, nodal2, L2, m2 = hodata2
    norm1 = honorm(nodal1, L1, b)
    norm2 = honorm(nodal2, L2, b)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                pt = np.array([i,j,k], dtype=np.float64)
                rvec = pt - origin
                r, theta, phi = cart2sph(rvec)
                # Evaluate the two 3D harmonic oscillator states
                t1 =norm1 * ho(nodal1, L1, b, r) * sph_harm_y(L1, m1, theta, phi)
                t2 =norm2 * ho(nodal2, L2, b, r) * sph_harm_y(L2, m2, theta, phi)
                data[i,j,k] =  w1 * t1 + w2 * t2
    print("Test data created", flush=True)
    return data

#
# Note:  If we wanted to pad with periodic halos to a larger size
# np.pad(data_3d, pad_width=pad_width, mode='wrap')
# Here we simply roll the origin to a lattice point next to the center
# of the lattice.   Then we have room to do spherical integrals around the
# origin and can use RegularGridInterpolator, which does not understand
# periodic data.
#
def fix_origin(data, oldorigin):
    """
    We create a new data array of shape(sz,sz,sz) with it's
    origin at neworigin=(sz//2,sz//2,sz//2).

    :param data:   Original periodic data on cubic lattice
    :param oldorigin:
    :return: new origin near center of array , array rolled to place oldorigin at new origin
    """
    sz = data.shape[0]
    szh = sz // 2  # assume that sz is even
    neworigin = np.array([szh,szh,szh])
    return neworigin, np.roll(data, shift=neworigin-oldorigin, axis=(0, 1, 2))

def FilterL0(data, origin, spacing, lebOrder):
    """
    Given periodic data with an origin we filter to L0 content only.
    The filter is applied to data in a ball of radius Length/2 around the origin.

    This filter will not go out into the 8 corners because it won't have access to the
    entire sphere.   We could use np.pad with wrapping option.
    :param data:  3D array of periodic data
    :param origin: Location of the origin (stationary center of mass in our case)
    :param spacing: Spacing of spherical integrals, can be smaller than lattice step of 1
    :param lebOrder:  Lebedev Order for spherical integrals
    :return:
    """
    # We have to move the origin to near the center so we can use RegularGridInterpolator
    dlen = float(data.shape[0])  # shape values are even
    radius = float(data.shape[0]/2 - 1.0)  # how far out to go
    corigin = np.array(data.shape, dtype=np.int32) // 2   # near center
    cshift = corigin - origin
    # print(f"FilterL0:  Shifting by {cshift} to put origin near center")
    cdata = np.roll(data, shift=cshift, axis=(0, 1, 2))
    # origin is now at corigin (c for center)
    # extractLm(origin, data, samplespacing, L, m, maxr)
    f0data = extractLm(corigin, cdata, spacing, 0, 0, radius, "lebedev", lebOrder)  # should be real
    # print(f"f0data = {f0data}")
    # f0data has the radial function sampled at i*0.5 points
    # We convert to an interpolator so we can overwrite cdata with the
    # radial function * Y_0,0
    rpts = np.array([i * spacing for i in range(len(f0data))])
    f0dataonly = np.array([f0data[i][1] for i in range(len(f0data))])
    finterp = RegularGridInterpolator((rpts,), f0dataonly, method='quintic')
    y00 = sph_harm_y(0,0, 0.0, 0.0)
    for i in range(cdata.shape[0]):
        for j in range(cdata.shape[1]):
            for k in range(cdata.shape[2]):
                rpos = np.array([i,j,k], dtype=np.float64) - corigin
                r = np.linalg.norm(rpos) # distance from origin
                # if r is inside valid range of finterp use it
                # otherwise pick a non-zero value to avoid divide by 0.
                v = finterp([r])[0] * y00 if r <= rpts[-1] else 1000.0
                cdata[i,j,k] = v
    # roll origin back to original position
    return np.roll(cdata, origin - corigin, axis=(0, 1, 2))

def Laplacian27(data):
    """
    Compute 27 point Laplacian with improved rotational symmetry.
    Caller is responsible for dividing by spacing^2
    :param data: periodic data
    :return: Unscaled Laplacian applied to data
    """
    t = -(44.0/3.0) * data

    # add contribution from the 6 faces
    for ax in range(3):
        t += np.roll(data, 1, ax)
        t += np.roll(data, -1, ax)

    # add in contributions from 12 centers of edges (two way roll)
    for ax in [(0,1), (0,2), (1,2)]:  # pairs of axes
        for e in [(-1,-1),(-1,1),(1,-1),(1,1)]: # which of 4 edges
            t += 0.5 * np.roll(data, shift=e, axis=ax)
    # add in contributions from 8 corners
    for c in [(-1,-1,-1),(-1,-1,1),(-1,1,-1),(-1,1,1),(1,-1,-1),(1,-1,1),(1,1,-1),(1,1,1)]:
        t += (1.0/3.0) * np.roll(data, shift=c, axis=(0,1,2))
    return (3.0/13.0) * t

#
# Test code for polynomial fitting
#
def makeCentralPolyOp(hn, order):
    '''
    Fit a polynomial to evenly space samples around 0.
    :param hn:   number of samples after and before 0.  hn=2 -> [-2, -1, 0, 1, 2]
    :param order: max power of polynomial to be fit.
    :param spacing:  space between samples
    :return:  Matrix MM[order+1:2hn+1] such that MM.data[-hn:hn] gives polynomial coefficients [a_0, a_1, ..., a_{order}]
    '''
    samples = np.array(range(-hn, hn+1), dtype=np.float64)
    M = np.array( [  [np.power(samples[i], p) for p in range(order+1)] for i in range(len(samples))], dtype=np.float64)
    # print(f"M = {M}")
    MM = scipy.linalg.inv(M.T.dot(M)).dot(M.T)
    MM[np.abs(MM) < 1e-14] = 0.0
    return MM

def testMakeCentralPolyOp():
    print("Testing makeCentralPolyOp")
    MM = makeCentralPolyOp(2, 4)
    print(f"MM = {24.0*MM}")

def testLaplacian27():
    print("Testing Laplacian27", flush=True)
    n = 48
    data = np.zeros( (n,n,n), dtype=np.float64 )
    ldata = np.zeros((n,n,n), dtype=np.float64)
    # fill in some periodic data
    w = 2 * np.pi / n
    print(f"w = {w}")
    for i in range(n):
        for j in range(n):
            for k in range(n):
                v1 = np.sin(w * i) * np.cos(2 * w * k)  # orthogonal to \hat{y}
                v2 = np.sin(w * i) * np.sin(w * j) * np.sin(w * k)  # diagonal wave
                data[i,j,k] = v1 + v2
                # compute Laplacian analytically
                ldata[i,j,k] = -w*w * (5.0 * v1 + 3.0 * v2)

    lap = Laplacian27(data)
    print(f"Laplacian stencil result at [{n-1},{n-1},:]")
    print(lap[-1,-1,:])
    print(f"Analytic Laplacian result at [{n-1},{n-1},:]")
    print(ldata[-1,-1,:])
    print("diff")
    print((lap -ldata)[-1,-1,:] )

def testFilterL0():
    """
    Generate mix of L=0 and L=4 states in lattice with origin at (0,0,0)
    Run FilterL0 and check that the resulting radial function matches the
    L0 radial function used to create the data.
    :return: None
    """
    n = 48
    nh = (n // 2)
    origin = np.array([0,0,0], dtype=np.int32) # location in lattice of origin
    w1, nodal1, L1, m1 = (3.0, 3, 0, 0)  # the one we want to extract
    w2, nodal2, L2, m2 = (4.0, 2, 4, 1)  # higher L noise we are adding in
    b = 5.0  # ho length scale for test
    norm1 = honorm(nodal1, L1, b)
    # print(f"norm1 = {norm1}")
    sphL1 = sph_harm_y(L1, m1, 0.0, 0.0)
    # print(f"sphL1 = {sphL1}")
    norm2 = honorm(nodal2, L2, b)
    data = np.zeros( (n,n,n), dtype=np.complex64 )
    for i in range(-nh+1,nh):   # from -nh+1 to nh-1
        for j in range(-nh+1,nh):
            for k in range(-nh+1,nh):
                pt = np.array([i, j, k], dtype=np.float64)
                r, theta, phi = cart2sph(pt)
                if r >= nh:
                    continue # leave out sites at or beyond the face of a cube centered about origin
                # relying on negative indices to wrap
                v = w1 * norm1 * ho(nodal1, L1, b, r) * sphL1  # when L1==0, sph_harm_y(L1, m1, theta, phi) will be constant
                v += w2 * norm2 * ho(nodal2, L2, b, r) * sph_harm_y(L2, m2, theta, phi)
                data[i,j,k] = v  # relying on negative indices wrapping
    print("Filtering to L=0", flush=True)
    # print(f"data[0,0,0] = {data[0,0,0]}", flush=True)
    fdata = FilterL0(data, origin, 0.5)
    # print(f"fdata[0,0,0] = {fdata[0, 0, 0]}", flush=True)
    # Now extract the radial function and compare to the original
    fdatar = np.roll(fdata, shift=(nh,nh,nh), axis=(0,1,2)) # roll to near center
    nhpt = np.array([nh,nh,nh], np.float64)
    print("Extracting L0 radial function from filtered data", flush=True)
    radf = extractLm(nhpt, fdatar, 0.5, L1, m1, nh - 1.0)
    # print("radf = ", radf, flush=True)
    print("Comparing mix->filter->extract to original L0 radial function")
    for i in range(len(radf)):
        r = radf[i][0]
        h = w1 * norm1 * ho(nodal1, L1, b, r)
        print(f"   {radf[i][0]}: {radf[i][1]}, {h}")

#
# Development version of L=0 projection after A1 projection
# Since moved directly into ipynb
#
def process_A1():
    '''
    The data we are processing comes from a file and has the
    shape  (nconfigs, ntimes, nx, ny, nz)
    This data is from the ratio correlator    R(t, r) = C_{NN}(t, r)/C_N(t)^2
    The origin is (0,0,0).
    The task is to extract the radial function for each cfg and time
    :return:
    '''
    fn = "data/nn_data_A1.dat.npy"
    da1 = np.load(fn)
    print(f"Loaded data has shape (ncfg,nt, nx,ny,nz)= {da1.shape}", flush=True)
    # how far out do we extract the radial function - not into the corners.
    radius = da1.shape[2] * 0.45
    origin = np.array([0,0,0], dtype=np.float64)
    norigin = np.array(da1.shape[2:5], dtype=np.int32) // 2
    for cfg in range(4,5):  # range(da1.shape[0]):
        for t in range(2, da1.shape[1]-2):
            data = da1[cfg,t,:,:]
            rdata = np.roll(data, shift=norigin, axis=(0,1,2))
            f0data = extractLm(norigin, rdata, 0.5, 0, 0, radius)
            print(f"t={t}, f0={f0data}")

def doL0Filter():
    nn_data_A1 = np.load("data/nn_data_A1.npy")
    nn_data_L0 = np.zeros_like(nn_data_A1)
    for cfg in range(nn_data_A1.shape[0]): # range(nn_data_A1.shape[0]):
        for t in range(nn_data_A1.shape[1]):
            # print(f"cfg={cfg}, t={t}: ", end="")
            nn_data_L0[cfg,t] = FilterL0(nn_data_A1[cfg,t], origin=np.array([0,0,0]), spacing=0.5, lebOrder=9)
        print(".", flush=True, end="")
    print("")
    np.save("data/nn_data_L0_Leb9_s5.npy", nn_data_L0)


if __name__ == "__main__":
    # Configure to run tests for development
    if False:
        data = np.zeros( (4,4,4), dtype=np.int32)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    data[i,j,k] = i*100+j*10+k
        print(data)
        ndata = np.roll(data, shift=(1,1,1), axis=(0,1,2))
        print(ndata)
        quit()
    doL0Filter()
    # process_A1()
    # testMakeCentralPolyOp()
    # testLaplacian27()
    # testFilterL0()
    quit(0)

