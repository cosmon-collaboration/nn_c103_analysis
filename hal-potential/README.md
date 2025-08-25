# Introduction  

We describe the usage of the code to fit the HAL Potential results we generated on the C103 ensemble.

# The Hal Potential Definition
Step one is to produce the correlation function
```math
R\left(t, r\right) = \frac{ C_{NN}\left(t, r\right)} { N(t)^2 }
```
for all t and r displacments.   The HalQCD Potential is then defined as
```math
V\left(r\right) = \frac{\nabla^2 R}{M_N R} - \frac{\partial_t R}{R} + \frac{\partial_t^2 R}{4 M_N R}
```

We are trying 3 numerical paths all beginning with the correlator $R$ calculated on the lattice.

1) We perform an $A_1$ projection and assume that $L=4$ and higher partial waves have small contributions.  The expression above is evaluated with numerical Laplacian and time-slice to time-slice deriviatives.  Some exploration of higher orders was attempted.  (Need to check for symmetric time derivative so that we aren't using a different time point for the derivative and the Laplacian).
2) We add an L=0 projection by extration of the L=0 radial function followed by repopulating the Lattice.  Then we repeat the process in (1).
3) We directly implement $V(r)$ on the radial function and associated $Y^L_m$.  This will have a much better Laplacian and will average the time derivative taken at different radial directions.

# Input Data  
Input data is hosted at  [cosmon nn_c103_205.05547](https://portal.nersc.gov/cfs/m2986/cosmon/nn_c103_2505.05547)  

Data can be downloaded with 
```sh
cd <your data directory>
wget -nd -r -P . -A "c103_n*.h5" https://portal.nersc.gov/cfs/m2986/cosmon/nn_c103_2505.05547/
```

The data files are large.   Verify that you have room for more than 22Gb.  
