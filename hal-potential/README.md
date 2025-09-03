# Introduction  

We describe the usage of the code to fit the HAL Potential results we generated on the C103 ensemble.

# The Hal Potential Definition
Step one is to produce the correlation function ratio
```math
R\left(t, r\right) = \frac{ C_{NN}\left(t, r\right)} { N(t)^2 }
```
for all t and r displacments.   The HalQCD Potential is then defined as
```math
V\left(r\right) = \frac{\nabla^2 R}{M_N R} - \frac{\partial_t R}{R} + \frac{\partial_t^2 R}{4 M_N R}
```

We are trying 2 numerical paths all beginning with the correlator $R$ calculated on the lattice.

1) We perform an $A_1$ projection and assume that $L=4$ and higher partial waves have small contributions.  The expression above is evaluated with numerical Laplacian and numerical time deriviatives.   
2) We add an L=0 projection by extration of the L=0 radial function followed by repopulating the Lattice.  Then we repeat the process in (1).

## Improved Time Derivatives  

For the defined potential above we require both a first and second derivative.   We've decided to improve the second derivative with additional symmetric points, so we chose five points at $\delta_t = \left[-2, -1, 0, 1, 2\right]$.   We fit a quartic polynomial to the correlator ratio at the 5 points, and then infer both first and second derivatives from the same polynomial for consistency.  Let 
```math
\begin{aligned}
&\mathbf{M} = \left( \begin{array}{rrrrr}
 1 & {-2} & 4 & -8 & 16 \\
1 & {-1} &  1 & -1 & 1 \\
1 & 0 & 0 & 0 & 0 \\
 1 & \space\space 1 & \space\space 1 & \space\space 1 & \space\space 1 \\
1 & 2 & 4 & 8 & 16
\end{array} \right)  \\
&\mathbf{M}^{-1} = \frac{1}{24} \left( \begin{array}{rrrrr}
0 & 0 & 24 & 0 & 0 \\
2 & -16 & 0 & 16 & -2 \\
-1 & 16 & -30 & 16 & -1 \\
-2 &  4 & 0 & -4 & 2 \\
1 & -4 & 6 & -4 & 1
\end{array}\right)
\end{aligned}
```
where each row contains powers of one of the $\delta_t$ selections.   Then we can apply these rows to a vector of the polynomial coefficients, $\mathbf{a}$, to yield function values   
```math
\mathbf{v}(t,r) = \left[R(t-2,r),R(t-1,r),R(t,r),R(t+1,r),R(t+2,r)\right]
```   
at each $\delta_t$ offset from $t$ the corresponding v is the result of applying a row of M to the polynomial coefficients $\mathbf{a}$.  
```math
\mathbf{v} = \mathbf{M} \mathbf{a}
```
We then solve for the coefficients in terms of the known $\mathbf{v}$ values.
```math
\mathbf{a} = \mathbf{M}^{-1} \mathbf{v}
```
and evaluate the derivatives of the polynomial at $\delta_t=0$, yielding  
```math
\begin{aligned}
&\left. \partial_t R(t+\delta_t,r) \right|_{\delta_t = 0} = a_1  \\
&\left. \partial_t^2 R(t+\delta_t,r) \right|_{\delta_t = 0} = a_2/2
\end{aligned}
```


# Input Data  
Input data is hosted at  [cosmon nn_c103_205.05547](https://portal.nersc.gov/cfs/m2986/cosmon/nn_c103_2505.05547)  

Data can be downloaded with 
```sh
cd <your data directory>
wget -nd -r -P . -A "c103_n*.h5" https://portal.nersc.gov/cfs/m2986/cosmon/nn_c103_2505.05547/
```

The data files are large.   Verify that you have room for $\sim 25$ GB.  
