# Introduction  

This directory has python code for extracting the S-matrix from a potential.
The initial code is for the single channel case.   If you need coupled-channel support ask Ken McElvain.

# The Schrodinger Equation  

$\partial_r^2 \psi + \left(k^2-\frac{L^2}{r^2}\right) \psi = W(r) \psi$   
For coupled channels $L$ is a diagonal matrix with $L_{i,i} = \ell_i(\ell_i + 1)$ and $W$ will have non-zero off diagonal elements.

$W(r) = \frac{2 \mu}{\left(\hbar c\right)^2} V(r)$   
where $V(r)$ is in some energy units.   

For it's use here we will use MeV, fm, and seconds.  The wave number $k$ will be in $\textrm{fm}^{-1}$.   $W(r)$ will be in fm$^{-2}$

# The Method   
The method used here is known as the variable phase method.    Essentially, one parameterizes the potential with a radial cutoff, $R$, outside of which the potential is forced to 0.   The S-matrix is written as a differential equation in the cutoff with the initial value of the S-matrix, $S(0) = 1$.   One uses an ODE solver to integrate out to where the potential is 0, accumulating phase shift as you go.    By integrating out in modest steps, one can accumulate the phase wrapping beyond $\pi$.    

The differential equation is  
$S'(R) = \frac{1}{{2ik}}\left[ {S(R)u_{out} \left( R \right) - u_{in} \left( R \right)} \right]\space W(R) \space \left[ {{u_{in}}\left( R \right) - {u_{out}}\left( R \right)S\left( R \right)} \right]$   

In this equation $u_{in}$ and $u_{out}$ are incoming and outgoing basis functions.   These will normally be Hankel functions.   If there is a coulomb interation, then one uses Coulomb wave functions for the basis functions and the Coulomb interation should be omitted from $W$.     

When working with coupled channels the various symbols $S$, $u$, $W$ become matrices.   The $u_{in}$ and $u_{out}$ matrices will be diagonal, with elements corresponding to the angular momentum matrix $L$ defined above.    

&nbsp; &nbsp;&nbsp; $u_{in} = \begin{pmatrix} h^{(2)}_{\ell_1}(r) & 0 \\ 0 & h^{(2)}_{\ell_2}(r) \\ \end{pmatrix}$,
&nbsp; &nbsp;&nbsp; $u_{out} = \begin{pmatrix} h^{(1)}_{\ell_1}(r) & 0 \\ 0 & h^{(1)}_{\ell_2}(r) \\ \end{pmatrix}$


