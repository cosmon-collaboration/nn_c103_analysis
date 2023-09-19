# Plan  

1) Fit several models to the HAL QCD potential data.
2) Extract phase shifts at a set of low energies.
3) Fit the effective range expansion to quadratic order (which defines the energy range for (2).
4) Conclude from the scattering length that there is or isn't a bound state.  $E_B = 1/(m a^2)$

THe smatrix subdirectory contains python code to compute the smatrix from a local potential.

# Fitting the HAL QCD Potential Data
To fit our HAL QCD potential data, we discussed using 3 models
- Yukawa with Gaussian regulator + r-weighted Woods-Saxon potentials.   Motivation:   This is the core of the S-channel Av18 potential which multiples a Woods-Saxon potential by a quadratic in $\mu r$ to fit the hard core and intermediate range.  The Yukawa regulator $1 - \exp(-c r^2)$ removes the non-physical short range behavior that comes from assuming that nucleons are point particles.
- Yukawa with Gaussian regulator + r-weighted Gaussians.    Motiviation:  r-weighted Gaussians are equivilent to the harmonic oscillator basis, which is complete.    You can include more powers of r and get improved representation of compact functions like the potential minus the regulated Yukawa.
- sum of Gaussians (as HAL QCD does), or is it r-weighted gaussians.   Motiviation:   We will do this mostly because HAL QCD did.   R-weighted Gaussians will eventually work with accurate long range data, but will require many powers to properly represent the tail.  The problem is that we don't have accurate long range data to configure the tail that we know must have a Yukawa form.  Previous HAL QCD analysis in [0909.5585](https://arxiv.org/abs/0909.5585) used a Yukawa + rho + Gaussian for the tensor force, but the central potential had too much noise to fit.

A starting paper for understanding the Gaussian regulated Yukawa and the Woods-Saxon potentials is the original AV18 paper
- [An Accurate nucleon-nucleon potential with charge independence breaking, Wiringa, Stoks, Schiavilla PRC 51 (1995) \[nucl-th/9408016\]](https://arxiv.org/pdf/nucl-th/9408016.pdf)
# Fitting the Effective Range Expansion
We will fit up to the $k^4$ term and use only $a$ and $r_{eff}$ later.  

# Bound State From Effective Range Expansion  
The S-channel effective range expansion is  
&nbsp;&nbsp;&nbsp; $k \cot\delta = -\frac{1}{a} + \frac{1}{2} r_{eff} k^2 + \cdots$   
The S-matrix pole occurs when $\cot\delta = i$, so we can solve for the value of $i\kappa = k$ where that occurs.  
&nbsp;&nbsp;&nbsp;$-\kappa =  -\frac{1}{a} - \frac{1}{2} r_{eff} \kappa^2$  
Solving for $\kappa$ we obtain  
&nbsp;&nbsp;&nbsp;$\kappa = \frac{1}{r}\left(1 \pm \sqrt{1 - 2 r_{eff}/a}\right)$   
Then we use the usual $\hbar^2 k^2 = 2 \mu E$ to obtain E.