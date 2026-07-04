## Natural Selection and Genetic Drift Simulation

This code implements solutions to Kimura's drift-diffusion equation, using the PDE to find the time evolution of the probability density function (PDF) of allele frequencies in a population under the influence of genetic drift and natural selection. 

The code also simulates the corresponding stochastic differential equation (SDEs) to sample trajectories of allele frequencies over time, allowing for a comparison between the PDE solution and the SDE simulations.

Comparisons to theoretical results in population genetics are also included (fixation and loss probabilities).

## Theory

Suppose we have a population of effective size $ N_e $ and a biallelic locus with alleles A and a. Let $ x $ denote the frequency of allele A in the population at time $ t $.

If the initial allele frequency is known to be exactly $ x_0 $ at time $ t = 0 $, we can represent this as a Dirac delta function:

$$ \phi(x, 0) = \delta(x - x_0) $$

where $ \phi(x, t) $ is the probability density function of the allele frequency being $ x $ at time $ t $.

The subsequent evolution (change in allele frequency over time) under genetic drift and natural selection can be described by Kimura's drift-diffusion equation, which is a partial differential equation (PDE):

$$ \frac{\partial \phi(x, t)}{\partial t} = -\frac{\partial}{\partial x} \left[ s x (1 - x) \phi(x, t) \right] + \frac{1}{2 N_e} \frac{\partial^2}{\partial x^2} \left[ x (1 - x) \phi(x, t) \right] $$

where $ s $ is the selection coefficient for allele A.

To give this problem a unique solution, we require boundary conditions at $ x = 0 $ and $ x = 1 $. These are absorbing boundaries, meaning that if the allele frequency reaches either boundary, it will remain there (fixation or loss of the allele).

A suitable numerical method for solving this PDE (as a Fokker-Planck equation) is the Chang-Cooper scheme, which is a finite difference method that preserves the positivity and normalization of the probability density function.

To calculate the probability of fixation or loss of allele A, we define the probability flux $ J(x, t) $ as:

$$ J(x, t) = s x (1 - x) \phi(x, t) - \frac{1}{2 N_e} \frac{\partial}{\partial x} \left[ x (1 - x) \phi(x, t) \right] $$

such that the PDE can be rewritten as a continuity equation:

$$ \frac{\partial \phi(x, t)}{\partial t} = -\frac{\partial J(x, t)}{\partial x}. $$

The probability of fixation of allele A is given by the time-integrated flux at $ x = 1 $:

$$ P_{\text{fix}}(t) = \int_0^t J(1, \tau) d\tau $$

Likewise, the probability of loss of allele A is given by the time-integrated flux at $ x = 0 $:

$$ P_{\text{loss}}(t) = \int_0^t J(0, \tau) d\tau $$

The probability that the allele frequency is still segregating (neither fixed nor lost) at time $ t $ is given by:

$$ P_{\text{seg}}(t) = \int_0^1 \phi(x, t) dx = 1 - P_{\text{fix}}(t) - P_{\text{loss}}(t). $$

To simulate individual trajectories of allele frequencies, we can use the corresponding stochastic differential equation (SDE):

$$ dx = s x (1 - x) dt + \sqrt{\frac{x (1 - x)}{2 N_e}} dW_t $$

where $ dW_t $ is a Wiener process (standard Brownian motion).

A suitable numerical method for generating sample paths of this SDE is the Milstein method.

## Results

![Kimura's PDE and SDE Comparison](output/kimura_pde_sde_comparison.svg)