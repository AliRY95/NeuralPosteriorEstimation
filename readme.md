* First assumption was that parameter space is $\Theta = \[\kappa, d_{\tau} \]$ with fixed preferred direction.

 * Next, added the location dependancy of preferred direction, where at each location:
   
 $\vec{\theta_0(x)} = \dfrac{\vec{\nabla C(x)}}{|\vec{\nabla C(x)}|}$
 
* Chemical gradient equation, assuming source points, diffusing radially symmetric at distance r from each source:

$D \nabla^2 C(r) - \lambda C(r) = - Q \delta (r)$

with asympthotic solution


$C(r) = \dfrac{Q}{2 \pi D} K_0 (\dfrac{r}{L}), \quad L = \sqrt{D/\lambda},\quad K_0 = \text{Modified Bessel function of second kind.}$
* The over all chemical gradient at each point is the net effect of closest point sources.
* New parameter space: $\Theta = \[\kappa, d_{\tau}, Q, D, \lambda, (x_i,y_i)\]$ where $i = 1, \dots, B$ are the closest B point sources.
* First, assuming 1 single source point fixed in domain.
* For cases with cancer cells, we don't have any physical insights into some parameters such as Q, D, lambda. So, it makes more sense to use a wider prior for these values and with some help from SNPE, gradually converge to the actual values for these parameters.
* For the chemical field, let's just assume Q = Q / D, lambda = lambda / D.
