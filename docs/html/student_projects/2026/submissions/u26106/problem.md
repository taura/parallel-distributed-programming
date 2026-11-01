# Making a Cornell Box Path Tracer from Scratch

# Problem

I implement a small serial Monte Carlo path tracer in C++17.
The target scene is the classic Cornell Box:

- red wall, green wall, white walls/floor/ceiling,
- one rectangular ceiling area light,
- two interior white boxes.

The renderer outputs a PPM image. The goal is to build a clean serial baseline that can be parallelized later.

# How the algorithm/baseline code works

## 1) Rendering equation and Lambertian BRDF

For outgoing radiance $L_o$:

$$
L_o(x, \omega_o)
= L_e(x, \omega_o)
+ \int_{\mathcal{H}^2} f_r(x, \omega_i, \omega_o)\,L_i(x, \omega_i)\,(n\cdot \omega_i)\,d\omega_i
$$

For Lambertian surfaces:

$$
f_r = \frac{\rho}{\pi}
$$

where $\rho$ is albedo.

## 2) Direct lighting (Next Event Estimation)

At each diffuse hit point, I sample one random point on the rectangular area light.
If the sampled point is visible (shadow ray is not blocked), direct light is added using area-domain Monte Carlo:

$$
L_{\text{direct}} \approx
\frac{f_r\,L_e\,G(x,x_l)}{p_A(x_l)},
\quad
G(x,x_l)=\frac{\max(0,n\cdot \omega_i)\,\max(0,n_l\cdot (-\omega_i))}{\|x_l-x\|^2}
$$

For uniform sampling on the light, $p_A = 1/A$ where $A$ is light area.

## 3) Indirect lighting (recursive path tracing)

Indirect bounce directions are sampled with cosine-weighted hemisphere sampling:

$$
p(\omega_i)=\frac{\cos\theta}{\pi}
$$

The recursive Monte Carlo contribution for one sample is:

$$
L_{\text{indirect}} \sim
\frac{f_r\,L_i\,(n\cdot \omega_i)}{p(\omega_i)}
$$

Russian Roulette terminates long paths probabilistically after a minimum depth.

- Final output applies gamma correction (gamma $=2.2$) and writes a PPM file.

# Where is parallelism

This baseline has many independent computations:

- **Across pixels:** each pixel can be rendered independently.
- **Across samples:** per-pixel Monte Carlo samples are independent before reduction.
- **Across light/shadow tests:** each path step and visibility test is independent.

So the main parallel decomposition is data parallelism over image pixels (CPU threads or GPU threads), with reductions for sample accumulation.
