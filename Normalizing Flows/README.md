# Normalizing Flows

- NICE
- Real Non Volume Preserving (RealNVP)
- Glow
- Planar Flow
- Radial Flow
- Neural Spline Flow
- Masked Autoregressive Flow (with MADE)
- Flow++
- Dynamic Linear FLow
- Convex Potential Flow (with Quadratic Dense ICNN)
- Block Neural Autoregressive Flow (BNAF)

On the Sphere:
- Radial Exponential map


To Do:
- Deep Dense Sigmoidal Flow
- Sylvester Flow (with VAE)
- ResFlow/IRes (problem with estimation gradient)

Investigate:
- cpu memory leak in train_mnist?
- IAF: error (out of [0,1]?)
- Flow++: on Mnist, problem with likelihood. (problem with dims in log_det???)
- iResNet: sometimes, likelood fall + problem estimation log_det (when we do not take the exact one)
