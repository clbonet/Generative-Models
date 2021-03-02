# Normalizing Flows

- NICE
- Real NVP
- Glow
- Planar Flow
- Radial Flow
- Neural Spline Flow
- Masked Autoregressive Flow (with MADE)
- Flow++
- Dynamic Linear FLow

To Do:
- Deep Dense Sigmoidal Flow
- Sylvester Flow (with VAE)
- Convex Potential Flow
- ResFlow/IRes (problem with estimation gradient)

Investigate:
- cpu memory leak in train_mnist?
- IAF: error (out of [0,1]?)
- Flow++: on Mnist, problem with likelihood. (problem with dims in log_det???)
- iResNet: sometimes, likelood fall + problem estimation log_det (when we do not take the exact one)
