## Instant Neural Graphics Primitives in JAX (non-official)
---
## Table of Contents
- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Prerequisites](#Prerequisites)
- [Reference](#reference)

## Overview
The instant ngp in pure JAX still needs optimization. Currently, it is way slower than the CUDA version. 


## Prerequisites
 - Set up a virtual Python3 environment, then install necessary packages with this command
   ```Shell
   conda create --name jax_ngp python=3
   conda activate jax_ngp
   pip install -r requirements.txt
   ```
 - Or, just open Google Colab and run **notebook_train.ipynb** 
---
## Reference
[[1]] Mildenhall, Ben, et al. "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis." ECCV (2020).

[1]: https://arxiv.org/abs/2003.08934 "NeRF"

[[2]] Müller, Thomas, et al. "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding." SIGGRAPH (2022).

[2]: https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf "NGP"
