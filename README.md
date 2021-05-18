# VGGish Pytorch
This repository provides a Pytorch implementation of the VGGish model architecture. In addition
it provides a script to load a Tensorflow `ckpt` file into the model. 

This repository was tested on Linux with Python 3.9.5, Pytorch 1.8 and Tensorflow 2.5.

# TODO

 - [x] Write network conversion script: see [`convert.py`](convert.py)
 - [x] Write PCA (Postprocessing) conversion script: see [`convert.py`](convert.py)
    * Note that this implements the conversion step as a `torch.nn.Linear` under the hood, by
    pre-calculating the biases using the PCA eigen-vector matrix and the PCA mean values.
 - [x] Write simple test: see [`adapted_smoketest.py`](adapted_smoketest.py).
    * Note that this test does not check properties (such as mean and std.) of the output, but
    rather checks that the original Tensorflow gives approximatly the same results.
 - [ ] Create `Dockerfile` to convert checkpoint in case Tensorflows `compat` is dropped. 
 - [ ] Make drop in replacement for `process` function.
 - [ ] Clean up code and documentation.
 - [ ] Extend `README.md`.

# Pretrained models

The [`model`](model) directory contains the pre-trained weights that can be loaded into the `VGGish` implementation.
Effectively this is rougly done as follows. You can also check [`adapted_smoketest.py`](adapted_smoketest.py)
for a slightly more complete example.

```python
import torch
from network.vggish import VGGish, Postprocessor

vggish_pytorch = VGGish()
postprocessor = Postprocessor()

vggish_pytorch.load_state_dict(torch.load("./model/vggish_model.pt"))
postprocessor.load_state_dict(torch.load("./model/vggish_postprocess.pt"))
```

# Notes
This repository does **not** provide code to train the VGGish, and was created to convert the 
VGGish model used by BMT into Pytorch compatible code.

The VGGish model is writen in Tensorflow V1 syntax (released by the authors of Tensorflow).
As such, the conversion script relies on Tensorflows compat to run a session with the model.

The results may deviate slightly, as Tensorflow and Pytorch use different optimization techniques, 
so the Pytorch network might give slightly different results.

# LICENSE

Released under the Apache 2 license, see [LICENSE](LICENSE).