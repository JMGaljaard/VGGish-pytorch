# VGGish Pytorch
This repository provides a Pytorch implementation of the VGGish model architecture. In addition
it provides a script to load a Tensorflow `ckpt` file into the model. 

This repository was tested on Linux with Python 3.9.5, Pytorch 1.8 and Tensorflow 2.5.

## TODO

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

## Create models

## Pre-generated files
You can download the models to the [`model`](models). These can then be directly loaded into the `VGGish` implementation.
The model files are available on Google Drive.
 
 * model/vggish_postprocess.pt [link](https://drive.google.com/file/d/1NK2V5n5AMUdGftaCTnvJ2sLry0xzNa7v/view?usp=sharing)
    * 66 KB 
    * md5sum `c79bb5af1ba6711de57bf680a22b052e`
 * model/vggish_model.pt [link](https://drive.google.com/file/d/1s4-n58ZClFJwVbnrO74qgn8leir8Dj4l/view?usp=sharing)
    * 275 MB
    * md5sum `d89a7384cf485a4039ad3fbb9a2612f3`

Effectively the loading is done as follows. You can also check [`adapted_smoketest.py`](adapted_smoketest.py)
for a more complete example.

```python
import torch
from network.vggish import VGGish, Postprocessor

vggish_pytorch = VGGish()
postprocessor = Postprocessor()

vggish_pytorch.load_state_dict(torch.load("model/vggish_model.pt"))
postprocessor.load_state_dict(torch.load("model/vggish_postprocess.pt"))
```

### Generate from Checkpoint
To generate the files yourself, first download the original Tensorflow checkpoint file as follows. In addition, download
the PCA parameter files. Alternatively, in case you need to convert your own version of VGGish, you can change the 
variables in [`convert.py`](convert.py) to point to your own files. Make sure that the layers have the same names
as the original Tensorflow implementation.

```bash
wget https://storage.googleapis.com/audioset/vggish_model.ckpt -O model/vggish_model.ckpt
wget https://storage.googleapis.com/audioset/vggish_pca_params.npz -O model/vggish_pca_params.npz
```

Then run the `convert.py` script from the vggish_torch model as follows.

```bash
python3 convert.py
```


## Notes
This repository does **not** provide code to train the VGGish, and was created to convert the 
VGGish model used by BMT into Pytorch compatible code.

The VGGish model is writen in Tensorflow V1 syntax (released by the authors of Tensorflow).
As such, the conversion script relies on Tensorflows compat to run a session with the model.

The results may deviate slightly, as Tensorflow and Pytorch use different optimization techniques, 
so the Pytorch network might give slightly different results.

The code in [`vggish`](vggish) is adapted code from [Tensorflow models](https://github.com/tensorflow/models/blob/4079c5d9693142a406f6ff392d14e2034b5f496d/research/audioset/vggish/)
with a few modifications to be compatible with Tensorflow 2.

## Issues
In case you run into something, open a new issue. (Or better yet, create a pull request!) 
Depending on my availability, my response may be a bit delayed.

## LICENSE

Released under the Apache 2 license, see [LICENSE](LICENSE).