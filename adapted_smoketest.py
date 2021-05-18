# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""An adapted smoke test for VGGish.
This is a simple smoke test of a local install of VGGish and its associated
downloaded files. We create a synthetic sound, extract log mel spectrogram
features, run them through VGGish, post-process the embedding ouputs, and
check some simple statistics of the results, allowing for variations that
might occur due to platform/version differences in the libraries we use.
Usage:
- Download the VGGish checkpoint and PCA parameters into the same directory as
  the VGGish source code. If you keep them elsewhere, update the checkpoint_path
  and pca_params_path variables below.
- Run:
  $ python adapted_smoketest.py
"""

from __future__ import print_function

from datetime import datetime

import numpy as np
import tensorflow.compat.v1 as tf
import torch

from network.vggish import VGGish, Postprocessor
from vggish.vggish_input import waveform_to_examples

torch.backends.cudnn.deterministic = True

from vggish import vggish_params, vggish_slim, vggish_postprocess

print('\nTesting your install of VGGish\n')

checkpoint_path = '/home/jeroen/Documents/CSE/MSc/year/1/Q4/CS4245/repo/BMT/submodules/video_features/models/vggish/checkpoints/vggish_model.ckpt'
pca_params_path = '/home/jeroen/Documents/CSE/MSc/year/1/Q4/CS4245/repo/BMT/submodules/video_features/models/vggish/checkpoints/vggish_pca_params.npz'

# Relative tolerance of errors in mean and standard deviation of embeddings.
rel_error = 0.1  # Up to 10%

# Generate a 1 kHz sine wave at 44.1 kHz (we use a high sampling rate
# to test resampling to 16 kHz during feature extraction).
num_secs = 100
freq = 1000
sr = 44100
t = np.arange(0, num_secs, 1 / sr)
x = np.sin(2 * np.pi * freq * t)

# Produce a batch of log mel spectrogram examples.
input_batch = waveform_to_examples(x, sr)
print('Log Mel Spectrogram example: ', input_batch[0])
np.testing.assert_equal(
    input_batch.shape,
    [104, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS])

# Define VGGish, load the checkpoint, and run the batch through the model to
# produce embeddings.
with tf.Graph().as_default(), tf.Session() as sess:
    # Load TF model
    vggish_slim.define_vggish_slim()
    vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)

    # Load Pytorch model
    torch_vgg = VGGish()
    torch_vgg.load_state_dict(torch.load("model/vggish_model.pt"))

    features_tensor = sess.graph.get_tensor_by_name(
        vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(
        vggish_params.OUTPUT_TENSOR_NAME)
    time = datetime.now()
    [embedding_batch] = sess.run([embedding_tensor],
                                 feed_dict={features_tensor: input_batch})
    print(f"Tensorflow embedding time:\t {datetime.now() - time}")

    time = datetime.now()
    embedding_batch_torch = torch_vgg(torch.from_numpy(input_batch).unsqueeze(1).float())
    print(f"Torch embedding time:\t {datetime.now() - time}")

torch_post_process = Postprocessor()
pca_params = np.load(pca_params_path)
torch_post_process.load_state_dict(torch.load("model/vggish_postprocess.pt"))

pproc = vggish_postprocess.Postprocessor(pca_params_path)
time = datetime.now()
postprocessed_batch = pproc.postprocess(embedding_batch)
print(f"Numpy postprocessing time:\t {datetime.now() - time}")
time = datetime.now()
postprocessed_batch_torch = torch_post_process(embedding_batch_torch)
print(f"Torch postprocessing time:\t {datetime.now() - time}")
# Test embeddings
np.testing.assert_equal(postprocessed_batch, postprocessed_batch_torch.detach().numpy())
print("Post processed values are all equal")
# Test hidden representation
np.testing.assert_allclose(embedding_batch, embedding_batch_torch.detach().numpy(), atol=1e-5)
print("Post processed values are ~equal (atol 1e-5)")
