import numpy as np
import tensorflow.compat.v1 as tf
import torch

import vggish.vggish_slim as vggish_slim
from network.vggish import VGGish, Postprocessor
from utils.params_to_torch import numpy_to_post_process, set_layer

checkpoint_path = 'model/vggish_model.ckpt'
pca_params_path = 'model/vggish_pca_params.npz'

model_dir = "model"
# Load the model with default graph (is a tensorflow v1 version).
with tf.Graph().as_default(), tf.Session() as sess:
    # Use implementation used by BMT authors
    vggish_slim.define_vggish_slim()
    vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)

    # Get the variables
    variables = tf.global_variables()
    variables_name = [x.name for x in variables]
    # Get the actual variables, these are lazily loaded and therefore require the session to run
    variables_real = sess.run(variables)

    # Create VGG(ish) network
    torch_vgg = VGGish()

    # For each layer set the pre-trained weights to the torch implementation
    for name, pre_trained in list(zip(variables_name, variables_real)):
        set_layer(name, pre_trained, torch_vgg)

    # Save the model to the save dir
    torch.save(torch_vgg.state_dict(), f"{model_dir}/vggish_model.pt")

    # Create torch implementation of PostProcess function
    torch_post_process = Postprocessor()

    # Load the parameters from the provided file
    pca_params = np.load(pca_params_path)

    # Load the parameters (eigen vectors of PCA) and means into implementation
    numpy_to_post_process(pca_params['pca_eigen_vectors'], pca_params['pca_means'], torch_post_process)

    # Save the models to the save dir
    torch.save(torch_post_process.state_dict(), f"{model_dir}/vggish_postprocess.pt")
