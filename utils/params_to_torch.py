import numpy as np
import torch

from network.vggish import Postprocessor, VGGish


def tensor_mapper(pre_trained: np.array) -> torch.Tensor:
    """
    Transpose the tensor depending on whether it is and FC or CN layer to match dimensions with
    Pytorch implementation.
    """
    if len(pre_trained.shape) == 4:
        tensor = torch.from_numpy(pre_trained.transpose(3, 2, 0, 1)).float()
    else:
        tensor = torch.from_numpy(pre_trained.T).float()
    return tensor


def set_layer(name, pre_trained, model: VGGish):
    """
    Utility function that sets the models layer with the pre-trained (Tensorflow) weights.
    """
    # Get the name corresponding to the networks layers.
    module_name = name.rsplit('/', 1)[0]
    if 'conv' in module_name:
        module: torch.nn.Module = model.features._modules[module_name]
    elif 'fc' in module_name:
        module = model.embedding._modules[module_name]
    else:
        raise Exception(f"{name} Unexpected name, please try again!")
    tensor = tensor_mapper(pre_trained)
    print(f"{name}\t Pytorch shape: {module.bias.shape}\t Tensorflow shape:  {pre_trained.T.shape} (transposed)")
    if 'bias' in name:
        module.bias = torch.nn.Parameter(tensor, requires_grad=False).float()
    else:
        module.weight = torch.nn.Parameter(tensor, requires_grad=False).float()


def numpy_to_post_process(pca_matrix, pca_means, model: Postprocessor):
    """
    # Note that we can 'pre-compute' the effect of subtracting the means, as a layer effectively
    # implements a linear system of equations (i.e. a matrix multiplication). (More precisely,
    # the commutativity property of linear systems allows for this trick)
    # As such, this is equivalent to the code provided by Tensorflow

    # Our code
    self.layer(embeddings_batch)
    # Tensorflow
    pca_applied = np.dot(self._pca_matrix,
                     (embeddings_batch.T - self._pca_means)).T
    """
    model.layer.weight = torch.nn.Parameter(torch.from_numpy(pca_matrix).float(), requires_grad=False)
    model.layer.bias = torch.nn.Parameter(-torch.from_numpy(np.matmul(pca_matrix, pca_means).T).float(),
                                          requires_grad=False)
