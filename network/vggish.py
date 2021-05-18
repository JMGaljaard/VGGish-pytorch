import collections
import torch.nn

from vggish import vggish_params


class VGGish(torch.nn.Module):
    def __init__(self):
        """
        Constructor for VGGish model. Note that the ordered dict is chosen to make setting the weight easier by the
        convert script. E.g. the module object can be accessed as follows:

         > module = self.features._modules['vggish/conv1']

        The features and embeddings module of this class follow the same naming convention as the original VGGish
        implementation of Tensorflow.
        """
        super(VGGish, self).__init__()
        self.features = torch.nn.Sequential(
            collections.OrderedDict([
                # Conv block 1
                ("vggish/conv1", torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)),
                ("vggish/relu1", torch.nn.ReLU(inplace=True)),
                ("vggish/maxpool1", torch.nn.MaxPool2d(kernel_size=2, stride=2)),
                # Conv block 2
                ("vggish/conv2", torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)),
                ("vggish/relu2", torch.nn.ReLU(inplace=True)),
                ("vggish/maxpool2", torch.nn.MaxPool2d(kernel_size=2, stride=2)),

                # Conv block 3
                ("vggish/conv3/conv3_1",
                 torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)),
                ("vggish/conv3/relu3_1", torch.nn.ReLU(inplace=True)),
                ("vggish/conv3/conv3_2",
                 torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)),
                ("vggish/conv3/relu3_2", torch.nn.ReLU(inplace=True)),
                ("vggish/maxpool3", torch.nn.MaxPool2d(kernel_size=2, stride=2)),

                # Conv block 4
                ("vggish/conv4/conv4_1",
                 torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)),
                ("vggish/conv4/relu4_1", torch.nn.ReLU()),
                ("vggish/conv4/conv4_2",
                 torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)),
                ("vggish/conv4/relu4_2", torch.nn.ReLU(inplace=True)),
                ("vggish/conv4/maxpool4", torch.nn.MaxPool2d(kernel_size=2, stride=2))]))

        self.embedding = torch.nn.Sequential(
            collections.OrderedDict([
                # Alternatively a view operation can be used.
                ("vggish/flatten", torch.nn.Flatten()),
                # FC block 1
                ("vggish/fc1/fc1_1", torch.nn.Linear(in_features=12288, out_features=4096)),
                ("vggish/fc1/relu1_1", torch.nn.ReLU(inplace=True)),
                ("vggish/fc1/fc1_2", torch.nn.Linear(in_features=4096, out_features=4096)),
                ("vggish/fc1/relu1_2", torch.nn.ReLU(inplace=True)),
                # FC block 2
                ("vggish/fc2", torch.nn.Linear(in_features=4096, out_features=128)),
                ("vggish/f2/relu", torch.nn.ReLU(inplace=True)),
            ]))


    def forward(self, x):
        # Perform feature extraction with CNN module, and change layout to make compatible with embedding layers.
        x = self.features(x).permute(0, 2, 3, 1)
        # Perform embedding generation with FC module
        x = self.embedding(x)
        return x


class Postprocessor(torch.nn.Module):

    def __init__(self) -> None:
        super(Postprocessor, self).__init__()
        # Initialize as layer. Originally 128 * 128
        self.layer = torch.nn.Linear(vggish_params.EMBEDDING_SIZE, vggish_params.EMBEDDING_SIZE)

    def forward(self, embeddings_batch):
        # Perform projection onto PCA vectors.
        embeddings_batch = self.layer(embeddings_batch)

        # The following code is based on the original Tensorflow code, see also:
        # https://github.com/tensorflow/models/blob/4079c5d9693142a406f6ff392d14e2034b5f496d/research/audioset/vggish/vggish_postprocess.py#L68-L89
        # Quantize by:
        # - Clip to domain of [min, max]
        clipped_embeddings = torch.clamp(
            embeddings_batch, vggish_params.QUANTIZE_MIN_VAL,
            vggish_params.QUANTIZE_MAX_VAL)
        # - Change to 8-bit domain [0.0, 255.0]
        quantized_embeddings = (
                (clipped_embeddings - vggish_params.QUANTIZE_MIN_VAL) *
                (255.0 /
                 (vggish_params.QUANTIZE_MAX_VAL - vggish_params.QUANTIZE_MIN_VAL)))
        # - Change data type from float to uint8
        return quantized_embeddings.to(torch.uint8)
