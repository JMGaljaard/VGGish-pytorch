import os
from typing import Union

import numpy as np
import torch
from models.vggish_torch.network.vggish import VGGish, Postprocessor
from models.vggish_torch.utils.utils import form_list_from_user_input, extract_wav_from_mp4
from models.vggish_torch.vggish import vggish_input
from tqdm import tqdm


class ExtractVGGish(torch.nn.Module):

    def __init__(self, args, initialize=True):
        super(ExtractVGGish, self).__init__()
        self.vggish = VGGish()
        self.postprocessor = Postprocessor()
        self.path_list = form_list_from_user_input(args)
        self.vggish_model_path = args.vggish_model_path
        self.vggish_pca_path = args.vggish_pca_path
        self.keep_audio_files = args.keep_frames
        self.on_extraction = args.on_extraction
        self.tmp_path = args.tmp_path
        self.output_path = args.output_path
        self.progress = tqdm(total=len(self.path_list))

        if initialize:
            self._initialize_pretrained_weights()

    def _initialize_pretrained_weights(self) -> None:
        self.vggish.load_state_dict(torch.load(self.vggish_model_path))
        self.postprocessor.load_state_dict(torch.load(self.vggish_pca_path))
        self.requires_grad_(False)

    def forward(self, indices: torch.LongTensor) -> None:
        '''
        Arguments:
            indices {torch.LongTensor} -- indices to self.path_list
        '''
        self.to(indices.device)
        # iterate over the list of videos
        for idx in indices:
            # when error occurs might fail silently when run from torch data parallel
            try:
                self.extract(indices.device, idx)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                # traceback.print_exc()  # for the whole traceback
                print(e)
                print(f'Extraction failed at: {self.path_list[idx]}. Continuing extraction')

            # update tqdm progress bar
            self.progress.update()

    def extract(self, device: torch.device, idx: int, video_path: Union[str, None] = None) -> torch.Tensor:
        '''The extraction call. Made to clean the forward call a bit.

        Args:
            idx (int): index to self.path_list
            video_path (Union[str, None], optional): . Defaults to None.
            device {torch.device}

        Keyword Arguments:
            video_path {Union[str, None]} -- if you would like to use import it and use it as
                                             "path -> i3d features"-fashion (default: {None})

        Returns:
            np.ndarray: extracted VGGish features
        '''
        # if video_path is not specified, take one from the self.path_list
        if video_path is None:
            video_path = self.path_list[idx]
        audio_wav_path, audio_aac_path = extract_wav_from_mp4(video_path, self.tmp_path)


        # Load Numpy data into tensor, first cast as float32, and move to device. Afterwards unsqueeze to match input
        # dimensions
        input_tensor = vggish_input.wavfile_to_examples(audio_wav_path).astype('float32', casting='same_kind')
        examples_batch = torch \
            .from_numpy(input_tensor) \
            .to(device) \
            .unsqueeze(1)

        # Move data back to CPU, when device is CPU, this is simply a noop
        vggish_stack = self.vggish(examples_batch).cpu()

        # removes the folder with extracted frames to preserve disk space
        if not self.keep_audio_files:
            os.remove(audio_wav_path)
            os.remove(audio_aac_path)

        # What to do once features are extracted.
        if self.on_extraction == 'print':
            print(vggish_stack)
            # print(vggish_stack.sum())
        elif self.on_extraction == 'save_numpy':
            # make dir if doesn't exist
            os.makedirs(self.output_path, exist_ok=True)
            # extract file name and change the extention
            filename = os.path.split(video_path)[-1].replace('.mp4', '_vggish.npy')
            # construct the paths to save the features
            feature_path = os.path.join(self.output_path, filename)
            # save features
            np.save(feature_path, vggish_stack)
        else:
            raise NotImplementedError

        return vggish_stack
