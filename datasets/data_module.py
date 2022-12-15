import numpy as np
import torch.utils.data as data
from models.multi_modality_perceiver import MultiModalityPerceiver, InputModality
import torch


class NpDataset(data.Dataset):

    def __init__(self, data_path, modalities, transforms=None):
        self.transforms = transforms
        d = np.load(data_path)
        for modality in modalities:

            setattr(self, modality, d[modality])
        self.modalities = sorted(modalities)
        self.transforms = transforms
        self.targets = torch.from_numpy(d['OCEAN'])

    def __getitem__(self, index):

        modality_data = []
        # sorted name of modalities
        for modality in self.modalities:
            data = getattr(self, modality)[index]
            # convert np to tensor
            data = torch.from_numpy(data).unsqueeze(-1)
            if self.transforms is not None:
                data = self.transforms(data)
            modality_data.append(data)
        # return dictionary with key is modality name and value is tensor
        return dict(zip(self.modalities, modality_data)), self.targets[index]


    def __len__(self):
        return len(self.targets)


# for each modality, the input_dim = input_axis * ((num_freq_bands * 2) + 1) + input_channels
        # e.g., the input_dim for video is 39

# input_dim
text_modality = InputModality(
        name='text',
        input_channels=1,  # number of channels for each token of the input
        input_axis=1,
        num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
        max_freq=8.,  # maximum frequency, hyperparameter depending on how fine the data is
    )

facebody_modality = InputModality(
        name='facebody',
        input_channels=1,  # number of channels for each token of the input
        input_axis=1,
        num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
        max_freq=8.,  # maximum frequency, hyperparameter depending on how fine the data is
    )

Modalities = {}
Modalities['text'] = text_modality
Modalities['facebody'] = facebody_modality