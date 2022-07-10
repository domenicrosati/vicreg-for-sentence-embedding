import random
from sentence_transformers import SentenceTransformer
from typing import Iterable, Dict, Tuple
import os
import json
from torch.utils.data import Dataset
from typing import List
from sentence_transformers.readers.InputExample import InputExample

import torch
from torch import Tensor, nn
from torch.nn import functional as F

class Expander(nn.Module):
    def __init__(self,
                 pooling_output_dimension: int,
                 expander_dimension: int
        ):
        super(Expander, self).__init__()

        self.config_keys = [
          'pooling_output_dimension',
          'expander_dimension'
        ]
s
        self.pooling_output_dimension = pooling_output_dimension
        self.expander_dimension = expander_dimension

        self.expander = nn.Sequential(
            nn.Linear(self.pooling_output_dimension, self.expander_dimension),
            nn.BatchNorm1d(self.expander_dimension),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(self.expander_dimension, self.expander_dimension),
            nn.BatchNorm1d(self.expander_dimension),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(self.expander_dimension, self.expander_dimension))

    def __repr__(self):
        return "Expander({})".format(self.get_config_dict())

    def forward(self, features: Dict[str, Tensor]):
        features.update({'expansion_embedding': self.expander(features['sentence_embedding'])})
        return features

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        return Expander(**config)



# loss has been adapted from https://github.com/facebookresearch/vicreg
sim_loss = nn.MSELoss()

# variance loss
def std_loss(z_a, z_b):
    std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-04)
    std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-04)
    std_loss = torch.mean(F.relu(1 - std_z_a)) + torch.mean(F.relu(1 - std_z_b))
    return std_loss


#function taken from https://github.com/facebookresearch/barlowtwins/tree/a655214c76c97d0150277b85d16e69328ea52fd9
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

# covariance loss
def cov_loss(z_a, z_b):
    N = z_a.shape[0]
    D = z_a.shape[1]
    z_a = z_a - z_a.mean(dim=0)
    z_b = z_b - z_b.mean(dim=0)
    cov_z_a = (z_a.T @ z_a) / (N - 1)
    cov_z_b = (z_b.T @ z_b) / (N - 1)
    cov_loss = off_diagonal(cov_z_a).pow_(2).sum() / D + off_diagonal(cov_z_b).pow_(2).sum() / D
    return cov_loss


class VicRegLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, l=25, mu=25, nu=1,  labeled_loss_fct = nn.MSELoss()):
        super(VicRegLoss, self).__init__()
        self.model = model
        self.l = l
        self.mu = mu
        self.nu = nu
        self.labeled_loss_fct = labeled_loss_fct


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        embeddings = [self.model(sentence_feature)['expansion_embedding'] for sentence_feature in sentence_features]

        _sim_loss = sim_loss(embeddings[0], embeddings[1])
        _std_loss = std_loss(embeddings[0], embeddings[1])
        _cov_loss = cov_loss(embeddings[0], embeddings[1])

        if len(labels) > 0:
            loss = self.l * self.labeled_loss_fct(_sim_loss, labels.view(-1)) + self.mu * _std_loss + self.nu * _cov_loss
        else:
            loss = self.l * _sim_loss + self.mu * _std_loss + self.nu * _cov_loss
        return loss


class SameDataset(Dataset):
    def __init__(self, sentences: List[str]):
        self.sentences = sentences


    def __getitem__(self, item):
        sent = self.sentences[item]
        return InputExample(texts=[sent, sent])


    def __len__(self):
        return len(self.sentences)


class WordCropDataset(Dataset):
    def __init__(self, sentences: List[str], crop_both=False):
        self.sentences = sentences
        self.crop_both = crop_both


    def __getitem__(self, item):
        sent = self.sentences[item]
        sent_a, sent_b = WordCropDataset.crop_sentence(sent, self.crop_both)
        return InputExample(texts=[sent_a, sent_b])


    def __len__(self):
        return len(self.sentences)

    @staticmethod
    def crop_sentence(sentence: str, crop_both=False):
        """
        Randomly crop a subset of tokens from a sentence
        """
        tokens = sentence.split()
        if len(tokens) < 2:
            return sentence, sentence

        start_position_a = random.randint(0, len(tokens) - 1)
        crop_length_a = random.randint(1, len(tokens) - start_position_a)
        if crop_both:
            start_position_b = random.randint(0, len(tokens) - 1)
            crop_length_b = random.randint(1, len(tokens) - start_position_b)
            return ' '.join(tokens[start_position_a:start_position_a + crop_length_a]), ' '.join(tokens[start_position_b:start_position_b + crop_length_b])

        return sentence, ' '.join(tokens[start_position_a:start_position_a + crop_length_a])
