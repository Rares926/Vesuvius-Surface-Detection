"""Fine-tuning trainers: nnU-Net's default trainer with a lower starting learning rate.

The default nnUNetTrainer starts at lr 0.01 and decays it to ~0 over the run (poly schedule). That suits training
from scratch; for a model that is already trained (rank 2's weights) restarting at 0.01 first shakes it up. These
trainers start at 0.001 instead and keep everything else (loss, augmentation, optimizer, 250 iterations per epoch)
the same.

nnU-Net finds trainers by class name. Classes outside its own package are found through the environment variable
`nnUNet_extTrainer`, which `src/train/train.py` sets to this folder.
"""
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_FT(nnUNetTrainer):
    """Fine-tune: start lr 1e-3, 500 epochs."""
    FT_EPOCHS = 500
    FT_LR = 1e-3

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = self.FT_EPOCHS
        self.initial_lr = self.FT_LR


class nnUNetTrainer_FT_5epochs(nnUNetTrainer_FT):
    """Timing test: same settings, 5 epochs."""
    FT_EPOCHS = 5


class nnUNetTrainer_FT_250epochs(nnUNetTrainer_FT):
    """Shorter fine-tune, if 500 epochs takes too long."""
    FT_EPOCHS = 250
