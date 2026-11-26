"""Fine-tuning trainers: nnU-Net's default trainer with a lower starting learning rate.

The default nnUNetTrainer starts at lr 0.01 and decays it to ~0 over the run (poly schedule). That suits training
from scratch; for a model that is already trained (rank 2's weights) restarting at 0.01 first shakes it up. These
trainers start at 0.001 instead and keep everything else (loss, augmentation, optimizer, 250 iterations per epoch)
the same.

nnU-Net finds trainers by class name. Classes outside its own package are found through the environment variable
`nnUNet_extTrainer`, which `src/train/train.py` sets to this folder.

Snapshots: nnU-Net keeps only `checkpoint_latest.pth` (overwritten every 50 epochs), `checkpoint_best.pth` and
`checkpoint_final.pth`. These trainers also keep a copy every 100 epochs, `checkpoint_ep100.pth`, `..._ep200.pth`,
..., so the model can be compared at several points of the run afterwards. Each is a full checkpoint (~0.8 GB:
weights + optimizer state), loadable for prediction or to continue from.
"""
import os
import shutil

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
        self.snapshot_every = 100
        assert self.snapshot_every % self.save_every == 0, "snapshots copy checkpoint_latest.pth"

    def on_epoch_end(self):
        epoch = self.current_epoch          # nnU-Net increments it at the end of on_epoch_end
        super().on_epoch_end()
        # At a multiple of save_every, nnU-Net has just written checkpoint_latest.pth for this epoch: keep a copy.
        # The last epoch is skipped (nnU-Net writes no latest there; checkpoint_final.pth covers it).
        if (self.local_rank == 0 and not self.disable_checkpointing
                and (epoch + 1) % self.snapshot_every == 0 and epoch != self.num_epochs - 1):
            latest = os.path.join(self.output_folder, 'checkpoint_latest.pth')
            snapshot = os.path.join(self.output_folder, f'checkpoint_ep{epoch + 1}.pth')
            shutil.copyfile(latest, snapshot)
            self.print_to_log_file(f'Snapshot saved: {os.path.basename(snapshot)}')


class nnUNetTrainer_FT_5epochs(nnUNetTrainer_FT):
    """Timing test: same settings, 5 epochs."""
    FT_EPOCHS = 5


class nnUNetTrainer_FT_250epochs(nnUNetTrainer_FT):
    """Shorter fine-tune, if 500 epochs takes too long."""
    FT_EPOCHS = 250
