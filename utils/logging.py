import os
import time
import logging
from torch import Tensor
from argparse import Namespace
from lightning_lite.utilities.types import _PATH
from lightning_lite.utilities.cloud_io import get_filesystem
from typing import Any, Dict, Mapping, Optional, Union
from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
from pytorch_lightning.utilities.logger import _add_prefix
from pytorch_lightning.utilities import rank_zero_only
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from .log_buffer import LogBuffer

log = logging.getLogger(__name__)


def get_logger(level_print, level_save, path_file, name_logger = "logger"):
    # level: logging.INFO / logging.WARN
    logger = logging.getLogger(name_logger)
    logger.setLevel(level = logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    # set file handler
    handler_file = logging.FileHandler(path_file)
    handler_file.setLevel(level_save)
    handler_file.setFormatter(formatter)
    logger.addHandler(handler_file)
    # set console holder
    handler_view = logging.StreamHandler()
    handler_view.setFormatter(formatter)
    handler_view.setLevel(level_print)
    logger.addHandler(handler_view)
    return logger

def get_logger_info(prefix, dict_info):
    info = prefix
    for key, value in dict_info.items():
        if 'T_' in key:
            info = info + '{}: {:.3f}\t'.format(key, value)
        elif 'lr' in key:
            info = info + '{}: {:.6f}\t'.format(key, value)
        else:
            info = info + '{}: {:.5f}\t'.format(key, value)

    return info


class MyPrintingCallback(Callback):
    def __init__(
        self,
        save_dir: _PATH,
        name: Optional[str] = "lightning_logs",
        version: Optional[Union[int, str]] = None,
        log_every_n_steps: int = 1,
    ):
        super().__init__()
        save_dir = os.fspath(save_dir)
        self._save_dir = save_dir
        self._name = name or ""
        self._version = version
        self._fs = get_filesystem(save_dir)

        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)
        self.logger = get_logger(
            level_print=logging.INFO, level_save=logging.WARNING, path_file=self.log_dir+"/training_logger.log"
        )

        self._log_every_n_steps = log_every_n_steps
        self.log_buffer = LogBuffer()
    
    @property
    def root_dir(self) -> str:
        return os.path.join(self.save_dir, self.name)
    
    @property
    def log_dir(self) -> str:
        # create a pseudo standard path ala test-tube
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        log_dir = os.path.join(self.root_dir, version)
        log_dir = os.path.expandvars(log_dir)
        log_dir = os.path.expanduser(log_dir)
        return log_dir

    @property
    def save_dir(self) -> str:
        return self._save_dir

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> Union[int, str]:
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    def _get_next_version(self) -> int:
        root_dir = self.root_dir

        try:
            listdir_info = self._fs.listdir(root_dir)
        except OSError:
            log.warning("Missing logger folder: %s", root_dir)
            return 0

        existing_versions = []
        for listing in listdir_info:
            d = listing["name"]
            bn = os.path.basename(d)
            if self._fs.isdir(d) and bn.startswith("version_"):
                dir_ver = bn.split("_")[1].replace("/", "")
                existing_versions.append(int(dir_ver))
        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1
    
    @rank_zero_only
    def log_experiment(self, outputs, batch_idx):
        self.log_buffer.update(outputs)
        if batch_idx % self._log_every_n_steps == 0:
            self.log_buffer.average(self._log_every_n_steps)
            prefix = 'Iter {} Train - '.format(str(batch_idx).zfill(6))
            write_info = get_logger_info(
                    prefix, dict_info=self.log_buffer._output)
            self.logger.info(write_info)

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.epoch_time = time.time()
        return super().on_train_epoch_start(trainer, pl_module)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        train_time = (time.time()-self.epoch_time) / 60.0
        dict_info_epoch = self.log_buffer.avg
        prefix = 'Epoch {} - train_time(min): {:.5f} '.format(trainer.current_epoch, train_time)
        write_info = get_logger_info(prefix, dict_info=dict_info_epoch)
        self.logger.warning(write_info)
        
        self.log_buffer.clear()
        return super().on_train_epoch_end(trainer, pl_module)

    def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int) -> None:
        return super().on_train_batch_start(trainer, pl_module, batch, batch_idx)

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module, outputs, batch, batch_idx) -> None:
        # outputs['lr'] = trainer.optimizers[0].state_dict()['param_groups'][0]['lr']
        outputs['lr'] = trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]

        self.log_experiment(outputs, batch_idx)

        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

