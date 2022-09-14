import os
import hydra
from hydra.core.hydra_config import HydraConfig

from omegaconf import DictConfig
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

from models import LitDLReg, LitGraDIRN
from utils.misc import MyModelCheckpoint

import random
random.seed(7)


import logging
log = logging.getLogger(__name__)


@hydra.main(config_path="conf/train", config_name="config")
def main(cfg: DictConfig) -> None:
    model_dir = HydraConfig.get().run.dir

    # use only one GPU
    gpus = None if cfg.gpu is None else 1
    if isinstance(cfg.gpu, int):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)

    # lightning model
    if cfg.network.type == 'gradirn':
        model = LitGraDIRN(**cfg)
    elif cfg.network.type == 'voxelmorph':
        model = LitDLReg(**cfg)
    else:
        raise ValueError(f'Network type {cfg.network.type} not recognised.')

    # log Lightning model summary using Hydra-configured logger
    log.info(model.summarize())

    # configure logger
    logger = TensorBoardLogger(model_dir, name='log')

    # model checkpoint callback with ckpt metric logging
    ckpt_callback = MyModelCheckpoint(save_last=True,
                                      dirpath=f'{model_dir}/checkpoints/',
                                      filename='{epoch}-{val_metrics/dice_avg:.3f}',
                                      monitor='val_metrics/dice_avg',
                                      mode='max',
                                      save_top_k=3,
                                      verbose=True
                                      )

    trainer = Trainer(default_root_dir=model_dir,
                      logger=logger,
                      callbacks=[ckpt_callback],
                      gpus=gpus,
                      precision=cfg.precision,
                      weights_summary=None,
                      **cfg.training.trainer
                      )

    # run training
    trainer.fit(model)


if __name__ == "__main__":
    main()
