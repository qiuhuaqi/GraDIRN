import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.base import merge_dicts
from torch.optim import Adam
from torch.utils.data import DataLoader

from core.transformations import warp
from core.utils import interpolate_nd
from models.utils import get_network, get_loss_fn, get_datasets
from utils.metrics import measure_metrics
from utils.misc import worker_init_fn
from utils.visualise import visualise_result


class LitDLReg(LightningModule):
    """ DL registration base Lightning module"""
    def __init__(self, *args, **kwargs):
        super(LitDLReg, self).__init__()
        self.save_hyperparameters()

        self.train_dataset, self.val_dataset = get_datasets(self.hparams)
        self.network = get_network(self.hparams)
        self.sim_loss_fn, self.reg_loss_fn = get_loss_fn(self.hparams)

        # initialise dummy best metrics results for initial logging
        self.hparam_metrics = {f'hparam_metrics/{m}': 0.0 for m in self.hparams.hparam_metrics}

    def on_fit_start(self):
        # log dummy initial hparams w/ best metrics (for tensorboard HPARAMS tab)
        self.logger.log_hyperparams(self.hparams, metrics=self.hparam_metrics)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.data.batch_size,
                          shuffle=self.hparams.data.shuffle,
                          num_workers=self.hparams.data.num_workers,
                          pin_memory=self.on_gpu,
                          worker_init_fn=worker_init_fn
                          )

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=self.hparams.data.num_workers,
                          pin_memory=self.on_gpu,
                          worker_init_fn=worker_init_fn
                          )

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.training.lr)
        if self.hparams.training.lr_decay_step:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                        step_size=self.hparams.training.lr_decay_step,
                                                        gamma=0.1,
                                                        last_epoch=-1)
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    def forward(self, tar, src):
        # pure network forward pass
        out = self.network(tar, src)
        return out

    def inference(self, batch):
        """ Forward pass and warping source image """
        tar, src = batch['tar'], batch['src']
        out = self.forward(tar, src)
        warped_src = warp(src, out['disp'][-1])  # TODO: use warp_fn() with shared grid
        return {'disp': out, 'warped_src': warped_src}

    def loss_fn(self, outputs, batch):
        tar = batch['tar']
        warped_src = outputs['warped_src']

        losses = {}

        # (dis-)similarity loss
        sim_loss = self.sim_loss_fn(tar, warped_src) * self.hparams.loss.sim_loss.weight
        losses['sim_loss'] = sim_loss
        loss = sim_loss

        # regularisation loss
        if self.reg_loss_fn:
            disp = outputs['disp'][-1]
            reg_loss = self.reg_loss_fn(disp) * self.hparams.loss.reg_loss.weight
            losses['reg_loss'] = reg_loss
            loss = loss + reg_loss
        return loss, losses

    def forward_and_loss(self, batch):
        """ Forward pass inference + compute loss """
        outputs = self.inference(batch)
        loss, losses = self.loss_fn(outputs, batch)
        return loss, losses, outputs

    def training_step(self, batch, batch_idx):
        train_loss, train_losses, train_outputs = self.forward_and_loss(batch)
        self._log_train_metrics(batch, train_loss, train_losses, train_outputs)
        self._log_train_visual(batch_idx, batch, train_outputs)
        return train_loss

    def _log_train_metrics(self, batch, train_loss, train_losses, train_outputs):
        self.log('train_loss/total_loss', train_loss)
        self.log_dict({f'train_loss/{k}': loss for k, loss in train_losses.items()})

        # (optional) log training metrics at validation epochs
        if self.hparams.training.log_train_metrics:
            if (self.current_epoch+1) % self.hparams.training.trainer.check_val_every_n_epoch == 0:
                with torch.no_grad():
                    train_metrics = self._measure_metrics(batch, train_outputs['disp'][-1])
                train_metrics = {f'train_metrics/{k}': metric for k, metric in train_metrics.items()}
                self.log_dict(train_metrics, on_step=False, on_epoch=True)  # epoch accumulated

    def _log_train_visual(self, batch_idx, batch, train_outputs):
        # (optional) log training visuals at validation epochs
        if self.hparams.training.log_train_visual:
            if batch_idx == 0 and (self.current_epoch+1) % self.hparams.training.log_visual_every_n_epoch == 0:
                self._log_visual(batch, train_outputs, stage='train')

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            for k, x in batch.items():
                # 2D: cardiac-only (1, N_slices, H, W) -> (N_slices, 1, H, W)
                # 3D: (1, 1, D, H, W)
                batch[k] = x.transpose(0, 1)
            val_loss, val_losses, val_outputs = self.forward_and_loss(batch)
        val_losses = {k: loss.cpu() for k, loss in val_losses.items()}
        val_metrics = self._measure_metrics(batch, val_outputs['disp'][-1])
        self._log_validation_visual(batch_idx, batch, val_outputs)
        return {'total_loss': val_loss.cpu(), **val_losses, **val_metrics}

    def validation_epoch_end(self, val_metrics):
        """ Process and log accumulated validation results in one epoch """
        val_metrics_epoch = merge_dicts(val_metrics)  # default merge by taking mean
        self.log_dict({f'val_metrics/{k}': metric for k, metric in val_metrics_epoch.items()})

        # update hparams metrics
        self.hparam_metrics = {f'hparam_metrics/{k}': val_metrics_epoch[k] for k in self.hparams.hparam_metrics}

    def _log_validation_visual(self, batch_idx, batch, val_outputs):
        if batch_idx == 0 and (self.current_epoch+1) % self.hparams.training.log_visual_every_n_epoch == 0:
            self._log_visual(batch, val_outputs, stage='val')

    def _log_visual(self, batch, outputs, stage='val'):
        vis_data = {k: batch[k].cpu().numpy() for k in ['tar', 'src', 'tar_seg', 'src_seg']}
        disp = outputs['disp'][-1].detach().cpu()
        src_seg = batch['src_seg'].cpu()
        with torch.no_grad():
            vis_data['warped_src_seg'] = warp(src_seg, disp, interp_mode='nearest').numpy()
        vis_data['disp'] = disp.numpy()
        vis_data['warped_src'] = outputs['warped_src'].detach().cpu().numpy()

        fig = visualise_result(vis_data, axis=2)
        self.logger.experiment.add_figure(f'{stage}_visual', fig, global_step=self.global_step, close=True)

    def _measure_metrics(self, batch, disp):
        metric_data = {k: x.detach().cpu() for k, x in batch.items()}
        disp = disp.detach().cpu()
        metric_data['disp'] = disp

        # match size with disp (for multi-resolution)
        for k, x in metric_data.items():
            if x.shape[2:] != disp.shape[2:]:
                metric_data[k] = interpolate_nd(x, size=disp.shape[2:], mode='nearest' if '_seg' in k else None)

        # warp source segmentation and image
        if 'seg_metrics' in self.hparams.metric_groups:
            metric_data['warped_src_seg'] = warp(metric_data['src_seg'], disp, interp_mode='nearest')
        if 'image_metrics' in self.hparams.metric_groups:
            metric_data['tar_pred'] = warp(metric_data['src_ref'], disp)
        metric_data = {k: x.numpy() for k, x in metric_data.items()}
        return measure_metrics(metric_data, self.hparams.metric_groups, spacing=self.val_dataset.spacing)
