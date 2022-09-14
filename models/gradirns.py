import torch

import matplotlib.pyplot as plt

from core.transformations import warp, warp_fn
from core.utils import create_img_pyramid
from models.base import LitDLReg
from utils.visualise import visualise_seq_results


class LitGraDIRN(LitDLReg):
    def __init__(self, *args, **kwargs):
        super(LitGraDIRN, self).__init__(*args, **kwargs)

    def inference(self, batch):
        num_resolutions = len(self.hparams.network.config.num_blocks)
        tars = create_img_pyramid(batch['tar'], num_resolutions)
        srcs = create_img_pyramid(batch['src'], num_resolutions)
        out = self.forward(tars, srcs)
        return {'disp': out, 'tars': tars, 'srcs': srcs}

    def loss_fn(self, outputs, batch):
        # only compute loss on the last output
        tar = outputs['tars'][-1]
        src = outputs['srcs'][-1]
        disp = outputs['disp'][-1]
        grid = getattr(self.network, f'grid_lvl{self.network.num_resolutions-1}')
        warped_src = warp_fn(src, disp, grid)

        losses = {}
        # (dis-)similarity loss
        sim_loss = self.sim_loss_fn(tar, warped_src) * self.hparams.loss.sim_loss.weight
        losses['sim_loss'] = sim_loss
        loss = sim_loss
        # regularisation loss
        if self.reg_loss_fn:
            reg_loss = self.reg_loss_fn(disp) * self.hparams.loss.reg_loss.weight
            losses['reg_loss'] = reg_loss
            loss = loss + reg_loss
        return loss, losses

    def _log_train_metrics(self, batch, train_loss, train_losses, train_outputs):
        super(LitGraDIRN, self)._log_train_metrics(batch, train_loss, train_losses, train_outputs)
        tau_list = [sim_block.tau_ for sim_blocks_lvl in self.network.sim_blocks for sim_block in sim_blocks_lvl]
        self.log_dict({f'tau/block_{n}': tau_ for n, tau_ in enumerate(tau_list)})

    def _log_validation_visual(self, batch_idx, batch, val_outputs):
        # (optional) log validation visual
        if batch_idx == 0 and (self.current_epoch+1) % self.hparams.training.log_visual_every_n_epoch == 0:
            self._log_visual(batch, val_outputs, stage='val')

    def _log_visual(self, batch, outputs, stage='val'):
        with torch.no_grad():
            # log images and transformation
            visual_data = {k: list()
                           for k in ['tars', 'srcs', 'warped_srcs', 'tar_segs', 'src_segs',
                                     'warped_src_segs', 'errors', 'errors_seg', 'disps']}

            # generate segmentation pyramids
            tar_segs = create_img_pyramid(batch['tar_seg'].cpu(), lvls=self.network.num_resolutions, label=True)
            src_segs = create_img_pyramid(batch['src_seg'].cpu(), lvls=self.network.num_resolutions, label=True)

            # iterator to iterate through the list of disps
            disps_iter = iter(outputs['disp'])

            for lvl in range(self.network.num_resolutions):
                for _ in range(self.network.num_blocks[lvl] * self.network.num_repeat[lvl]):
                    tar = outputs['tars'][lvl].detach().cpu()
                    src = outputs['srcs'][lvl].detach().cpu()
                    tar_seg = tar_segs[lvl]
                    src_seg = src_segs[lvl]
                    disp = next(disps_iter).detach().cpu()
                    warped_src = warp(src, disp)
                    error = tar - warped_src
                    warped_src_seg = warp(src_seg, disp, interp_mode='nearest')
                    error_seg = tar_seg - warped_src_seg

                    visual_data_n = {'tars': tar, 'srcs': src, 'warped_srcs': warped_src,
                                     'tar_segs': tar_seg, 'src_segs': src_seg, 'warped_src_segs': warped_src_seg,
                                     'errors': error, 'errors_seg': error_seg, 'disps': disp}
                    assert visual_data_n.keys() == visual_data.keys()
                    for k in visual_data.keys():
                        visual_data[k].append(visual_data_n[k].numpy())

        fig = visualise_seq_results(visual_data)
        self.logger.experiment.add_figure(f'{stage}_visual', fig, global_step=self.global_step, close=True)

    def _log_energy(self, batch, stage='val'):
        # log energy
        self.network.compute_energy = True
        # run forward pass again to populate network.energy
        _, _, _ = self.forward_and_loss(batch)
        energy = self.network.get_energy()
        self.network.compute_energy = False
        fig_energy, ax = plt.subplots()
        ax.plot(range(len(energy)), [e.cpu() for e in energy], 'b^--')
        self.logger.experiment.add_figure(f'{stage}_energy', fig_energy, global_step=self.global_step, close=True)
