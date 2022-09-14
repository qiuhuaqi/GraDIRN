import torch
import torch.nn as nn

from core.sim_terms import SSDSimilarity, NCCSimilarity
from core.reg_terms import CNNRegulariser
from core.transformations import warp_fn
from core.utils import interpolate_nd, convNd


class UNet(nn.Module):
    r"""
    Adapted from the U-net used in VoxelMorph:
    https://github.com/voxelmorph/voxelmorph/blob/master/pytorch/model.py
    """
    def __init__(self,
                 ndim,
                 enc_channels=(16, 32, 32, 32, 32),
                 dec_channels=(32, 32, 32, 32),
                 out_channels=(16, 16),
                 conv_before_out=True
                 ):
        super(UNet, self).__init__()

        self.ndim = ndim

        # encoder layers
        self.enc = nn.ModuleList()
        for i in range(len(enc_channels)):
            in_ch = 2 if i == 0 else enc_channels[i - 1]
            stride = 1 if i == 0 else 2
            self.enc.append(
                nn.Sequential(
                    convNd(ndim, in_ch, enc_channels[i], stride=stride, a=0.2),
                    nn.LeakyReLU(0.2)
                )
            )

        # decoder layers
        self.dec = nn.ModuleList()
        for i in range(len(dec_channels)):
            in_ch = enc_channels[-1] if i == 0 else dec_channels[i-1] + enc_channels[-i-1]
            self.dec.append(
                nn.Sequential(
                    convNd(ndim, in_ch, dec_channels[i], a=0.2),
                    nn.LeakyReLU(0.2)
                )
            )

        # upsampler
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # (optional) conv layers before prediction
        if conv_before_out:
            self.out_layers = nn.ModuleList()
            for i in range(len(out_channels)):
                in_ch = dec_channels[-1] + enc_channels[0] if i == 0 else out_channels[i-1]
                self.out_layers.append(
                    nn.Sequential(
                        convNd(ndim, in_ch, out_channels[i], a=0.2),  # stride=1
                        nn.LeakyReLU(0.2)
                    )
                )

            # final prediction layer with additional conv layers
            self.out_layers.append(
                convNd(ndim, out_channels[-1], ndim)
            )

        else:

            # final prediction layer without additional conv layers
            self.out_layers = nn.ModuleList()
            self.out_layers.append(
                convNd(ndim, dec_channels[-1] + enc_channels[0], ndim)
            )

    def forward(self, tar, src):
        x = torch.cat((tar, src), dim=1)

        # encoder
        fm_enc = [x]
        for enc in self.enc:
            fm_enc.append(enc(fm_enc[-1]))

        # decoder: conv + upsample + concatenate skip-connections (to full resolution)
        dec_out = fm_enc[-1]
        for i, dec in enumerate(self.dec):
            dec_out = dec(dec_out)
            dec_out = self.upsample(dec_out)
            dec_out = torch.cat([dec_out, fm_enc[-2-i]], dim=1)

        # further convs and prediction
        y = dec_out
        for out_layer in self.out_layers:
            y = out_layer(y)
        return [y]


class GraDIRN(nn.Module):
    r"""
    Gradient Descent Based Image Registration Network

    `num_blocks` is a list of number of blocks in each resolution.
    The similarity and regulariser blocks are each a nn.ModuleList of nn.ModuleList(s),
    each list for one resolution:
    `[
    [block1, block2, ..., block_<num_blocks[0]>],
    [block1, block2, ..., block_<num_blocks[1]>],
    ...]`
    """
    def __init__(self,
                 ndim,
                 size,
                 init_mode='identity',
                 num_blocks=(3, 3, 3),
                 num_repeat=(1, 1, 1),
                 scale_step_size=True,
                 tau_config=None,
                 similarity='ssd',
                 similarity_config=None,
                 regulariser_config=None,
                 ):
        super(GraDIRN, self).__init__()

        self.ndim = ndim
        self.init_mode = init_mode
        self.num_blocks = num_blocks  # number of blocks per resolution
        self.num_repeat = num_repeat  # number of times blocks are repeated per resolution
        self.num_resolutions = len(num_blocks)

        self.tau_config = tau_config
        self.similarity_config = similarity_config
        self.regulariser_config = regulariser_config

        if scale_step_size:
            # scale the initial tau with 1/(number_or_blocks)
            step_size_scale = 1 / (sum([a*b for a, b in zip(num_blocks, num_repeat)]) + 1)
            self.tau_config.init = step_size_scale * self.tau_config.init

        # configure similarity term and regulariser term
        self.SIMILARITY = {'ssd': SSDSimilarity, 'ncc': NCCSimilarity}[similarity]
        self.REGULARISER = CNNRegulariser
        self.reg_input_images = self.regulariser_config.input_images

        # main blocks
        self.sim_blocks = nn.ModuleList()
        self.reg_blocks = nn.ModuleList()
        for nb in self.num_blocks:
            self.sim_blocks.append(nn.ModuleList([self.SIMILARITY(self.tau_config) for _ in range(nb)]))
            self.reg_blocks.append(nn.ModuleList([self.REGULARISER(self.regulariser_config) for _ in range(nb)]))

        # initialise grids
        for lvl in range(self.num_resolutions):
            grid = self.get_norm_grid([s // 2**lvl for s in size])
            self.register_buffer(f'grid_lvl{self.num_resolutions - lvl - 1}', grid)

    @staticmethod
    def get_norm_grid(size):
        grid = torch.meshgrid([torch.linspace(-1, 1, s) for s in size])
        grid = torch.stack(grid, 0).requires_grad_(False)  # (ndims, *size)
        return grid

    def forward(self, tars: list, srcs: list):
        """" Input `tars` and `srcs` are list of images with increasing resolution """
        # initialise disp
        device = tars[0].device
        disp = torch.zeros(tars[0].shape[0], self.ndim, *tars[0].shape[2:], device=device)

        disps = []

        for lvl in range(self.num_resolutions):
            for sim_block, reg_block in zip(self.sim_blocks[lvl], self.reg_blocks[lvl]):
                for _ in range(self.num_repeat[lvl]):  # repeating blocks
                    tar, src = tars[lvl], srcs[lvl]

                    # similarity step
                    # locally reverse Lightning's torch.set_grad_enabled(False) context during validation
                    with torch.set_grad_enabled(True):
                        disp.requires_grad_()
                        warped_src = warp_fn(src, disp, getattr(self, f'grid_lvl{lvl}'), backwards_grad=True)
                        sim_step = sim_block(tar, warped_src, disp)

                    # regulariser step
                    if self.reg_input_images == 'explicit':
                        reg_input = torch.cat((disp, tar, warped_src), dim=1)
                    elif self.reg_input_images == 'implicit':
                        reg_input = torch.cat((disp, tar, src), dim=1)
                    else:
                        reg_input = disp
                    reg_step = reg_block(reg_input)

                    # taking the optimisation steps
                    disp = disp - sim_step - reg_step
                    disps.append(disp)

            if lvl < self.num_resolutions - 1:
                disp = interpolate_nd(disp, scale_factor=2.) * 2.
        return disps
