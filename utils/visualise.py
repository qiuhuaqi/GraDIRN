import numpy as np
import os
from matplotlib import pyplot as plt


def plot_warped_grid(ax, disp, bg_img=None, interval=3, title="$\mathcal{T}_\phi$", title_font_size='large', color='c'):
    """disp shape (2, H, W)"""
    if bg_img is not None:
        background = bg_img
    else:
        background = np.zeros(disp.shape[1:])  # black background

    id_grid_H, id_grid_W = np.meshgrid(range(0, background.shape[0] - 1, interval),
                                       range(0, background.shape[1] - 1, interval),
                                       indexing='ij')

    new_grid_H = id_grid_H + disp[0, id_grid_H, id_grid_W]
    new_grid_W = id_grid_W + disp[1, id_grid_H, id_grid_W]

    kwargs = {"linewidth": 1.5, "color": color}
    # matplotlib.plot() uses x-y(-z) indexing
    for i in range(new_grid_H.shape[0]):
        ax.plot(new_grid_W[i, :], new_grid_H[i, :], **kwargs)  # each draws a horizontal line
    for i in range(new_grid_H.shape[1]):
        ax.plot(new_grid_W[:, i], new_grid_H[:, i], **kwargs)  # each draws a vertical line

    ax.set_title(title, fontsize=title_font_size)
    ax.imshow(background, cmap='gray')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def plot_result_fig(vis_data_dict, save_path=None, title_font_size=20, dpi=100, show=False, close=False):
    """Plot visual results in a single figure/subplots.
    Images should be shaped (*sizes), disp should be shaped (ndim, *sizes)
    vis_data_dict.keys() = ['tar', 'src', 'src_ref', 'tar_pred', 'warped_src', 'disp_gt', 'disp']
    """
    fig = plt.figure(figsize=(30, 18))
    title_pad = 10

    ax = plt.subplot(2, 4, 1)
    plt.imshow(vis_data_dict["tar"], cmap='gray')
    plt.axis('off')
    ax.set_title('$I_f$', fontsize=title_font_size, pad=title_pad)

    ax = plt.subplot(2, 4, 2)
    plt.imshow(vis_data_dict["src"], cmap='gray')
    plt.axis('off')
    ax.set_title('$I_m$', fontsize=title_font_size, pad=title_pad)

    # calculate the error before and after reg
    error_before = vis_data_dict["tar"] - vis_data_dict["src_ref"]
    error_after = vis_data_dict["tar"] - vis_data_dict["tar_pred"]

    # warped source image
    ax = plt.subplot(2, 4, 3)
    plt.imshow(vis_data_dict["warped_src"], cmap='gray')
    plt.axis('off')
    ax.set_title('$I_m \circ \phi$', fontsize=title_font_size, pad=title_pad)

    # warped grid: prediction
    ax = plt.subplot(2, 4, 4)
    plot_warped_grid(ax, vis_data_dict["disp"], interval=3, title="$\phi_{pred}$", title_font_size=title_font_size)

    # error before
    ax = plt.subplot(2, 4, 5)
    plt.imshow(error_before, vmin=-2, vmax=2, cmap='seismic')  # assuming images are normalised to [0, 1]
    plt.axis('off')
    ax.set_title('$\epsilon$ before', fontsize=title_font_size, pad=title_pad)

    # error after
    ax = plt.subplot(2, 4, 6)
    plt.imshow(error_after, vmin=-2, vmax=2, cmap='seismic')  # assuming images were normalised to [0, 1]
    plt.axis('off')
    ax.set_title('$\epsilon$ after', fontsize=title_font_size, pad=title_pad)

    # target segmentation
    ax = plt.subplot(2, 4, 7)
    plt.imshow(vis_data_dict['tar_seg'])  # assuming images were normalised to [0, 1]
    plt.axis('off')
    ax.set_title('$S_f$', fontsize=title_font_size, pad=title_pad)

    # warped source segmentation
    ax = plt.subplot(2, 4, 8)
    plt.imshow(vis_data_dict['warped_src_seg'])  # assuming images were normalised to [0, 1]
    plt.axis('off')
    ax.set_title('$S_m \circ \phi$', fontsize=title_font_size, pad=title_pad)

    # adjust subplot placements and spacing
    plt.subplots_adjust(left=0.0001, right=0.99, top=0.9, bottom=0.1, wspace=0.001, hspace=0.1)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()
    if close:
        plt.close()
    return fig


def visualise_result(data_dict, axis=0, save_result_dir=None, epoch=None, dpi=50):
    """
    Save one validation visualisation figure for each epoch
    - 2D: the middle-slice from the N-slice stack
    - 3D: the middle slice on the chosen axis

    Args:
        data_dict: (dict, key: ndarray) images shape (N, 1, *sizes), disp shape (N, ndim, *sizes)
        save_result_dir: (string) Path to visualisation result directory
        epoch: (int) Epoch number (for naming when saving)
        axis: (int) For 3D only, choose the 2D plane orthogonal to this axis in 3D volume
        dpi: (int) Image resolution of saved figure
    """
    ndim = data_dict["tar"].ndim - 2
    sizes = data_dict["tar"].shape[2:]

    # put 2D slices into visualisation data dict
    vis_data_dict = {}
    if ndim == 2:
        z = data_dict["tar"].shape[0]//2
        for name, d in data_dict.items():
            vis_data_dict[name] = data_dict[name][z, ...].squeeze()  # (H, W) or (2, H, W)
    else:  # 3D
        # visualise the middle slice of the chosen axis
        z = sizes[axis] // 2
        for name, d in data_dict.items():
            if name in ["disp", "disp_gt"]:
                # choose the two axes/directions to visualise for disp
                axes = [0, 1, 2]
                axes.remove(axis)
                vis_data_dict[name] = d[0, axes, ...].take(z, axis=axis+1)  # (2, X, X)
            else:
                # images
                vis_data_dict[name] = d[0, 0, ...].take(z, axis=axis)  # (X, X)

    # check and fill figure grid data
    if not "disp_gt" in data_dict.keys():
        # housekeeping: dummy dvf_gt for inter-subject case
        vis_data_dict["disp_gt"] = np.zeros_like(vis_data_dict["disp"])

    if 'tar_pred' not in vis_data_dict.keys():
        # 'tar_pred' might be in the dict for multi-modal (e.g. warped T1 of the source subject)
        vis_data_dict['tar_pred'] = vis_data_dict['warped_src']

    if 'src_ref' not in vis_data_dict.keys():
        # 'src_ref' might be in the dict for multi-modal (e.g. T1 of the source subject)
        vis_data_dict['src_ref'] = vis_data_dict['src']

    if save_result_dir is not None:
        fig_save_path = os.path.join(save_result_dir, f'epoch{epoch}_axis_{axis}_slice_{z}.png')
    else:
        fig_save_path = None

    fig = plot_result_fig(vis_data_dict, save_path=fig_save_path, dpi=dpi)
    return fig


def visualise_seq_results(data_dict,
                          axis=2,
                          fig_size=(30, 25),
                          title_pad=0.25,
                          w_pad=0.1,
                          h_pad=0.2,
                          title_font_size='xx-large',
                          dpi=100):
    """Visualisation function for a sequence of results"""
    ndims = data_dict['tars'][0].ndim - 2
    num_blocks = len(data_dict['tars'])
    num_data = len(data_dict.keys())

    # segmentation error range
    error_seg_min = np.min([e.min() for e in data_dict['errors_seg']])
    error_seg_max = np.max([e.max() for e in data_dict['errors_seg']])

    fig, subplots = plt.subplots(num_data, num_blocks)
    fig.set_size_inches(fig_size)

    for n in range(num_blocks):
        if ndims == 2:
            z = data_dict["tars"][n].shape[0] // 2
            tar = data_dict['tars'][n][z, 0, :, :]
            src = data_dict['srcs'][n][z, 0, :, :]
            warped_src = data_dict['warped_srcs'][n][z, 0, :, :]
            tar_seg = data_dict['tar_segs'][n][z, 0, :, :]
            src_seg = data_dict['src_segs'][n][z, 0, :, :]
            warped_src_seg = data_dict['warped_src_segs'][n][z, 0, :, :]
            error = data_dict['errors'][n][z, 0, :, :]
            error_seg = data_dict['errors_seg'][n][z, 0, :, :]
            disp = data_dict['disps'][n][z, :, :, :]  # 2, H, W

        else:  # ndim == 3
            axes = [0, 1, 2]
            axes.remove(axis)  # set the plane to visualise
            z = data_dict['tars'][n].shape[2:][axis] // 2
            tar = np.rot90(data_dict['tars'][n][0, 0, :, :, :].take(z, axis=axis))
            src = np.rot90(data_dict['srcs'][n][0, 0, :, :, :].take(z, axis=axis))
            warped_src = np.rot90(data_dict['warped_srcs'][n][0, 0, :, :, :].take(z, axis=axis))
            tar_seg = np.rot90(data_dict['tar_segs'][n][0, 0, :, :, :].take(z, axis=axis))
            src_seg = np.rot90(data_dict['src_segs'][n][0, 0, :, :, :].take(z, axis=axis))
            warped_src_seg = np.rot90(data_dict['warped_src_segs'][n][0, 0, :, :, :].take(z, axis=axis))
            error = np.rot90(data_dict['errors'][n][0, 0, :, :, :].take(z, axis=axis))
            error_seg = np.rot90(data_dict['errors_seg'][n][0, 0, :, :, :].take(z, axis=axis))
            disp = np.rot90(data_dict['disps'][n][0, axes, :, :, :].take(z, axis=axis + 1), axes=(1, 2))

        subplots[0, n].imshow(tar, cmap='gray')
        subplots[1, n].imshow(src, cmap='gray')
        subplots[2, n].imshow(warped_src, cmap='gray')
        subplots[3, n].imshow(error, vmin=-2, vmax=2, cmap='seismic')
        plot_warped_grid(subplots[4, n], disp, title=None, title_font_size=title_font_size)
        subplots[5, n].imshow(tar_seg)
        subplots[6, n].imshow(src_seg)
        subplots[7, n].imshow(warped_src_seg)
        subplots[8, n].imshow(error_seg, vmin=error_seg_min*2, vmax=error_seg_max*2, cmap='seismic')

        # add titles (block number)
        subplots[0, n].set_title(f'Step {n+1}', fontsize=title_font_size, pad=title_pad)

        # turn off ticks
        for i in range(num_data):
            subplots[i, n].grid(False)
            subplots[i, n].set_xticks([])
            subplots[i, n].set_yticks([])
            subplots[i, n].set_frame_on(False)

    # y labels (type of data displayed)
    subplots[0, 0].set_ylabel('$I_f$', fontdict={'fontsize': title_font_size})
    subplots[1, 0].set_ylabel('$I_m$', fontdict={'fontsize': title_font_size})
    subplots[2, 0].set_ylabel('$I_m \circ \phi$', fontdict={'fontsize': title_font_size})
    subplots[3, 0].set_ylabel('$\epsilon_I$', fontdict={'fontsize': title_font_size})
    subplots[4, 0].set_ylabel('$\phi$', fontdict={'fontsize': title_font_size})
    subplots[5, 0].set_ylabel('$S_f$',  fontdict={'fontsize': title_font_size})
    subplots[6, 0].set_ylabel('$S_m$',  fontdict={'fontsize': title_font_size})
    subplots[7, 0].set_ylabel('$S_m \circ \phi$', fontdict={'fontsize': title_font_size})
    subplots[8, 0].set_ylabel('$\epsilon_S$', fontdict={'fontsize': title_font_size})

    fig.tight_layout(w_pad=w_pad, h_pad=h_pad)
    fig.set_dpi(dpi)
    return fig
