import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import theano

floatX = theano.config.floatX

M_x = 8.18586
M_y = 0.11457
s_x = 40.3719
s_y = 37.0466


def plot_generated_sequences(pt_batch, other_mats=None, pt_mask=None,
                             show=False, folder_path=None, file_name=None):
    """
    Plot sequences of (x, y, penup) points and other information

    Parameters:
    -----------
    pt_batch: numpy array (seq_length, batch_size, 3)
    other_mats: list of numpy arrays of shapes (seq_length, batch_size)
    pt_mask: numpy array (seq_length, batch_size)
    """
    if not other_mats:
        other_mats = []

    # Renorm batch
    batch_normed = np.zeros_like(pt_batch, dtype=pt_batch.dtype)
    batch_normed[:] = pt_batch
    batch_normed[:, :, 0] = s_x * batch_normed[:, :, 0] + M_x
    batch_normed[:, :, 1] = s_y * batch_normed[:, :, 1] + M_y

    n_samples = batch_normed.shape[1]
    n_mats = len(other_mats)
    fig = plt.figure(figsize=(10*n_samples, (1 + n_mats) * 3))

    for i in range(n_samples):
        mask_term = pt_mask[:, i].astype(bool)
        splot_pt = plt.subplot2grid((1 + n_mats, n_samples), (0, i))
        plot_seq(splot_pt, batch_normed[:, i], mask_term)

        for j, (mat, title) in enumerate(other_mats):
            splot_pt = plt.subplot2grid((1 + n_mats, n_samples), (j+1, i))
            plot_matrix(splot_pt, mat[:, i], mask_term, title)

    if folder_path and file_name:
        fig.savefig(os.path.join(folder_path, file_name + '.png'),
                    bbox_inches='tight', dpi=200)
    if show:
        plt.show()
    plt.close()


def plot_seq_pt(pt_batch, pt_mask_batch=None, use_mask=False, show=False,
                folder_path=None, file_name=None):
    batch_normed = np.zeros_like(pt_batch, dtype=pt_batch.dtype)
    batch_normed[:] = pt_batch
    batch_normed[:, :, 0] = s_x * batch_normed[:, :, 0] + M_x
    batch_normed[:, :, 1] = s_y * batch_normed[:, :, 1] + M_y

    # print 'mean x: {}'.format(batch_normed[:, :, 0].mean())
    # print 'std x: {}'.format(batch_normed[:, :, 0].std())
    # print 'mean y: {}'.format(batch_normed[:, :, 1].mean())
    # print 'std y: {}'.format(batch_normed[:, :, 1].std())

    n_samples = batch_normed.shape[1]
    fig = plt.figure(figsize=(10, n_samples*3))

    for i in range(n_samples):
        mask_term = np.array([])
        if use_mask:
            mask_term = pt_mask_batch[:, i].astype(bool)
        subplot = fig.add_subplot(n_samples, 1, i + 1)
        plot_seq(subplot, batch_normed[:, i], mask_term)

    if folder_path and file_name:
        fig.savefig(os.path.join(folder_path, file_name + '.png'),
                    bbox_inches='tight', dpi=200)
    if show:
        plt.show()
    plt.close()


def plot_seq(subplot, seq_pt, seq_mask=np.array([]), norm=False):
    if norm:
        seq_normed = np.zeros_like(seq_pt, dtype=seq_pt.dtype)
        seq_normed[:] = seq_pt
        seq_normed[:, 0] = s_x * seq_normed[:, 0] + M_x
        seq_normed[:, 1] = s_y * seq_normed[:, 1] + M_y
    else:
        seq_normed = seq_pt

    if seq_mask.size:
        seq_normed = seq_normed[seq_mask.astype(bool)]

    pt = seq_normed[:, 0:2]
    penup = seq_normed[:, 2].astype(bool)

    pos = np.where(penup)[0]+1

    pt = np.cumsum(pt, axis=0)
    pt = np.insert(pt, pos, [np.nan, np.nan], axis=0)

    subplot.plot(pt[:, 0], -pt[:, 1])
    subplot.axis('equal')


def plot_matrix(subplot, mat, seq_mask=np.array([]), title=''):
    if seq_mask.size:
        mat = mat[seq_mask.astype(bool)]

    im = subplot.matshow(mat.T)
    subplot.set_aspect('auto')
    subplot.set_title(title)
    cax = make_axes_locatable(subplot).append_axes("right", size="1%",
                                                   pad=0.05)
    plt.colorbar(im, cax=cax)


def create_train_tag_values(seq_pt, seq_str, seq_tg, seq_pt_mask,
                            seq_str_mask, batch_size):
    f_s_pt = 6
    f_s_str = 7
    seq_pt.tag.test_value = np.zeros((f_s_pt, batch_size, 3), dtype=floatX)
    seq_str.tag.test_value = np.zeros((f_s_str, batch_size), dtype='int32')
    seq_tg.tag.test_value = np.ones((f_s_pt, batch_size, 3), dtype=floatX)
    seq_pt_mask.tag.test_value = np.ones((f_s_pt, batch_size), dtype=floatX)
    seq_str_mask.tag.test_value = np.ones((f_s_str, batch_size), dtype=floatX)


def create_gen_tag_values(model, pt_ini, h_ini_pred, k_ini_pred, w_ini_pred,
                          bias, seq_str, seq_str_mask):
    f_s_str = 7
    n_samples = 3
    n_hidden, n_chars = model.n_hidden, model.n_chars
    n_mixt_attention = model.n_mixt_attention
    pt_ini.tag.test_value = np.zeros((n_samples, 3), floatX)
    h_ini_pred.tag.test_value = np.zeros((n_samples, n_hidden), floatX)
    k_ini_pred.tag.test_value = np.zeros((n_samples, n_mixt_attention), floatX)
    w_ini_pred.tag.test_value = np.zeros((n_samples, n_chars), floatX)
    bias.tag.test_value = np.float32(0.0)
    seq_str.tag.test_value = np.zeros((f_s_str, n_samples), dtype='int32')
    seq_str_mask.tag.test_value = np.ones((f_s_str, n_samples), dtype=floatX)
