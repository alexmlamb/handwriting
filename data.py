import os
import h5py
import numpy as np
import theano


theano.config.floatX = 'float32'
floatX = theano.config.floatX


def load_data(filename='hand_training.hdf5'):
    data_folder = os.path.join(os.environ['DATA_PATH'], 'handwriting')
    training_data_file = os.path.join(data_folder, filename)
    train_data = h5py.File(training_data_file, 'r')

    pt_seq = train_data['pt_seq'][:]
    pt_idx = train_data['pt_idx'][:]
    strings_seq = train_data['str_seq'][:]
    strings_idx = train_data['str_idx'][:]

    train_data.close()
    return pt_seq, pt_idx, strings_seq, strings_idx


def create_generator(shuffle, batch_size, seq_pt, pt_idx,
                     seq_strings, strings_idx, bias_value, chunk=None):
    n_seq = pt_idx.shape[0]

    if shuffle:
        idx = np.arange(n_seq)
        np.random.shuffle(idx)
        pt_idx = pt_idx[idx]
        strings_idx = strings_idx[idx]

    def generator():
        for i in range(0, n_seq-batch_size, batch_size):
            pt, pt_mask, str, str_mask = \
                extract_sequence(slice(i, i + batch_size),
                                 seq_pt, pt_idx, seq_strings, strings_idx)

            pt_input = pt[:-1]
            pt_tg = pt[1:]
            pt_mask = pt_mask[1:]

            n_samples = 50
            n_hidden = 400
            n_chars = 81
            n_mixt_attention = 10

            #TODO: don't hardcode.
            pt_ini_mat = np.zeros((n_samples, 3), floatX)
            h_ini_mat = np.zeros((n_samples, n_hidden), floatX)
            k_ini_mat = np.zeros((n_samples, n_mixt_attention), floatX)
            w_ini_mat = np.zeros((n_samples, n_chars), floatX)
            bias = np.asarray(bias_value).astype('float32')

            if not chunk:
                yield (pt_input, pt_tg, pt_mask, str, str_mask, pt_ini_mat, h_ini_mat, k_ini_mat, w_ini_mat, bias), True
                continue

            l_seq = pt_input.shape[0]
            for j in range(0, l_seq-chunk-1, chunk):
                s = slice(j, j+chunk)
                yield (pt_input[s], pt_tg[s], pt_mask[s], str, str_mask, pt_ini_mat, h_ini_mat, k_ini_mat, w_ini_mat, bias), False
            s = slice(j + chunk, None)


            yield (pt_input[s], pt_tg[s], pt_mask[s], str, str_mask, pt_ini_mat, h_ini_mat, k_ini_mat, w_ini_mat, bias), True

    return generator


def extract_sequence(s, pt, pt_idx, strings, str_idx, M=None):
    """
    the slice s represents the minibatch
    - pt: shape (number points, 3)
    - pt_idx: shape (number of sequences, 2): the starting and end points of
        each sequence
    """
    if not M:
        M = 1500

    pt_idxs = pt_idx[s]
    str_idxs = str_idx[s]

    longuest_pt_seq = max([b - a for a, b in pt_idxs])
    longuest_pt_seq = min(M, longuest_pt_seq)
    pt_batch = np.zeros((longuest_pt_seq, len(pt_idxs), 3), dtype=floatX)
    pt_mask_batch = np.zeros((longuest_pt_seq, len(pt_idxs)), dtype=floatX)

    longuest_str_seq = max([b - a for a, b in str_idxs])
    str_batch = np.zeros((longuest_str_seq, len(str_idxs)), dtype='int32')
    str_mask_batch = np.zeros((longuest_str_seq, len(str_idxs)), dtype=floatX)

    for i, (pt_seq, str_seq) in enumerate(zip(pt_idxs, str_idxs)):
        pts = pt[pt_seq[0]:pt_seq[1]]
        limit2 = min(pts.shape[0], longuest_pt_seq)
        pt_batch[:limit2, i] = pts[:limit2]
        pt_mask_batch[:limit2, i] = 1

        strs = strings[str_seq[0]:str_seq[1]]
        limit2 = min(strs.shape[0], longuest_pt_seq)
        str_batch[:limit2, i] = strs[:limit2]
        str_mask_batch[:limit2, i] = 1

    return pt_batch, pt_mask_batch, str_batch, str_mask_batch


def char2int(ls_str, dict_char2int):

    longuest_str = 0
    for s in ls_str:
        if len(s) > longuest_str:
            longuest_str = len(s)

    res = np.zeros((longuest_str, len(ls_str)), dtype='int32')
    res_mask = np.zeros((longuest_str, len(ls_str)), dtype=floatX)

    for i, s in enumerate(ls_str):
        for j, c in enumerate(s):
            res[j, i] = dict_char2int[ls_str[i][j]]
            res_mask[j, i] = 1

    return res, res_mask

