import os
import sys

from lasagne.updates import adam
import numpy as np
import theano
import theano.tensor as T

from raccoon.trainer import Trainer
from raccoon.extensions import TrainMonitor
from raccoon.archi.utils import clip_norm_gradients

from data import create_generator, load_data, extract_sequence
from model import UnconditionedModel
from extensions import Sampler

from utilities import plot_seq, plot_seq_pt

floatX = theano.config.floatX = 'float32'
# theano.config.optimizer = 'None'
# np.random.seed(42)

# CONFIG
learning_rate = 0.1
n_hidden = 400
n_mixtures = 20
gain = 0.01
batch_size = 100
chunk = None
every = 1000
tmp_path = os.environ.get('TMP_PATH')
dump_path = os.path.join(tmp_path, 'handwriting',
                         str(np.random.randint(0, 100000000, 1)[0]))
if not os.path.exists(dump_path):
    os.makedirs(dump_path)

# DATA
tr_coord_seq, tr_coord_idx, tr_strings_seq, tr_strings_idx = \
    load_data('hand_training.hdf5')
# pt_batch, pt_mask_batch, str_batch = \
#     extract_sequence(slice(0, 4),
#                    tr_coord_seq, tr_coord_idx, tr_strings_seq, tr_strings_idx)
# plot_seq_pt(pt_batch, pt_mask_batch, use_mask=True, show=True)
batch_gen = create_generator(
        True, batch_size,
        tr_coord_seq, tr_coord_idx,
        tr_strings_seq, tr_strings_idx, chunk=chunk)


# MODEL CREATION
# shape (seq, element_id, features)
seq_coord = T.tensor3('input', floatX)
seq_tg = T.tensor3('tg', floatX)
seq_mask = T.matrix('mask', floatX)
h_ini = theano.shared(np.zeros((batch_size, n_hidden), floatX), 'hidden_state')

model = UnconditionedModel(gain, n_hidden, n_mixtures)
loss, updates, monitoring = model.apply(seq_coord, seq_mask, seq_tg, h_ini)
loss.name = 'negll'


# GRADIENT AND UPDATES
params = model.params
grads = T.grad(loss, params)
grads = clip_norm_gradients(grads)

# updates_params = adam(grads, params, 0.0003)

updates_params = []
for p, g in zip(params, grads):
    updates_params.append((p, p - learning_rate * g))

updates_all = updates + updates_params

coord_ini = T.matrix('coord', floatX)
h_ini_pred = T.matrix('h_ini_pred', floatX)
gen_coord, updates_pred = model.prediction(coord_ini, h_ini_pred)
f_sampling = theano.function([coord_ini, h_ini_pred], gen_coord,
                             updates=updates_pred)

# MONITORING
train_monitor = TrainMonitor(every, [seq_coord, seq_tg, seq_mask],
                             [loss] + monitoring, updates_all)

sampler = Sampler('sampler', every, dump_path, 'essai',
                  f_sampling, n_hidden)

train_m = Trainer(train_monitor, [sampler], [])


it = 0
epoch = 0
h_ini.set_value(np.zeros((batch_size, n_hidden), dtype=floatX))
try:
    while True:
        epoch += 1
        for (pt_in, pt_tg, pt_mask, str, str_mask), next_seq in batch_gen():
            res = train_m.process_batch(epoch, it,
                                        pt_in, pt_tg,
                                        pt_mask)
            if next_seq:
                h_ini.set_value(np.zeros((batch_size, n_hidden), dtype=floatX))
            it += 1
            if res:
                train_m.finish(it)
                sys.exit()
except KeyboardInterrupt:
    print 'Training interrupted by user.'
    train_m.finish(it)
