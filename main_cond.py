import os
import cPickle

from lasagne.updates import adam
import numpy as np
import theano
import theano.tensor as T

from raccoon.trainer import Trainer
from raccoon.extensions import TrainMonitor
from raccoon.layers.utils import clip_norm_gradients

from data import create_generator, load_data
from model import ConditionedModel
from extensions import SamplerCond, SamplingFunctionSaver, ValMonitorHandwriting
from utilities import create_train_tag_values, create_gen_tag_values

from Discriminator import Discriminator

floatX = theano.config.floatX = 'float32'
# theano.config.optimizer = 'None'
# theano.config.compute_test_value = 'raise'
# np.random.seed(42)

##########
# CONFIG #
##########
learning_rate = 0.1
n_hidden = 400
n_chars = 81
n_mixt_attention = 10
n_mixt_output = 20
gain = 0.01
batch_size = 50
chunk = None
train_freq_print = 100
valid_freq_print = 1000
sample_strings = ['Sous le pont Mirabeau coule la Seine.']*50
algo = 'adam'  # adam, sgd

dump_path = os.path.join(os.environ.get('TMP_PATH'), 'handwriting',str(np.random.randint(0, 100000000, 1)[0]))

if not os.path.exists(dump_path):
    os.makedirs(dump_path)

########
# DATA #
########
char_dict, inv_char_dict = cPickle.load(open('char_dict.pkl', 'r'))

model = ConditionedModel(gain, n_hidden, n_chars, n_mixt_attention,
                         n_mixt_output)
pt_ini, h_ini_pred, k_ini_pred, w_ini_pred, bias = model.create_sym_init_states()

# All the data is loaded in memory
train_pt_seq, train_pt_idx, train_str_seq, train_str_idx = \
    load_data('hand_training.hdf5')

train_batch_gen = create_generator(
    True, batch_size,
    train_pt_seq, train_pt_idx, train_str_seq, train_str_idx, chunk=chunk)

valid_pt_seq, valid_pt_idx, valid_str_seq, valid_str_idx = \
    load_data('hand_training.hdf5')

valid_batch_gen = create_generator(
    True, batch_size,
    valid_pt_seq, valid_pt_idx, valid_str_seq, valid_str_idx, chunk=chunk)

##################
# MODEL CREATION #
##################
# shape (seq, element_id, features)
seq_pt = T.tensor3('input', floatX)
seq_str = T.matrix('str_input', 'int32')
seq_tg = T.tensor3('tg', floatX)
seq_pt_mask = T.matrix('pt_mask', floatX)
seq_str_mask = T.matrix('str_mask', floatX)
create_train_tag_values(seq_pt, seq_str, seq_tg, seq_pt_mask,
                        seq_str_mask, batch_size)  # for debug


#model = ConditionedModel(gain, n_hidden, n_chars, n_mixt_attention,
#                         n_mixt_output)
# Initial values of the variables that are transmitted through the recursion
h_ini, k_ini, w_ini = model.create_shared_init_states(batch_size)
loss, updates_ini, monitoring, seq_h_tf = model.apply(seq_pt, seq_pt_mask, seq_tg,
                                            seq_str, seq_str_mask,
                                            h_ini, k_ini, w_ini)
loss.name = 'negll'

#seq_h_tf = T.specify_shape(seq_h_tf, (994, 50, 400))


########################
#Teacher Forcing Disc  #
#######################
#Loss is loss.
#generator params is model.params
#

#####################
# SAMPLING FUNCTION #
#####################
#pt_ini, h_ini_pred, k_ini_pred, w_ini_pred, bias = \
#    model.create_sym_init_states()

print "pt_ini shape", pt_ini.shape

create_gen_tag_values(model, pt_ini, h_ini_pred, k_ini_pred, w_ini_pred,
                      bias, seq_str, seq_str_mask)  # for debug

(pt_gen, a_gen, k_gen, p_gen, w_gen, mask_gen, seq_h_sampled), updates_pred = \
    model.prediction(pt_ini, seq_str, seq_str_mask,
                     h_ini_pred, k_ini_pred, w_ini_pred, bias=bias)

#seq_h_sampled = T.specify_shape(seq_h_sampled, (99,99,99))

discriminator = Discriminator(num_hidden = 400, num_features = 400, mb_size = 50, hidden_state_features = T.concatenate([seq_h_sampled, seq_h_tf], axis = 1), target = theano.shared(np.asarray([0] * 50 + [1] * 50).astype('int32')))


#print "compiling train_pf function"

loss_pf = loss + T.mean(seq_h_sampled)
grad_pf = T.grad(loss_pf, model.params)

updates_pf = adam(grad_pf, model.params, 0.001)

#train_pf = theano.function([pt_ini, seq_str, seq_str_mask, h_ini_pred, k_ini_pred, w_ini_pred,bias] + [seq_pt, seq_tg, seq_pt_mask], [loss + 0.0 * T.mean(seq_h_sampled) + 0.0 * T.mean(discriminator.classification)] + monitoring, on_unused_input = 'ignore', updates = updates_pred + updates_pf + updates_ini)

#train_pf = theano.function([seq_pt, seq_tg, seq_pt_mask, seq_str, seq_str_mask, pt_ini, h_ini_pred, k_ini_pred, w_ini_pred, bias], [loss + 0.0 * T.mean(seq_h_sampled) + 0.0 * T.mean(discriminator.classification)] + monitoring, on_unused_input = 'ignore', updates = updates_pred + updates_pf + updates_ini)

#print "compiled train_pf function"

#raise Exception("DONE")

f_sampling = theano.function(
    [pt_ini, seq_str, seq_str_mask, h_ini_pred, k_ini_pred, w_ini_pred,
     bias], [pt_gen, a_gen, k_gen, p_gen, w_gen, mask_gen],
    updates=updates_pred)

########################
# GRADIENT AND UPDATES #
########################


params = model.params

loss_update = loss + T.mean(discriminator.classification)

print "params", params
grads = T.grad(loss_update, params)
grads = clip_norm_gradients(grads)

if algo == 'adam':
    updates_params = adam(grads, params, 0.0003)
elif algo == 'sgd':
    updates_params = []
    for p, g in zip(params, grads):
        updates_params.append((p, p - learning_rate * g))
else:
    raise ValueError('Specified algo does not exist')

updates_all = updates_ini + updates_params + updates_pred

print "type", type(updates_all)

##############
# MONITORING #
##############

#print "compiling train pf"

#train_pf = theano.function([seq_pt, seq_tg, seq_pt_mask, seq_str, seq_str_mask, pt_ini, h_ini_pred, k_ini_pred, w_ini_pred, bias], [loss + 0.0 * T.mean(seq_h_sampled) + 0.0 * T.mean(discriminator.classification)] + monitoring, updates = updates_all)

#print "compiled train pf"

train_monitor = TrainMonitor(
    train_freq_print, [seq_pt, seq_tg, seq_pt_mask, seq_str, seq_str_mask, pt_ini, h_ini_pred, k_ini_pred, w_ini_pred, bias],
    [loss + 0.0 * T.mean(seq_h_sampled) + 0.0 * T.mean(discriminator.classification)] + monitoring, updates_all)

valid_monitor = ValMonitorHandwriting(
    'Validation', valid_freq_print, [seq_pt, seq_tg, seq_pt_mask, seq_str,
                                     seq_str_mask], [loss] + monitoring,
    valid_batch_gen, updates_ini, model, h_ini, k_ini, w_ini, batch_size,
    apply_at_the_start=False)


sampler = SamplerCond('sampler', train_freq_print, dump_path, 'essai',
                      model, f_sampling, sample_strings,
                      dict_char2int=char_dict, bias_value=0.5)

sampling_saver = SamplingFunctionSaver(
    valid_monitor, loss, valid_freq_print, dump_path, 'f_sampling', model,
    f_sampling, char_dict, apply_at_the_start=True)

train_m = Trainer(train_monitor, train_batch_gen,
                  [valid_monitor, sampler, sampling_saver], [])


############
# TRAINING #
############

def custom_process_fun(generator_output):
    inputs, next_seq = generator_output

    res = train_m.process_batch(*inputs)

    if next_seq:
        model.reset_shared_init_states(h_ini, k_ini, w_ini, batch_size)

    return res

model.reset_shared_init_states(h_ini, k_ini, w_ini, batch_size)
train_m.train(custom_process_fun)






