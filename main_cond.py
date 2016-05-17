import os
import cPickle

from lasagne.updates import adam
import numpy as np
import theano
import theano.tensor as T

from consider_constant import consider_constant

import shutil

from raccoon.trainer import Trainer
from raccoon.extensions import TrainMonitor
from raccoon.layers.utils import clip_norm_gradients

from data import create_generator, load_data
from model import ConditionedModel
from extensions import SamplerCond, SamplingFunctionSaver, ValMonitorHandwriting
from utilities import create_train_tag_values, create_gen_tag_values

import cPickle as pickle

import sys

sys.setrecursionlimit(1000000)

from Discriminator import Discriminator

theano.config.scan.allow_gc = True
floatX = theano.config.floatX = 'float32'
# theano.config.optimizer = 'None'
# theano.config.compute_test_value = 'raise'
# np.random.seed(42)

##########
# CONFIG #
##########
learning_rate = 0.1
generator_lr = 0.0

print "generator lr", generator_lr

n_hidden = 400
n_chars = 81
n_mixt_attention = 10
n_mixt_output = 20
gain = 0.01
batch_size = 50
chunk = None
#freqs were 100, 1000, 10 ,200
train_freq_print = 20
valid_freq_print = 40
sample_strings = ['I am Alex Lamb.  I love the neural nets'] * 50#['Sous le pont Mirabeau coule la Seine.']*50
algo = 'adam'  # adam, sgd

#model_file_load = "/u/lambalex/models/handwriting/handwriting/71535347/saved_model.pkl"
#model_file_load = "/u/lambalex/models/handwriting/handwriting/81356894/saved_model.pkl"
#model_file_load = "/u/lambalex/models/handwriting/handwriting/10406114/saved_model.pkl"
model_file_load = "saved_model.pkl"

exp_id = np.random.randint(0, 100000000, 1)[0]

dump_path = os.path.join(os.environ.get('TMP_PATH'), 'handwriting',str(exp_id))

os.makedirs(dump_path)

os.mkdir(dump_path + "/src")
shutil.copyfile("main_cond.py", dump_path + "/src/main_cond.py")
shutil.copyfile("Discriminator.py", dump_path + "/src/Discriminator.py")

model_file_save = dump_path + "/saved_model.pkl"

print "DUMP PATH", dump_path

########
# DATA #
########
char_dict, inv_char_dict = cPickle.load(open('char_dict.pkl', 'r'))


if model_file_load is None:
    model = ConditionedModel(gain, n_hidden, n_chars, n_mixt_attention,
                         n_mixt_output)
else:
    print "Loading model file", model_file_load
    fh = open(model_file_load, "r")
    model = pickle.load(fh)
    fh.close()

pt_ini, h_ini_pred, k_ini_pred, w_ini_pred, bias = model.create_sym_init_states()

# All the data is loaded in memory
train_pt_seq, train_pt_idx, train_str_seq, train_str_idx = \
    load_data('hand_training.hdf5')


train_batch_gen = create_generator(
    True, batch_size,
    train_pt_seq, train_pt_idx, train_str_seq, train_str_idx, bias_value = 0.0, chunk=chunk)

valid_pt_seq, valid_pt_idx, valid_str_seq, valid_str_idx = \
    load_data('hand_training.hdf5')

valid_batch_gen = create_generator(
    True, batch_size,
    valid_pt_seq, valid_pt_idx, valid_str_seq, valid_str_idx, bias_value = 0.0, chunk=chunk)

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


discriminator = Discriminator(num_hidden = 800,
                              num_features = 400,
                              mb_size = batch_size,
                              hidden_state_features = T.concatenate([seq_h_sampled[:seq_h_tf.shape[0]], consider_constant(seq_h_tf[:seq_h_sampled.shape[0]])], axis = 1),
                              target = theano.shared(np.asarray([1] * batch_size + [0] * batch_size).astype('int32')))



f_sampling = theano.function([pt_ini, seq_str, seq_str_mask, h_ini_pred, k_ini_pred, w_ini_pred,bias],
                             [pt_gen, a_gen, k_gen, p_gen, w_gen, mask_gen],
                             updates=updates_pred)

########################
# GRADIENT AND UPDATES #
########################


params = model.params

loss_update = loss + generator_lr * discriminator.g_cost

print "params", params
grads = T.grad(loss_update, params)
grads = clip_norm_gradients(grads)

grads_disc = T.grad(discriminator.d_cost, discriminator.params)
grads_disc = clip_norm_gradients(grads_disc)

updates_disc = adam(grads_disc, discriminator.params, 0.0001, beta1 = 0.5)

if algo == 'adam':
    updates_params = adam(grads, params, 0.0003)
elif algo == 'sgd':
    updates_params = []
    for p, g in zip(params, grads):
        updates_params.append((p, p - learning_rate * g))
else:
    raise ValueError('Specified algo does not exist')

updates_all = updates_ini + updates_params + updates_pred + updates_disc

print "type", type(updates_all)

##############
# MONITORING #
##############

#print "compiling train pf"

#train_pf = theano.function([seq_pt, seq_tg, seq_pt_mask, seq_str, seq_str_mask, pt_ini, h_ini_pred, k_ini_pred, w_ini_pred, bias], [loss + 0.0 * T.mean(seq_h_sampled) + 0.0 * T.mean(discriminator.classification)] + monitoring, updates = updates_all)

#print "compiled train pf"

sampled_h_len = seq_h_sampled.shape[0]
sampled_h_len.name = "sampled sequence length"

tf_h_len = seq_h_tf.shape[0]
tf_h_len.name = "tf sequence length"

classification_real = T.mean(discriminator.classification[:batch_size])
classification_real.name = "p(real)"

classification_fake = T.mean(discriminator.classification[batch_size:])
classification_fake.name = "p(fake)"

len_c_real = discriminator.classification[batch_size:].shape[0]
len_c_real.name = "len_c_real"

len_c_fake = discriminator.classification[0:batch_size].shape[0]
len_c_fake.name = "len_c_fake"

print "EXPERIMENT ID", exp_id
exp_id_shared = theano.shared(np.asarray(exp_id).astype('int32'))
exp_id_shared.name = "experiment id"

generator_lr_shared = theano.shared(generator_lr)
generator_lr_shared.name = "generator lr"

mean_diff = ((seq_h_sampled[-1].mean(0) - seq_h_tf[-1].mean(0))**2).sum()
mean_diff.name = "Sampled-TF Mean Diff"

std_diff = ((seq_h_sampled[-1].std(0) - seq_h_tf[-1].std(0))**2).sum()
std_diff.name = "Sampled-TF Var Diff"

#[sampled_h_len, tf_h_len, classification_real, classification_fake, len_c_real, len_c_fake, exp_id_shared, generator_lr_shared, mean_diff, std_diff]

train_monitor = TrainMonitor(
    train_freq_print, [seq_pt, seq_tg, seq_pt_mask, seq_str, seq_str_mask, pt_ini, h_ini_pred, k_ini_pred, w_ini_pred, bias],
    [loss] + [sampled_h_len, tf_h_len, classification_real, classification_fake, len_c_real, len_c_fake, exp_id_shared, generator_lr_shared, mean_diff, std_diff] + monitoring, updates_all)

    #use updates_all

valid_monitor = ValMonitorHandwriting(
    'Validation', valid_freq_print, [seq_pt, seq_tg, seq_pt_mask, seq_str, seq_str_mask, pt_ini, h_ini_pred, k_ini_pred, w_ini_pred, bias], [loss] + monitoring,
    valid_batch_gen, updates_ini, model, h_ini, k_ini, w_ini, batch_size,
    apply_at_the_start=False)

#valid_monitor = ValMonitorHandwriting(
#    'Validation', valid_freq_print, [seq_pt, seq_tg, seq_pt_mask, seq_str,
#                                     seq_str_mask], [loss] + monitoring,
#    valid_batch_gen, updates_ini, model, h_ini, k_ini, w_ini, batch_size,
#    apply_at_the_start=False)


sampler = SamplerCond('sampler', train_freq_print, dump_path, 'essai_halfbias',
                      model, f_sampling, sample_strings,
                      dict_char2int=char_dict, bias_value=0.5)

sampler_0bias = SamplerCond('sampler', train_freq_print, dump_path, 'essai_0bias',
                      model, f_sampling, sample_strings,
                      dict_char2int=char_dict, bias_value=0.0)

sampler_2bias = SamplerCond('sampler', train_freq_print, dump_path, 'essai_2bias',
                      model, f_sampling, sample_strings,
                      dict_char2int=char_dict, bias_value=2.0)

sampling_saver = SamplingFunctionSaver(
    valid_monitor, loss, valid_freq_print, dump_path, 'f_sampling', model,
    f_sampling, char_dict, apply_at_the_start=True)

train_m = Trainer(train_monitor, train_batch_gen,
                  [valid_monitor, sampling_saver, sampler, sampler_0bias, sampler_2bias], [], num_iterations = 40)

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

while True:
    #model.reset_shared_init_states(h_ini, k_ini, w_ini, batch_size)
    train_m.train(custom_process_fun)
    print "saving training function"
    print dump_path
    fh = open(model_file_save, "w")
    pickle.dump(model, fh)
    fh.close()



