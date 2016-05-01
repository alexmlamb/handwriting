
import theano
import numpy as np

from raccoon import Extension
from raccoon.extensions import Saver, ValMonitor

from data import char2int
from utilities import plot_seq_pt, plot_generated_sequences

floatX = theano.config.floatX

class Sampler(Extension):
    """Extension to pickle objects.

    Only the compute_object method should be overwritten.
    """
    def __init__(self, name_extension, freq, folder_path, file_name,
                 fun_pred, n_hidden, apply_at_the_start=True,
                 apply_at_the_end=True, n_samples=4):
        super(Sampler, self).__init__(name_extension, freq,
                                      apply_at_the_end=apply_at_the_end,
                                      apply_at_the_start=apply_at_the_start)
        self.folder_path = folder_path
        self.file_name = file_name
        self.fun_pred = fun_pred
        self.n_samples = n_samples
        self.n_hidden = n_hidden

    def execute_virtual(self, batch_id):
        sample = self.fun_pred(np.zeros((self.n_samples, 3), floatX),
                               np.zeros((self.n_samples, self.n_hidden), floatX))

        plot_seq_pt(sample,
                    folder_path=self.folder_path,
                    file_name='{}_'.format(batch_id) + self.file_name)

        return ['executed']


class SamplerCond(Extension):
    """Extension to pickle objects.

    Only the compute_object method should be overwritten.
    """
    def __init__(self, name_extension, freq, folder_path, file_name,
                 model, f_sampling, sample_strings, dict_char2int,
                 bias_value=0.5,
                 apply_at_the_start=True, apply_at_the_end=True):
        super(SamplerCond, self).__init__(name_extension, freq,
                                          apply_at_the_end=apply_at_the_end,
                                          apply_at_the_start=apply_at_the_start)
        self.folder_path = folder_path
        self.file_name = file_name

        self.sample_strings = [s + ' ' for s in sample_strings]
        n_samples = len(sample_strings)
        self.dict_char2int = dict_char2int
        self.f_sampling = f_sampling
        self.bias_value = bias_value

        # Initial values
        self.pt_ini_mat = np.zeros((n_samples, 3), floatX)
        self.h_ini_mat = np.zeros((n_samples, model.n_hidden), floatX)
        self.k_ini_mat = np.zeros((n_samples, model.n_mixt_attention), floatX)
        self.w_ini_mat = np.zeros((n_samples, model.n_chars), floatX)

    def execute_virtual(self, batch_id):

        cond, cond_mask = char2int(self.sample_strings, self.dict_char2int)

        #raise Exception("This is where pt_ini is being passed into f_sampling")

        pt_gen, a_gen, k_gen, p_gen, w_gen, mask_gen = self.f_sampling(
                self.pt_ini_mat, cond, cond_mask,
                self.h_ini_mat, self.k_ini_mat, self.w_ini_mat, self.bias_value)

        # plot_seq_pt(pt_gen,
        #            folder_path=self.folder_path,
        #            file_name='{}_'.format(batch_id) + self.file_name)
        p_gen = np.swapaxes(p_gen, 1, 2)
        mats = [(a_gen, 'alpha'), (k_gen, 'kapa'), (p_gen, 'phi'),
                (w_gen, 'omega')]
        #TODO:start plotting again
        #plot_generated_sequences(
        #    pt_gen, mats,
        #    mask_gen, folder_path=self.folder_path,
        #    file_name='{}_'.format(batch_id) + self.file_name)

        return ['executed']


class SamplingFunctionSaver(Saver):
    def __init__(self, monitor, var, freq, folder_path, file_name,
                 model, f_sampling, dict_char2int, **kwargs):
        Saver.__init__(self, 'Sampling function saver', freq, folder_path,
                       file_name, apply_at_the_end=False, **kwargs)

        self.val_monitor = monitor
        # Index of the variable to check in the monitoring extension
        self.var_idx = monitor.output_links[var][0]
        self.best_value = np.inf

        self.model = model
        self.f_sampling = f_sampling
        self.dict_char2int = dict_char2int

    def condition(self, batch_id):
        return True
        # if not self.val_monitor.history:
        #     return False
        # current_value = self.val_monitor.history[-1][self.var_idx]
        # if current_value < self.best_value:
        #     self.best_value = current_value
        #     return True
        # return False

    def compute_object(self):
        return (self.model, self.f_sampling, self.dict_char2int), \
               ['extension executed']

    def finish(self, bath_id):
        return -1, ['not executed at the end']


class ValMonitorHandwriting(ValMonitor):
    """
    Extension to monitor tensor variables and MonitoredQuantity objects on an
    external fuel stream.
    """
    def __init__(self, name_extension, freq, inputs, monitored_variables,
                 stream, updates, model, h_ini, k_ini, w_ini, batch_size,
                 **kwargs):
        ValMonitor.__init__(self, name_extension, freq, inputs,
                            monitored_variables, stream, updates, **kwargs)
        self.stream = stream
        self.model = model

        self.h_ini = h_ini
        self.k_ini = k_ini
        self.w_ini = w_ini
        self.var = [h_ini, k_ini, w_ini]
        self.batch_size = batch_size

    def compute_current_values(self):

        # Save current state
        previous_states = [v.get_value() for v in self.var]

        self.model.reset_shared_init_states(
            self.h_ini, self.k_ini, self.w_ini, self.batch_size)

        c = 0.0
        for inputs, signal in self.stream():
            self.inc_values(*inputs)
            c += 1

            if signal:
                self.model.reset_shared_init_states(
                    self.h_ini, self.k_ini, self.w_ini, self.batch_size)

        # restore states
        for s, v in zip(previous_states, self.var):
            v.set_value(s)

        for i, agg_fun in enumerate(self.agg_fun):
            self.current_values[i] = agg_fun(self.current_values[i], c)
