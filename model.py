import theano
from theano import shared
import theano.tensor as T
#from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy as np
from lasagne.init import GlorotNormal

from raccoon.layers.attention import PositionAttentionLayer
from raccoon.layers.reccurrent import GRULayer
from raccoon.layers.utils import create_uneven_weight

theano.config.floatX = 'float32'
floatX = theano.config.floatX


def logsumexp(x, axis=None):
    """
    Efficient log of a sum of exponentials
    """
    x_max = T.max(x, axis=axis, keepdims=True)
    z = T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    return z.sum(axis=axis)


class MixtureGaussians2D:
    def __init__(self, ls_n_in, n_mixtures, initializer, eps=1e-5):
        if not isinstance(ls_n_in, (tuple, list)):
            ls_n_in = [ls_n_in]

        self.n_in = sum(ls_n_in)
        self.n_mixtures = n_mixtures
        self.eps = eps

        self.n_out = (n_mixtures +  # proportions
                      n_mixtures * 2 +  # means
                      n_mixtures * 2 +  # stds
                      n_mixtures +  # correlations
                      1)  # bernoulli

        w_in_mat = create_uneven_weight(ls_n_in, self.n_out, initializer)
        self.w = shared(w_in_mat, 'w_mixt')
        self.b = shared(np.random.normal(
            0, 0.001, size=(self.n_out,)).astype(floatX), 'b_mixt')

        self.params = [self.w, self.b]

    def compute_parameters(self, h, bias):
        """
        h: (batch or batch*seq, features)
        """
        n = self.n_mixtures
        out = T.dot(h, self.w) + self.b
        prop = T.nnet.softmax(out[:, :n]*(1 + bias))
        mean_x = out[:, n:n*2]
        mean_y = out[:, n*2:n*3]
        std_x = T.exp(out[:, n*3:n*4] - bias) + self.eps
        std_y = T.exp(out[:, n*4:n*5] - bias) + self.eps
        rho = T.tanh(out[:, n*5:n*6])
        rho = (1+rho + self.eps) / (2 + 2*self.eps) - 1
        bernoulli = T.nnet.sigmoid(out[:, -1])
        bernoulli = (bernoulli + self.eps) / (1 + 2*self.eps)

        return prop, mean_x, mean_y, std_x, std_y, rho, bernoulli

    def prediction(self, h, bias):
        srng = RandomStreams(seed=42)

        prop, mean_x, mean_y, std_x, std_y, rho, bernoulli = \
            self.compute_parameters(h, bias)

        mode = T.argmax(srng.multinomial(pvals=prop, dtype=prop.dtype), axis=1)

        v = T.arange(0, mean_x.shape[0])
        m_x = mean_x[v, mode]
        m_y = mean_y[v, mode]
        s_x = std_x[v, mode]
        s_y = std_y[v, mode]
        r = rho[v, mode]
        # cov = r * (s_x * s_y)

        normal = srng.normal((h.shape[0], 2))
        x = normal[:, 0]
        y = normal[:, 1]

        # x_n = T.shape_padright(s_x * x + cov * y + m_x)
        # y_n = T.shape_padright(s_y * y + cov * x + m_y)

        x_n = T.shape_padright(m_x + s_x * x)
        y_n = T.shape_padright(m_y + s_y * (x * r + y * T.sqrt(1.-r**2)))

        uniform = srng.uniform((h.shape[0],))
        pin = T.shape_padright(T.cast(bernoulli > uniform, floatX))

        return T.concatenate([x_n, y_n, pin], axis=1)

    def apply(self, h_seq, mask_seq, tg_seq):
        """
        h_seq: (seq, batch, features)
        mask_seq: (seq, batch)
        tg_seq: (seq, batch, features=3)
        """
        h_seq = T.reshape(h_seq, (-1, h_seq.shape[-1]))
        tg_seq = T.reshape(tg_seq, (-1, tg_seq.shape[-1]))
        mask_seq = T.reshape(mask_seq, (-1,))

        prop, mean_x, mean_y, std_x, std_y, rho, bernoulli = \
            self.compute_parameters(h_seq, .0)

        tg_x = T.addbroadcast(tg_seq[:, 0:1], 1)
        tg_y = T.addbroadcast(tg_seq[:, 1:2], 1)
        tg_pin = tg_seq[:, 2]

        tg_x_s = (tg_x - mean_x) / std_x
        tg_y_s = (tg_y - mean_y) / std_y

        z = tg_x_s**2 + tg_y_s**2 - 2*rho*tg_x_s*tg_y_s
        buff = 1-rho**2

        tmp = (-z / (2 * buff) -
               T.log(2*np.pi) - T.log(std_x) - T.log(std_y) - 0.5*T.log(buff) +
               T.log(prop))

        c = (-logsumexp(tmp, axis=1) -
             tg_pin * T.log(bernoulli) -
             (1-tg_pin) * T.log(1 - bernoulli))

        c = T.sum(c * mask_seq) / T.sum(mask_seq)

        max_prop = T.argmax(prop, axis=1).mean()
        max_prop.name = 'max_prop'

        std_max_prop = T.argmax(prop, axis=1).std()
        std_max_prop.name = 'std_max_prop'

        return c, [max_prop, std_max_prop]


class UnconditionedModel:
    def __init__(self, gain_ini, n_hidden, n_mixtures):
        ini = GlorotNormal(gain_ini)

        self.gru_layer = GRULayer(3, n_hidden, ini)

        self.mixture = MixtureGaussians2D(n_hidden, n_mixtures, ini)

        self.params = self.gru_layer.params + self.mixture.params

    def apply(self, seq_pt, seq_mask, seq_tg, h_ini):

        seq_h, scan_updates = self.gru_layer.apply(seq_pt, seq_mask, h_ini)
        loss, monitoring = self.mixture.apply(seq_h, seq_mask, seq_tg)

        h_mean = seq_h.mean()
        h_mean.name = 'h_mean'
        monitoring.append(h_mean)

        return loss, [(h_ini, seq_h[-1])] + scan_updates, monitoring

    def prediction(self, pt_ini, h_ini, bias=.0, n_steps=1000):

        def gru_step(pt_pre, h_pre):

            h = self.gru_layer.step(pt_pre, h_pre,
                                    mask=None, process_inputs=True)

            pt = self.mixture.prediction(h, bias)

            return pt, h

        res, scan_updates = theano.scan(
            fn=gru_step,
            outputs_info=[pt_ini, h_ini],
            n_steps=n_steps)

        return res[0], scan_updates


class ConditionedModel:
    def __init__(self, gain_ini, n_hidden, n_chars, n_mixt_attention,
                 n_mixtures):
        """
        Parameters
        ----------
        n_mixt_attention: int
            Number of mixtures used by the attention mechanism
        n_chars: int
            Number of different characters
        n_mixtures: int
            Number of mixtures in the Gaussian Mixture model
        """
        self.n_hidden = n_hidden
        self.n_chars = n_chars
        self.n_mixt_attention = n_mixt_attention
        self.n_mixtures = n_mixtures

        ini = GlorotNormal(gain_ini)

        self.pos_layer = PositionAttentionLayer(
            GRULayer([3, self.n_chars], n_hidden, ini),
            self.n_chars,
            self.n_mixt_attention, ini)

        self.mixture = MixtureGaussians2D([n_hidden, self.n_chars],
                                          n_mixtures, ini)

        self.params = self.pos_layer.params + self.mixture.params


    def apply(self, seq_pt, seq_mask, seq_tg, seq_str, seq_str_mask,
              h_ini, k_ini, w_ini):
        """
        Parameters
        ----------
        seq_pt: (length_pt_seq, batch_size, 3)
        seq_mask: (length_pt_seq, batch_size)
        seq_tg: (length_pt_seq, batch_size, 3)
        seq_str: (length_str_seq, batch_size)
            Each character is represented by an integer
        seq_str_mask: (length_str_seq, batch_size)

        h_ini: (batch_size, n_hidden)
        k_ini: (batch_size, n_mixture_attention)
        w_ini: (batch_size, n_chars)
        """

        # Convert the integers representing chars into one-hot encodings
        # seq_str will have shape (seq_length, batch_size, n_chars)
        seq_str = T.eye(self.n_chars, dtype=floatX)[seq_str]

        (seq_h, seq_k, seq_w), scan_updates = self.pos_layer.apply(
            seq_pt, seq_mask, seq_str, seq_str_mask,
            h_ini, k_ini, w_ini)

        #seq_h = T.specify_shape(seq_h, (99,99,99))

        seq_h_conc = T.concatenate([seq_h, seq_w], axis=-1)

        loss, monitoring = self.mixture.apply(seq_h_conc, seq_mask, seq_tg)

        updates = [(h_ini, seq_h[-1]), (k_ini, seq_k[-1]), (w_ini, seq_w[-1])]

        # Monitoring variables
        monitoring.extend(
            self.create_monitoring_variables(seq_h, seq_k, seq_w, seq_mask))

        return loss, updates + scan_updates, monitoring, seq_h

    def prediction(self, pt_ini, seq_str, seq_str_mask,
                   h_ini, k_ini, w_ini, bias=.0, n_steps=1000):
        """
        Parameters
        ----------
        pt_ini: (batch_size, 3)
        seq_str: (length_str_seq, batch_size)
        seq_str_mask: (length_str_seq, batch_size)

        h_ini: (batch_size, n_hidden)
        k_ini: (batch_size, n_mixture_attention)
        w_ini: (batch_size, n_chars)

        bias: float
            The bias that controls the variance of the generation
        n_steps: int
            The maximal number of generation steps.
        """

        # Convert the integers representing chars into one-hot encodings
        # seq_str will have shape (seq_length, batch_size, n_chars)
        seq_str = T.eye(self.n_chars, dtype=floatX)[seq_str]
        #TODO: don't hardcode batch size
        batch_size = 50 #pt_ini.shape[0]

        def scan_step(pt_pre, h_pre, k_pre, w_pre, mask,
                      seq_str, seq_str_mask, bias):

            h, a, k, p, w = self.pos_layer.step(
                pt_pre, h_pre, k_pre, w_pre,
                seq_str, seq_str_mask, mask=mask)

            h_conc = T.concatenate([h, w], axis=-1)

            pt = self.mixture.prediction(h_conc, bias)

            # ending condition
            last_char = T.cast(T.sum(seq_str_mask, axis=0)-1, 'int32')
            last_phi = p[last_char, T.arange(last_char.shape[0])]
            max_phi = T.max(p, axis=0)
            condition = last_phi >= 0.95*max_phi
            mask = T.switch(condition, .0, mask)

            return ((pt, h, a, k, p, w, mask),
                    theano.scan_module.until(T.all(mask < 1.)))

        (seq_pt, seq_h, seq_a, seq_k, seq_p, seq_w, seq_mask), scan_updates = \
            theano.scan(
                fn=scan_step,
                outputs_info=[pt_ini, h_ini, None, k_ini, None, w_ini,
                              T.alloc(1., batch_size)],
                non_sequences=[seq_str, seq_str_mask, bias],
                n_steps=n_steps)

        return (seq_pt, seq_a, seq_k, seq_p, seq_w, seq_mask, seq_h), scan_updates

    def create_monitoring_variables(self, seq_h, seq_k, seq_w, seq_mask):
        """
        seq_h: (length_pt_seq, batch_size, n_hidden)
        seq_k: (length_pt_seq, batch_size, n_mixt)
        """

        seq_h = seq_h * seq_mask[:, :, None]
        seq_k = seq_k * seq_mask[:, :, None]
        seq_w = seq_w * seq_mask[:, :, None]

        n = seq_mask[:, :, None].sum()

        seq_h_mean = T.sum(seq_h.mean(axis=-1)) / n
        seq_h_mean.name = 'seq_h_mean'

        seq_k_mean = T.sum(seq_k.mean(axis=-1)) / n
        seq_k_mean.name = 'seq_k_mean'

        seq_w_mean = T.sum(seq_w.mean(axis=-1)) / n
        seq_w_mean.name = 'seq_w_mean'

        return [seq_h_mean, seq_k_mean, seq_w_mean]

    def create_shared_init_states(self, batch_size):
        def create_shared(size, name):
            return theano.shared(np.zeros(size, floatX), name)

        h_ini = create_shared((batch_size, self.n_hidden), 'h_ini')
        k_ini = create_shared((batch_size, self.n_mixt_attention), 'k_ini')
        w_ini = create_shared((batch_size, self.n_chars), 'w_ini')

        return h_ini, k_ini, w_ini

    def reset_shared_init_states(self, h_ini, k_ini, w_ini, batch_size):
        def set_value(var, size):
            var.set_value(np.zeros(size, dtype=floatX))

        set_value(h_ini, (batch_size, self.n_hidden))
        set_value(k_ini, (batch_size, self.n_mixt_attention))
        set_value(w_ini, (batch_size, self.n_chars))

    def create_sym_init_states(self):
        pt_ini = T.matrix('pt_pred', floatX)
        h_ini_pred = T.matrix('h_ini_pred', floatX)
        k_ini_pred = T.matrix('k_ini_pred', floatX)
        w_ini_pred = T.matrix('w_ini_pred', floatX)
        bias = T.scalar('bias_generation_pred', floatX)
        return pt_ini, h_ini_pred, k_ini_pred, w_ini_pred, bias
