__author__ = 'alexflint'

import math
import itertools
import unittest

import numpy as np
import numdifftools
import scipy.optimize
import scipy.signal

import matplotlib
matplotlib.use('Agg', warn=False)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages as PdfPages


def chop(x, i):
    return x[:i], x[i:]


def pieces(x, *sizes):
    assert len(x) == np.sum(sizes)
    sum = 0
    for size in sizes:
        yield x[sum:sum+size]
        sum += size


def bitstring(x):
    return ''.join(map(str, map(int, x)))


def rbm_str(v, h):
    return 'RBM(v=%s,h=%s)' % (bitstring(v), bitstring(h))


def log_sum_exp(xs):
    xs = np.asarray(xs)
    m = np.max(xs)
    return math.log(np.sum(np.exp(xs - m))) + m


def log_sum_neg_exp(xs):
    return log_sum_exp(np.negative(xs))


def sigmoid(x):
    return np.reciprocal(1 + np.exp(-x))


def bit_product(n):
    if np.isscalar(n):
        return itertools.product((0,1), repeat=int(n))
    else:
        return (np.reshape(x, n)
                for x in itertools.product((0,1), repeat=np.prod(n)))


def boxrange(shape):
    return itertools.product(*map(xrange, shape))


class Rbm(object):
    def __init__(self, w, bv, bh):
        self.w = np.asarray(w)
        self.bv = np.asarray(bv)
        self.bh = np.asarray(bh)
        assert self.w.shape == (len(self.bv), len(self.bh))

    @property
    def visible_size(self):
        return len(self.bv)

    @property
    def hidden_size(self):
        return len(self.bh)

    @property
    def state_size(self):
        return len(self.bv) + len(self.bh)

    @classmethod
    def from_vector(cls, x, nv, nh):
        w, bv, bh = pieces(x, nv*nh, nv, nh)
        return Rbm(np.reshape(w, (nv, nh)), bv, bh)

    @classmethod
    def zero(cls, nv, nh):
        return Rbm(w=np.zeros((nv, nh)),
                   bv=np.zeros(nv),
                   bh=np.zeros(nh))

    @classmethod
    def zero_like(cls, rbm):
        return cls.zero(rbm.visible_size, rbm.hidden_size)

    @classmethod
    def random(cls, nv, nh, stddev=.1):
        w = np.random.normal(loc=0, scale=stddev, size=(nv, nh))
        bv = np.random.normal(loc=0, scale=stddev, size=nv)
        bh = np.random.normal(loc=0, scale=stddev, size=nh)
        return Rbm(w, bv, bh)



    def as_vector(self):
        return np.hstack((self.w.flatten(), self.bv, self.bh))

    def copy(self):
        return Rbm(self.w.copy(), self.bv.copy(), self.bh.copy())

    def __str__(self):
        return 'Visible biases: %s\nHidden biases: %s\nWeights:\n%s' % \
              (self.bv, self.bh, self.w)

    def energy(self, v, h):
        ew = -np.dot(v, np.dot(self.w, h))
        ev = -np.dot(self.bv, v)
        eh = -np.dot(self.bh, h)
        return ew + ev + eh


    def hidden_conditional_naive(self, v, hi):
        energies = [[], []]
        for h in bit_product(self.hidden_size):
            energies[h[hi]].append(self.energy(v, h))
        logp0 = log_sum_exp(np.negative(energies[0]))
        logp1 = log_sum_exp(np.negative(energies[1]))
        return sigmoid(logp1 - logp0)


    def hidden_conditional(self, v, hi):
        return sigmoid(np.dot(self.w[:, hi], v) + self.bh[hi])


    def hidden_conditionals(self, v):
        return sigmoid(np.dot(self.w.T, v) + self.bh)


    def visible_conditional(self, h, vi):
        return sigmoid(np.dot(self.w[vi, :], h) + self.bv[vi])


    def visible_conditionals(self, h):
        return sigmoid(np.dot(self.w, h) + self.bv)


    def sample_visible(self, h):
        return np.random.rand(self.visible_size) < self.visible_conditionals(h)


    def sample_hidden(self, v):
        return np.random.rand(self.hidden_size) < self.hidden_conditionals(v)


    def loglikelihood_naive(self, v):
        cond_energies = []
        for h in bit_product(self.hidden_size):
            cond_energies.append(self.energy(v, h))
        joint_energies = []
        for x in bit_product(self.state_size):
            v, h = chop(x, self.visible_size)
            joint_energies.append(self.energy(v, h))
        return log_sum_neg_exp(cond_energies) - log_sum_neg_exp(joint_energies)


    def sum_loglikelihood_naive(self, vs):
        return sum(self.loglikelihood_naive(v) for v in vs)


    def likelihood_naive(self, v):
        return math.exp(self.loglikelihood_naive(v))


    def print_table(self):
        for v in bit_product(self.visible_size):
            L = self.loglikelihood_naive(v)
            print '  %s: %.2f' % (bitstring(v), math.exp(L))

    def save(self, path):
        with open(str(path), 'w') as fd:
            fd.write('%d %d\n' % (self.visible_size, self.hidden_size))
            fd.write(' '.join(map(str, self.bv.flatten())) + '\n')
            fd.write(' '.join(map(str, self.bh.flatten())) + '\n')
            fd.write(' '.join(map(str, self.w.flatten())) + '\n')

    @classmethod
    def load(cls, path):
        with open(str(path)) as fd:
            lines = list(fd)
            assert len(lines) == 4
            nv, nh = map(int, lines[0].split())
            bv = map(float, lines[1].split())
            bh = map(float, lines[2].split())
            w = np.array(map(float, lines[3].split())).reshape((nv, nh))
            assert len(bv) == nv
            assert len(bh) == nh
            return Rbm(w, bv, bh)


class ConvolutionalRbm(object):
    def __init__(self, w, bv, bh, vshape):
        self.w = np.asarray(w)
        self.bv = np.squeeze(np.asarray(bv))
        self.bh = np.asarray(bh)

        assert len(vshape) == 2
        assert self.w.shape[1] % 2 == 1
        assert self.w.shape[2] % 2 == 1
        self.vshape = vshape
        self.hshape = (self.w.shape[0],
                       vshape[0] - self.w.shape[1] + 1,
                       vshape[1] - self.w.shape[2] + 1)

        assert np.ndim(self.w) == 3
        assert np.ndim(self.bh) == 1
        assert np.ndim(self.bv) == 0
        assert len(self.w) == len(self.bh)

    @property
    def visible_shape(self):
        return self.vshape

    @property
    def hidden_shape(self):
        return self.hshape

    @property
    def state_size(self):
        return np.prod(self.visible_shape) + np.prod(self.hidden_shape)

    def state_from_vector(self, x):
        assert len(x) == self.state_size
        v, h = chop(x, np.prod(self.vshape))
        return np.reshape(v, self.vshape), np.reshape(h, self.hshape)

    @classmethod
    def from_vector(cls, x, vshape, wshape):
        w, bv, bh = pieces(x, np.prod(wshape), 1, wshape[0])
        return ConvolutionalRbm(np.reshape(w, wshape), bv, bh, vshape)

    @classmethod
    def from_vector_like(cls, x, crbm):
        return cls.from_vector(x, crbm.vshape, crbm.w.shape)

    @classmethod
    def zero(cls, vshape, wshape):
        return ConvolutionalRbm(w=np.zeros(wshape),
                                bv=0.,
                                bh=np.zeros(wshape[0]),
                                vshape=vshape)

    @classmethod
    def zero_like(cls, crbm):
        return cls.zero_like(crbm.vshape, crbm.w.shape)

    def as_vector(self):
        return np.hstack((self.w.flatten(), self.bv, self.bh))

    def copy(self):
        return Rbm(self.w.copy(), self.bv.copy(), self.bh.copy())

    def __str__(self):
        return 'Visible biases: %s\nHidden biases: %s\nWeights:\n%s' % \
               (self.bv, self.bh, self.w.flatten())

    def forwards_convolve(self, v):
        # Note that using correlate2d below is equivalent to convolving with a flipped kernel,
        # which is the analogue to transposing w in the pure RBM case.
        return np.array([scipy.signal.correlate2d(v, k, 'valid')
                        for k in self.w])

    def backwards_convolve(self, h):
        # Note that we want to "enlarge" the hidden units so we use 'full' size correlations
        return np.sum([scipy.signal.convolve2d(h, k, 'full')
                       for h, k in zip(h, self.w)], axis=0)

    def hidden_conditional_naive(self, v, hi):
        energies = [[], []]
        for h in bit_product(self.hshape):
            energies[h[hi]].append(self.energy(v, h))
        logp0 = log_sum_exp(np.negative(energies[0]))
        logp1 = log_sum_exp(np.negative(energies[1]))
        return sigmoid(logp1 - logp0)

    def hidden_conditionals_naive(self, v):
        h = np.empty(self.hshape)
        for i in range(h.shape[0]):
            for j in range(h.shape[1]):
                for k in range(h.shape[2]):
                    h[i, j, k] = self.hidden_conditional_naive(v, (i, j, k))
        return h

    def hidden_conditionals(self, v):
        return sigmoid(self.forwards_convolve(v) + self.bh[:, np.newaxis, np.newaxis])

    def sample_hidden(self, v):
        return np.random.rand(*self.hshape) < self.hidden_conditionals(v)

    def visible_conditional_naive(self, h, vi):
        energies = [[], []]
        for v in bit_product(self.vshape):
            energies[v[vi]].append(self.energy(v, h))
        logp0 = log_sum_exp(np.negative(energies[0]))
        logp1 = log_sum_exp(np.negative(energies[1]))
        return sigmoid(logp1 - logp0)

    def visible_conditionals_naive(self, h):
        v = np.empty(self.vshape)
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                v[i, j] = self.visible_conditional_naive(h, (i, j))
        return v

    def visible_conditionals(self, h):
        return sigmoid(self.backwards_convolve(h) + self.bv)

    def sample_visible(self, h):
        return np.random.rand(*self.vshape) < self.visible_conditionals(h)

    def energy(self, v, h):
        ew = -np.sum(h * self.forwards_convolve(v))
        ev = -np.sum(v) * self.bv
        eh = -np.sum(h * self.bh[:, np.newaxis, np.newaxis])
        return ew + ev + eh

    def loglikelihood_naive(self, v):
        cond_energies = []
        for h in bit_product(self.hshape):
            cond_energies.append(self.energy(v, h))
        joint_energies = []
        for x in bit_product(self.state_size):
            v, h = self.state_from_vector(x)
            joint_energies.append(self.energy(v, h))
        return log_sum_neg_exp(cond_energies) - log_sum_neg_exp(joint_energies)

    def sum_loglikelihood_naive(self, vs):
        return sum(self.loglikelihood_naive(v) for v in vs)

    def likelihood_naive(self, v):
        return math.exp(self.loglikelihood_naive(v))

    def print_table(self):
        for v in bit_product(self.visible_size):
            loglik = self.loglikelihood_naive(v)
            print '  %s: %.2f' % (bitstring(v), math.exp(loglik))


def rbm_gradient_naive(rbm, v0):
    G = Rbm.zero_like(rbm)

    G.w = np.zeros((rbm.visible_size, rbm.hidden_size))
    for i in range(rbm.hidden_size):
        G.w[:, i] += rbm.hidden_conditional(v0, i) * v0
    for v in bit_product(rbm.visible_size):
        # TODO: avoid underflow here by using log_sum_exp appropriately
        lik = rbm.likelihood_naive(v)
        condps = rbm.hidden_conditionals(v)
        G.w -= lik * np.outer(v, condps)

    G.bv = np.asarray(v0).astype(float).copy()
    for v in bit_product(rbm.visible_size):
        # TODO: avoid underflow here by using log_sum_exp appropriately
        G.bv -= rbm.likelihood_naive(v) * np.asarray(v).astype(float)


    G.bh = rbm.hidden_conditionals(v0)
    for v in bit_product(rbm.visible_size):
        # TODO: avoid underflow here by using log_sum_exp appropriately
        G.bh -= rbm.likelihood_naive(v) * np.asarray(rbm.hidden_conditionals(v))

    return G



def free_energy_gradient(rbm, v):
    v = np.asarray(v, dtype=float)
    h = rbm.hidden_conditionals(v)
    return Rbm(np.outer(v, h), v, h)


def train_rbm(dataset,
              learning_rate,
              num_steps,
              seed,
              num_gibbs_steps=1,
              weight_decay=0.):
    """Implements vanilla contrastive divergence."""
    cur = seed.copy()
    dataset = np.asarray(dataset)
    last_progress = None
    for step in range(num_steps):
        if num_steps > 1:
            progress = int(100. * step / num_steps)
            if progress != last_progress:
                print 'Training %d%% complete (step %d of %d)' % (progress, step+1, num_steps)
                last_progress = progress

        gradient = Rbm.zero(cur.visible_size, cur.hidden_size)

        for item in dataset:
            vpos = vneg = item
            for gibbs_step in range(num_gibbs_steps):
                hneg = cur.sample_hidden(vneg)
                vneg = cur.sample_visible(hneg)

            #print '  Positive: %s' % bitstring(vpos)
            #print '  Negative: %s' % bitstring(vneg)

            pos_gradient = free_energy_gradient(cur, vpos)
            neg_gradient = free_energy_gradient(cur, vneg)
            gradient.bv += pos_gradient.bv - neg_gradient.bv
            gradient.bh += pos_gradient.bh - neg_gradient.bh
            gradient.w += pos_gradient.w - neg_gradient.w

        # Normalize for dataset size
        gradient.bv /= len(dataset)
        gradient.bh /= len(dataset)
        gradient.w /= len(dataset)

        # Add weight decay term
        if weight_decay != 0:
            gradient.bv -= weight_decay * cur.bv
            gradient.bh -= weight_decay * cur.bh
            gradient.w -= weight_decay * cur.w

        #G_true = gradient_naive(cur_rbm, data)
        #print 'CD gradient:'
        #print G
        #print 'True gradient:'
        #print G_true
        ##G = G_true

        cur.w += gradient.w * learning_rate
        cur.bv += gradient.bv * learning_rate
        cur.bh += gradient.bh * learning_rate

        #loglik = loglikelihood_naive(cur_rbm, data)
        #print '  Log likelihood: %.2f' % loglik
        #loglikelihoods.append(loglik)

    return cur


def compute_compression_error(rbm, dataset):
    sum_mse = 0
    for item in dataset:
        item = np.asarray(item, float)
        compressed = rbm.hidden_conditionals(item)
        reconstructed = rbm.visible_conditionals(compressed)
        mse = np.sum(np.square(reconstructed - item)) / np.prod(np.shape(item))
        sum_mse += mse
    return sum_mse / len(dataset)


def plot_reconstructions(rbm, dataset, out, shape=None):
    pdf = PdfPages(out)
    for item in dataset:
        item = np.asarray(item, dtype=float)
        compressed = rbm.hidden_conditionals(item)
        recon = rbm.visible_conditionals(compressed)

        plt.clf()

        if shape is not None:
            item = np.reshape(item, shape)
            recon = np.reshape(recon, shape)
        if item.ndim == 1:
            item = np.atleast_2d(item).T
            recon = np.atleast_2d(recon).T

        plt.subplot(121)
        plt.imshow(item, vmin=0., vmax=1., interpolation='nearest', cmap='summer')
        plt.axis('equal')

        plt.subplot(122)
        plt.imshow(recon, vmin=0., vmax=1., interpolation='nearest', cmap='summer')
        plt.axis('equal')

        pdf.savefig()

    pdf.close()


def plot_features(rbm, dataset, out, shape=None, features_shape=None):
    pdf = PdfPages(out)
    for item in dataset:
        item = np.asarray(item, dtype=float)
        features = rbm.hidden_conditionals(item)
        #recon = visible_conditionals(rbm, compressed)

        if shape is not None:
            item = np.reshape(item, shape)
        if features_shape is not None:
            features = np.reshape(features, features_shape)
        if item.ndim == 1:
            item = np.atleast_2d(item).T
        if features.ndim == 1:
            features = np.atleast_2d(features).T

        plt.clf()

        plt.subplot(121)
        plt.imshow(item, vmin=0., vmax=1., interpolation='nearest', cmap='summer')
        plt.axis('equal')

        plt.subplot(122)
        plt.imshow(features, vmin=0., vmax=1., interpolation='nearest', cmap='summer')
        plt.axis('equal')

        pdf.savefig()

    pdf.close()


class RbmTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(124)
        self.nv = 3
        self.nh = 2
        w = np.random.normal(loc=0, scale=.1, size=(self.nv, self.nh))
        bv = np.random.normal(loc=0, scale=.1, size=self.nv)
        bh = np.random.normal(loc=0, scale=.1, size=self.nh)
        self.rbm = Rbm(w, bv, bh)
        self.v = np.random.randint(0, 2, self.nv)
        self.h = np.random.randint(0, 2, self.nh)

    def test_conditional(self):
        self.assertAlmostEqual(
            self.rbm.hidden_conditional(self.v, 0),
            self.rbm.hidden_conditional_naive(self.v, 0),
            8)

    def test_hidden_conditionals(self):
        c1 = [self.rbm.hidden_conditional(self.v, i) for i in range(self.nh)]
        c2 = self.rbm.hidden_conditionals(self.v)
        np.testing.assert_array_almost_equal(c1, c2)

    def test_visible_conditionals(self):
        c1 = [self.rbm.visible_conditional(self.h, i) for i in range(self.nv)]
        c2 = self.rbm.visible_conditionals(self.h)
        np.testing.assert_array_almost_equal(c1, c2)

    def test_sum_likelihood(self):
        sum = 0.
        for v in bit_product(self.rbm.visible_size):
            sum += math.exp(self.rbm.loglikelihood_naive(v))
        self.assertAlmostEqual(sum, 1.)

    def test_gradient(self):
        L = lambda x: Rbm.from_vector(x, self.nv, self.nh).loglikelihood_naive(self.v)
        G = rbm_gradient_naive(self.rbm, self.v)

        GG = numdifftools.Gradient(L)(self.rbm.as_vector())
        G_numeric = Rbm.from_vector(GG, self.nv, self.nh)

        np.testing.assert_array_almost_equal(G.w, G_numeric.w)
        np.testing.assert_array_almost_equal(G.bv, G_numeric.bv)
        np.testing.assert_array_almost_equal(G.bh, G_numeric.bh)


class ConvolutionalRbmTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(124)
        vshape = (4, 1)
        wshape = (2, 3, 1)
        w = np.random.normal(loc=0, scale=.1, size=wshape)
        bv = np.random.normal(loc=0, scale=.1, size=1)
        bh = np.random.normal(loc=0, scale=.1, size=wshape[0])
        self.crbm = ConvolutionalRbm(w, bv, bh, vshape)
        self.v = np.random.randint(0, 2, self.crbm.vshape)
        self.h = np.random.randint(0, 2, self.crbm.hshape)

    def test_hidden_conditionals(self):
        c1 = self.crbm.hidden_conditionals_naive(self.v)
        c2 = self.crbm.hidden_conditionals(self.v)
        np.testing.assert_array_almost_equal(c1, c2)

    def test_visible_conditionals(self):
        c1 = self.crbm.visible_conditionals_naive(self.h)
        c2 = self.crbm.visible_conditionals(self.h)
        np.testing.assert_array_almost_equal(c1, c2)

    def test_sum_likelihood(self):
        sum = 0.
        for v in bit_product(self.crbm.vshape):
            sum += math.exp(self.crbm.loglikelihood_naive(v))
        self.assertAlmostEqual(sum, 1.)

    def _test_gradient(self):
        def L(x):
            crbm = ConvolutionalRbm.from_vector_like(x, self.crbm)
            return crbm.loglikelihood_naive(self.v)
        G = gradient_naive_crbm(self.crbm, self.v)

        GG = numdifftools.Gradient(L)(self.crbm.as_vector())
        G_numeric = ConvolutionalRbm.from_vector_like(GG, self.crbm)

        np.testing.assert_array_almost_equal(G.w, G_numeric.w)
        np.testing.assert_array_almost_equal(G.bv, G_numeric.bv)
        np.testing.assert_array_almost_equal(G.bh, G_numeric.bh)



def main():
    np.random.seed(124)

    def run_training():
        nv = 3
        nh = 2
        rbm = Rbm.random(nv, nh)
        v = np.array((1, 0, 0))
        L = lambda x: Rbm.from_vector(x, nv, nh).loglikelihood_naive(v)
        C = lambda x: -L(x)

        print 'Training data: ' + ''.join(map(str, map(int, v)))
        print 'Optimizing...'
        xopt = scipy.optimize.fmin(C, rbm.as_vector())
        opt_rbm = Rbm.from_vector(xopt, nv, nh)

        print opt_rbm
        print 'Likelihood of data:', opt_rbm.loglikelihood_naive(v)
        opt_rbm.print_table()

    def run_training2():
        nv = 4
        nh = 1
        rbm = Rbm.random(nv, nh)

        vs = [np.array([1, 0, 0, 0]),
              np.array([0, 0, 0, 1]),
              np.array([1, 0, 0, 1])]

        L = lambda x: Rbm.from_vector(x, nv, nh).sum_loglikelihood_naive(vs)
        C = lambda x: -L(x)

        xopt = scipy.optimize.fmin(C, rbm.as_vector())
        opt_rbm = Rbm.from_vector(xopt, nv, nh)

        print 'Final parameters:'
        print opt_rbm
        print 'Log likelihood of data:', opt_rbm.sum_loglikelihood_naive(vs)
        opt_rbm.print_table()

    def run_gradient():
        nv = 3
        nh = 2
        rbm = Rbm.random(nv, nh)
        v = np.array((1, 0, 0))
        L = lambda x: Rbm.from_vector(x, nv, nh).loglikelihood_naive(v)

        G = rbm_gradient_naive(rbm, v)

        GG = numdifftools.Gradient(L)(rbm.as_vector())
        G_numeric = Rbm.from_vector(GG, nv, nh)

        print '\nAnalytic gradient of weights:'
        print G.w
        print 'Numerical gradient of weights:'
        print G_numeric.w

        print '\nAnalytic gradient of visible biases:'
        print G.bv
        print 'Numerical gradient of visible biases:'
        print G_numeric.bv

        print '\nAnalytic gradient of hidden bias:'
        print G.bh
        print 'Numerical gradient of hidden biases:'
        print G_numeric.bh

    def run_contrastive_divergence():
        data = np.array([[1, 0, 0, 0],
                         [0, 0, 0, 1]])
        nv = data.shape[1]
        nh = 1
        seed_rbm = Rbm.random(nv, nh)

        seed_rbm.print_table()
        learned_rbm = train_rbm(data, learning_rate=.1, num_steps=10000, seed=seed_rbm)
        learned_rbm.print_table()

        learned_rbm.save('rbms/2x4.txt')

    def run_contrastive_divergence2():
        data = np.array([[1, 0, 0, 0],
                         [0, 0, 0, 1]])
        nv = data.shape[1]
        nh = 1
        seed_rbm = Rbm.random(nv, nh)

        seed_rbm.print_table()
        learned_rbm = train_rbm(data, learning_rate=.1, num_steps=10, seed=seed_rbm)
        learned_rbm.print_table()
        print compute_compression_error(learned_rbm, data)

    def run_compression():
        data = np.array([[1, 0, 0, 0],
                         [0, 0, 0, 1]])
        rbm = Rbm.load('rbms/2x4.txt')
        mse = compute_compression_error(rbm, data)
        print 'Mean squared error: %.2f%%' % (mse * 100.)

    def make_block_image(center, radius, shape):
        center = np.asarray(center)
        image = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                if np.linalg.norm(center - (i,j)) < radius:
                    image[i,j] = 1
        return image

    def make_blocks_data():
        dataset = []
        #data.append(np.zeros(100))
        #data.append(np.ones(100))
        #data.append([i%2 for i in range(100)])
        #data.append([(i+1)%2 for i in range(100)])
        for i in range(100):
            center = np.random.randint(2, 8, size=2)
            image = make_block_image(center, 3, (10,10))
            dataset.append(image.flatten())
        return dataset

    def run_train_blocks():
        data = make_blocks_data()
        nv = len(data[0])
        nh = 50
        seed = Rbm.random(nv, nh, stddev=1e-2)

        rbm = train_rbm(data, learning_rate=.01, weight_decay=1e-4, num_steps=2000, seed=seed)
        #rbm = train_rbm(data, learning_rate=.001, weight_decay=1e-4, num_steps=1000, seed=rbm)

        rbm.save('rbms/blocks.txt')

        mse = compute_compression_error(rbm, data)
        print 'Mean squared error: %.2f%%' % (mse * 100.)

        plot_reconstructions(rbm, data[:10], shape=(10,10), out='out/reconstructions.pdf')

    def run_plot_reconstructions():
        data = make_blocks_data()
        rbm = Rbm.load('rbms/blocks.txt')
        plot_reconstructions(rbm, data, shape=(10,10), out='out/reconstructions.pdf')

    def run_plot_novel_blocks():
        images = []
        for i in range(2, 8):
            for j in range(2, 8):
                images.append(make_block_image((i,j), 3, (10,10)).flatten())

        rbm = Rbm.load('rbms/blocks.txt')
        plot_reconstructions(rbm, images, shape=(10,10), out='out/novel_reconstruction.pdf')

    def run_plot_features():
        images = []
        for i in range(2, 8):
            for j in range(2, 8):
                images.append(make_block_image((i,j), 3, (10,10)).flatten())

        rbm = Rbm.load('rbms/blocks.txt')
        plot_features(rbm, images, shape=(10,10), features_shape=(10,5), out='out/novel_reconstruction.pdf')

    #run_gradient()
    #run_training()
    #run_compression()
    run_contrastive_divergence()
    #run_contrastive_divergence2()
    #run_train_blocks()
    #run_plot_reconstructions()
    #run_plot_novel_blocks()
    #run_plot_features()


if __name__ == '__main__':
    main()
