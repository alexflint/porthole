__author__ = 'alexflint'

import math
import itertools
import unittest

import scipy.optimize
import numpy as np
import numdifftools

import matplotlib
matplotlib.use('Agg', warn=False)
import matplotlib.pyplot as plt


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

    def as_vector(self):
        return np.hstack((self.w.flatten(), self.bv, self.bh))

    def copy(self):
        return Rbm(self.w.copy(), self.bv.copy(), self.bh.copy())

    def __str__(self):
        return 'Visible biases: %s\nHidden biases: %s\nWeights:\n%s' % \
              (self.bv, self.bh, self.w)


def save_rbm(params, path):
    with open(str(path), 'w') as fd:
        fd.write('%d %d\n' % (params.visible_size, params.hidden_size))
        fd.write(' '.join(map(str, params.bv.flatten())) + '\n')
        fd.write(' '.join(map(str, params.bh.flatten())) + '\n')
        fd.write(' '.join(map(str, params.w.flatten())) + '\n')


def load_rbm(path):
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


def log_sum_exp(xs):
    xs = np.asarray(xs)
    m = np.max(xs)
    return math.log(np.sum(np.exp(xs - m))) + m


def log_sum_neg_exp(xs):
    return log_sum_exp(np.negative(xs))


def sigmoid(x):
    return np.reciprocal(1 + np.exp(-x))


def bit_product(n):
    return itertools.product((0,1), repeat=n)


def energy(v, h, params):
    ew = -np.dot(v, np.dot(params.w, h))
    ev = -np.dot(params.bv, v)
    eh = -np.dot(params.bh, h)
    return ew + ev + eh


def hidden_conditional_naive(params, v, hi):
    energies = [[], []]
    for h in itertools.product((0,1), repeat=params.hidden_size):
        energies[h[hi]].append(energy(v, h, params))
    logp0 = log_sum_exp(np.negative(energies[0]))
    logp1 = log_sum_exp(np.negative(energies[1]))
    return sigmoid(logp1 - logp0)


def hidden_conditional(params, v, hi):
    return sigmoid(np.dot(params.w[:, hi], v) + params.bh[hi])


def hidden_conditionals(params, v):
    return sigmoid(np.dot(params.w.T, v) + params.bh)


def visible_conditional(params, h, vi):
    return sigmoid(np.dot(params.w[vi, :], h) + params.bv[vi])


def visible_conditionals(params, h):
    return sigmoid(np.dot(params.w, h) + params.bv)


def sample_visible(params, h):
    return np.random.rand(params.visible_size) < visible_conditionals(params, h)


def sample_hidden(params, v):
    return np.random.rand(params.hidden_size) < hidden_conditionals(params, v)


def loglikelihood_naive(params, v):
    cond_energies = []
    for h in bit_product(params.hidden_size):
        cond_energies.append(energy(v, h, params))
    joint_energies = []
    for x in bit_product(params.state_size):
        v, h = chop(x, params.visible_size)
        joint_energies.append(energy(v, h, params))
    return log_sum_neg_exp(cond_energies) - log_sum_neg_exp(joint_energies)


def sum_loglikelihood_naive(params, vs):
    return sum(loglikelihood_naive(params, v) for v in vs)


def likelihood_naive(params, v):
    return math.exp(loglikelihood_naive(params, v))


def print_table(params):
    for v in bit_product(params.visible_size):
        L = loglikelihood_naive(params, v)
        print '  %s: %.2f' % (bitstring(v), np.exp(L))


def weight_gradient_naive(params, v0):
    G = np.zeros((params.visible_size, params.hidden_size))
    for i in range(params.hidden_size):
        G[:, i] += hidden_conditional(params, v0, i) * v0
    for v in bit_product(params.visible_size):
        # TODO: avoid underflow here by using log_sum_exp appropriately
        lik = likelihood_naive(params, v)
        condps = [hidden_conditional(params, v, i)
                  for i in range(params.hidden_size)]
        G -= lik * np.outer(v, condps)
    return G


def visible_bias_gradient_naive(params, v0):
    G = np.asarray(v0).astype(float).copy()
    for v in bit_product(params.visible_size):
        # TODO: avoid underflow here by using log_sum_exp appropriately
        G -= likelihood_naive(params, v) * np.asarray(v).astype(float)
    return G


def hidden_bias_gradient_naive(params, v0):
    G = hidden_conditionals(params, v0)
    for v in bit_product(params.visible_size):
        # TODO: avoid underflow here by using log_sum_exp appropriately
        G -= likelihood_naive(params, v) * np.asarray(hidden_conditionals(params, v))
    return G


def gradient_naive(params, v0):
    return Rbm(weight_gradient_naive(params, v0),
               visible_bias_gradient_naive(params, v0),
               hidden_bias_gradient_naive(params, v0))


def random_rbm(nv, nh, stddev=.1):
    w = np.random.normal(loc=0, scale=stddev, size=(nv, nh))
    bv = np.random.normal(loc=0, scale=stddev, size=nv)
    bh = np.random.normal(loc=0, scale=stddev, size=nh)
    return Rbm(w, bv, bh)


def cd_gradient(rbm, v):
    v = np.asarray(v, dtype=float)
    gradient = Rbm.zero(rbm.visible_size, rbm.hidden_size)
    gradient.bv[:] = v
    gradient.bh[:] = hidden_conditionals(rbm, v)
    gradient.w = np.outer(v, hidden_conditionals(rbm, v))
    return gradient


def train_rbm(dataset,
              learning_rate,
              num_steps,
              seed,
              num_gibbs_steps=1):
    """Implements vanilla contrastive divergence."""
    nv = seed.visible_size
    nh = seed.hidden_size
    cur_params = seed.copy()
    dataset = np.asarray(dataset)
    for step in range(num_steps):
        print 'Step %d' % step
        gradient = Rbm.zero(cur_params.visible_size, cur_params.hidden_size)
        for item in dataset:
            vpos = vneg = item
            for gibbs_step in range(num_gibbs_steps):
                hneg = sample_hidden(cur_params, vneg)
                vneg = sample_visible(cur_params, hneg)

            print '  Positive: %s' % bitstring(vpos)
            print '  Negative: %s' % bitstring(vneg)

            pos_gradient = cd_gradient(cur_params, vpos)
            neg_gradient = cd_gradient(cur_params, vneg)
            gradient.bv += pos_gradient.bv - neg_gradient.bv
            gradient.bh += pos_gradient.bh - neg_gradient.bh
            gradient.w += pos_gradient.w - neg_gradient.w


        #G_true = gradient_naive(cur_params, data)
        #print 'CD gradient:'
        #print G
        #print 'True gradient:'
        #print G_true
        ##G = G_true

        cur_params.w += gradient.w * learning_rate
        cur_params.bv += gradient.bv * learning_rate
        cur_params.bh += gradient.bh * learning_rate

        #loglik = loglikelihood_naive(cur_params, data)
        #print '  Log likelihood: %.2f' % loglik
        #loglikelihoods.append(loglik)

    return cur_params


def compute_compression_error(rbm, dataset):
    sum_mse = 0
    for item in dataset:
        compressed = hidden_conditionals(rbm, item)
        uncompressed = visible_conditionals(rbm, compressed)
        print 'uncompressed:',uncompressed
        mse = np.sum(np.square(uncompressed - item.astype(float))) / np.prod(np.shape(item))
        sum_mse += mse
    return sum_mse / len(dataset)


class RbmTest(unittest.TestCase):
    def setUp(self):
        self.nv = 3
        self.nh = 2
        w = np.random.normal(loc=0, scale=.1, size=(self.nv, self.nh))
        bv = np.random.normal(loc=0, scale=.1, size=self.nv)
        bh = np.random.normal(loc=0, scale=.1, size=self.nh)
        self.params = Rbm(w, bv, bh)
        self.v = np.random.randint(0, 2, self.nv)
        self.h = np.random.randint(0, 2, self.nh)

    def test_conditional(self):
        self.assertAlmostEqual(
            hidden_conditional(self.params, self.v, 0),
            hidden_conditional_naive(self.params, self.v, 0),
            8)

    def test_hidden_conditionals(self):
        c1 = [hidden_conditional(self.params, self.v, i) for i in range(self.nh)]
        c2 = hidden_conditionals(self.params, self.v)
        np.testing.assert_array_almost_equal(c1, c2)


    def test_visible_conditionals(self):
        c1 = [visible_conditional(self.params, self.h, i) for i in range(self.nv)]
        c2 = visible_conditionals(self.params, self.h)
        np.testing.assert_array_almost_equal(c1, c2)


    def test_sum_likelihood(self):
        sum = 0.
        for v in bit_product(self.params.visible_size):
            sum += math.exp(loglikelihood_naive(self.params, v))
        self.assertAlmostEqual(sum, 1.)


    def test_gradient(self):
        L = lambda x: loglikelihood_naive(Rbm.from_vector(x, self.nv, self.nh), self.v)
        G = gradient_naive(self.params, self.v)

        GG = numdifftools.Gradient(L)(self.params.as_vector())
        G_numeric = Rbm.from_vector(GG, self.nv, self.nh)

        np.testing.assert_array_almost_equal(G.w, G_numeric.w)
        np.testing.assert_array_almost_equal(G.bv, G_numeric.bv)
        np.testing.assert_array_almost_equal(G.bh, G_numeric.bh)


def main():
    np.random.seed(124)

    def run_training():
        nv = 3
        nh = 2
        params = random_rbm(nv, nh)
        v = np.array((1, 0, 0))
        L = lambda x: loglikelihood_naive(Rbm.from_vector(x, nv, nh), v)
        C = lambda x: -L(x)

        print 'Training data: ' + ''.join(map(str, map(int, v)))
        print 'Optimizing...'
        xopt = scipy.optimize.fmin(C, params.as_vector())
        opt_params = Rbm.from_vector(xopt, nv, nh)

        print opt_params
        print 'Likelihood of data:', loglikelihood_naive(opt_params, v)
        print_table(opt_params)

    def run_training2():
        nv = 4
        nh = 1
        params = random_rbm(nv, nh)

        vs = [np.array([1, 0, 0, 0]),
              np.array([0, 0, 0, 1]),
              np.array([1, 0, 0, 1])]

        L = lambda x: sum_loglikelihood_naive(Rbm.from_vector(x, nv, nh), vs)
        C = lambda x: -L(x)

        xopt = scipy.optimize.fmin(C, params.as_vector())
        opt_params = Rbm.from_vector(xopt, nv, nh)

        print 'Final parameters:'
        print opt_params
        print 'Log likelihood of data:', sum_loglikelihood_naive(opt_params, vs)
        print_table(opt_params)


    def run_gradient():
        nv = 3
        nh = 2
        params = random_rbm(nv, nh)
        v = np.array((1, 0, 0))
        L = lambda x: loglikelihood_naive(Rbm.from_vector(x, nv, nh), v)

        G = gradient_naive(params, v)

        GG = numdifftools.Gradient(L)(params.as_vector())
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
        seed_params = random_rbm(nv, nh)

        print_table(seed_params)
        learned_params = train_rbm(data, learning_rate=.1, num_steps=10000, seed=seed_params)
        print_table(learned_params)

        save_rbm(learned_params, 'rbms/2x4.txt')


    def run_contrastive_divergence2():
        data = np.array([[1, 0, 0, 0],
                         [0, 0, 0, 1]])
        nv = data.shape[1]
        nh = 1
        seed_params = random_rbm(nv, nh)

        print_table(seed_params)
        learned_params = train_rbm(data, learning_rate=.1, num_steps=10, seed=seed_params)
        print_table(learned_params)
        print compute_compression_error(learned_params, data)


    def run_compression():
        data = np.array([[1, 0, 0, 0],
                         [0, 0, 0, 1]])
        rbm = load_rbm('rbms/2x4.txt')
        mse = compute_compression_error(rbm, data)
        print 'Mean squared error: %.2f%%' % (mse * 100.)


    #run_gradient()
    #run_training()
    #run_contrastive_divergence()
    run_contrastive_divergence2()
    #run_compression()


if __name__ == '__main__':
    main()
