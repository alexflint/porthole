__author__ = 'alexflint'

import math
import itertools
import unittest

import scipy.optimize
import numpy as np
import numdifftools

def chop(x, i):
    return x[:i], x[i:]


def pieces(x, *sizes):
    assert len(x) == np.sum(sizes)
    sum = 0
    for size in sizes:
        yield x[sum:sum+size]
        sum += size


def rbm_str(v, h):
    return 'RBM(v=%s,h=%s)' % (''.join(map(int, v)), ''.join(map(int, h)))


class RbmParams(object):
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
        return RbmParams(np.reshape(w, (nv, nh)), bv, bh)

    def as_vector(self):
        return np.hstack((self.w.flatten(), self.bv, self.bh))

    def __str__(self):
        return 'Visible biases: %s\nHidden biases: %s\nWeights:\n%s' % \
              (self.bv, self.bh, self.w)



def log_sum_exp(xs):
    xs = np.asarray(xs)
    m = np.max(xs)
    return math.log(np.sum(np.exp(xs - m))) + m


def log_sum_neg_exp(xs):
    return log_sum_exp(np.negative(xs))


def sigmoid(x):
    return np.reciprocal(1 + np.exp(-x))


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


def bit_product(n):
    return itertools.product((0,1), repeat=n)


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


def visible_conditional(params, h, vi):
    return sigmoid(np.dot(params.w[vi, :], h) + params.bv[vi])


def print_table(params):
    for v in bit_product(params.visible_size):
        L = loglikelihood_naive(params, v)
        print ''.join(map(str, map(int, v))) + ': %.2f' % np.exp(L)


def weight_gradient_naive(params, v0):
    v0 = np.asarray(v0)
    G = np.zeros((params.visible_size, params.hidden_size))
    for i in range(params.hidden_size):
        G[:, i] += hidden_conditional(params, v0, i) * v0
    for v in bit_product(params.visible_size):
        lik = likelihood_naive(params, v)
        condps = [hidden_conditional(params, v, i)
                  for i in range(params.hidden_size)]
        G -= lik * np.outer(v, condps)
    return G


class RbmTest(unittest.TestCase):
    def setUp(self):
        self.nv = 3
        self.nh = 2
        w = np.random.normal(loc=0, scale=.1, size=(self.nv, self.nh))
        bv = np.random.normal(loc=0, scale=.1, size=self.nv)
        bh = np.random.normal(loc=0, scale=.1, size=self.nh)
        self.params = RbmParams(w, bv, bh)
        self.v = np.random.randint(0, 2, self.nv).astype(bool)
        self.h = np.random.randint(0, 2, self.nh).astype(bool)

    def test_conditional(self):
        self.assertAlmostEqual(
            hidden_conditional(self.params, self.v, 0),
            hidden_conditional_naive(self.params, self.v, 0),
            8)

    def test_sum_likelihood(self):
        sum = 0.
        for v in bit_product(self.params.visible_size):
            sum += math.exp(loglikelihood_naive(self.params, v))
        self.assertAlmostEqual(sum, 1.)


def main():
    np.random.seed(124)

    nv = 3
    nh = 2

    w = np.random.normal(loc=0, scale=.1, size=(nv, nh))
    bv = np.random.normal(loc=0, scale=.1, size=nv)
    bh = np.random.normal(loc=0, scale=.1, size=nh)
    params = RbmParams(w, bv, bh)

    v = np.random.randint(0, 2, nv).astype(bool)
    h = np.random.randint(0, 2, nh).astype(bool)

    L = lambda vec: loglikelihood_naive(RbmParams.from_vector(vec, nv, nh), v)
    C = lambda vec: -L(vec)

    def run_training():
        print 'Training data: ' + ''.join(map(str, map(int, v)))
        print 'Optimizing...'
        xopt = scipy.optimize.fmin(C, params.as_vector())
        opt_params = RbmParams.from_vector(xopt, nv, nh)

        print opt_params
        print 'Likelihood of data:', loglikelihood_naive(opt_params, v)
        print_table(opt_params)

    def run_gradient():
        Gw = weight_gradient_naive(params, v)

        GG = numdifftools.Gradient(L)(params.as_vector())
        Gw_numeric = RbmParams.from_vector(GG, nv, nh).w

        print 'Analytic gradient:'
        print Gw
        print 'Numerical gradient:'
        print Gw_numeric

        print 'Errors:'
        print (np.abs(Gw - Gw_numeric) > 1e-8).astype(int)

        #print '\nfull:'
        #print GG

    run_training()


if __name__ == '__main__':
    main()
