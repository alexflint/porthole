__author__ = 'alexflint'

import math
import numpy as np
import itertools

import numdifftools as nd

def chop(x, i):
    return x[:i], x[i:]


def pieces(x, *sizes):
    assert len(x) == sum(sizes)
    sum = 0
    for size in sizes:
        yield x[sum:sum:size]
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



def log_sum_exp(xs):
    xs = np.asarray(xs)
    m = np.max(xs)
    return math.log(np.sum(np.exp(xs - m))) + m


def sigmoid(x):
    return np.reciprocal(1 + np.exp(-x))


def energy(v, h, params):
    ew = -np.dot(v, np.dot(params.w, h))
    ev = -np.dot(params.bv, v)
    eh = -np.dot(params.bh, h)
    return ew + ev + eh


def hidden_marginal_naive(params, hi):
    energies = [[], []]
    for x in itertools.product((0,1), repeat=params.state_size):
        v, h = chop(x, params.visible_size)
        energies[h[hi]].append(energy(v, h, params))
    logp0 = log_sum_exp(np.negative(energies[0]))
    logp1 = log_sum_exp(np.negative(energies[1]))
    return sigmoid(logp1 - logp0)


def hidden_conditional_naive(params, v, hi):
    energies = [[], []]
    for h in itertools.product((0,1), repeat=params.hidden_size):
        energies[h[hi]].append(energy(v, h, params))
    logp0 = log_sum_exp(np.negative(energies[0]))
    logp1 = log_sum_exp(np.negative(energies[1]))
    return sigmoid(logp1 - logp0)


def hidden_conditional(params, v, hi):
    return sigmoid(np.dot(params.w[:, hi], v) + params.bh[hi])


def visible_conditional(params, h, vi):
    return sigmoid(np.dot(params.w[vi, :], h) + params.bv[vi])




def main():
    np.random.seed(123)
    w = np.random.normal(loc=0, scale=.1, size=(3,2))
    bv = np.random.normal(loc=0, scale=.1, size=3)
    bh = np.random.normal(loc=0, scale=.1, size=2)
    params = RbmParams(w, bv, bh)

    v = np.random.randint(0, 2, params.visible_size).astype(bool)
    h = np.random.randint(0, 2, params.hidden_size).astype(bool)

    #print hidden_marginal_naive(params, 0)
    print hidden_conditional_naive(params, v, 0)
    print hidden_conditional(params, v, 0)


if __name__ == '__main__':
    main()
