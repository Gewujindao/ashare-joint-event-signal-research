"""Frozen 110-parameter W transition likelihood and 13-parameter reduction.

X is retained once per stock/origin. Each gradient evaluation computes X beta
once, accumulates the ten transition derivatives, then computes X' derivative.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit, logsumexp

METHOD = 'SHORT_W_JOINT_EVENT_DISTRIBUTION_BUY_V1'
CONTROL = 'SHORT_W_CLEAN_EVENT_LOGISTIC_REDUCTION_V1'
EDGES = ((0, 1), (0, 2), (0, 3), (1, 3), (2, 3))


class TrainingPopulation:
    def __init__(self, dates):
        dates = [r for r in dates if len(r['x'])]
        if not dates:
            raise ValueError('NO_COMPLETE_MATURE_TRAIN_EVENTS')
        self.D = len(dates)
        self.x = np.concatenate([r['x'] for r in dates]).astype(np.float64)
        self.states = np.concatenate([r['states'] for r in dates]).astype(np.int8)
        self.lengths = np.concatenate([np.full(len(r['x']), r['L'], np.int8) for r in dates])
        self.stock_weight = np.concatenate([np.full(len(r['x']), 1 / (self.D * len(r['x']))) for r in dates])
        self.path_weight = self.stock_weight / self.lengths
        self.prev = np.column_stack([np.zeros(len(self.x), np.int8), self.states[:, :-1]])
        self.y = (self.states[np.arange(len(self.x)), self.lengths - 1] == 1).astype(float)
        self.constant = (float(self.y.sum()) + .5) / (len(self.y) + 1)
        # Per-day sparse row views avoid replicating the twelve feature columns.
        self.groups = []
        for day in range(10):
            active = self.lengths > day
            self.groups.append(tuple(np.flatnonzero(active & (self.prev[:, day] == state)) for state in (0, 1, 2)))


class JointObjective:
    parameters = 110

    def __init__(self, population, offsets=None):
        self.data = population
        self.D = population.D
        self.offsets = np.zeros((5, 10)) if offsets is None else np.array(offsets, dtype=np.float64, copy=True)
        if self.offsets.shape != (5, 10):
            raise ValueError('FIVE_BY_TEN_FIXED_OFFSETS_REQUIRED')
        self.offsets.flags.writeable = False

    def value_gradient(self, theta):
        pars = np.asarray(theta, np.float64).reshape(5, 22)
        offsets, beta = pars[:, :10] + self.offsets, pars[:, 10:]
        data = self.data
        base = data.x @ beta.T
        db = np.zeros_like(base)
        da = np.zeros((5, 10), np.float64)
        value = 0.
        for day, (nrows, urows, drows) in enumerate(data.groups):
            if len(nrows):
                eta = base[nrows, :3] + offsets[:3, day]
                denom = logsumexp(np.column_stack([np.zeros(len(nrows)), eta]), axis=1)
                dest = data.states[nrows, day]
                target = np.column_stack([dest == state for state in (1, 2, 3)])
                weight = data.path_weight[nrows]
                value += float(weight @ (denom - np.sum(eta * target, axis=1)))
                deriv = (np.exp(eta - denom[:, None]) - target) * weight[:, None]
                db[nrows, :3] += deriv
                da[:3, day] += deriv.sum(axis=0)
            for edge, rows in ((3, urows), (4, drows)):
                if not len(rows):
                    continue
                eta = base[rows, edge] + offsets[edge, day]
                target = data.states[rows, day] == 3
                weight = data.path_weight[rows]
                value += float(weight @ (np.logaddexp(0, eta) - eta * target))
                deriv = (expit(eta) - target) * weight
                db[rows, edge] += deriv
                da[edge, day] += deriv.sum()
        gradient = np.column_stack([da, db.T @ data.x]).ravel()
        flat = pars.ravel()
        return value + float(flat @ flat) / (2 * self.D), gradient + flat / self.D


class ReductionObjective:
    parameters = 13

    def __init__(self, population, offset=0.):
        self.data = population
        self.D = population.D
        self.offset = float(offset)

    def value_gradient(self, theta):
        data = self.data
        eta = self.offset + theta[0] + data.x @ theta[1:]
        deriv = (expit(eta) - data.y) * data.stock_weight
        value = float(data.stock_weight @ (np.logaddexp(0, eta) - data.y * eta))
        gradient = np.r_[deriv.sum(), data.x.T @ deriv]
        return value + float(theta @ theta) / (2 * self.D), gradient + theta / self.D


def transition_matrix(theta, x, day, *, offsets=None):
    """P(day | origin X), with no observed-state or label argument."""
    pars = np.asarray(theta, np.float64).reshape(5, 22)
    eta = np.asarray(x, np.float64) @ pars[:, 10:].T + pars[:, day]
    if offsets is not None:
        eta += np.asarray(offsets)[:, day]
    logits = np.column_stack([np.zeros(len(x)), eta[:, :3]])
    probs = np.exp(logits - logsumexp(logits, axis=1)[:, None])
    matrix = np.zeros((len(x), 4, 4), np.float64)
    matrix[:, 0, :] = probs
    ub, db = expit(eta[:, 3]), expit(eta[:, 4])
    matrix[:, 1, 1], matrix[:, 1, 3] = 1 - ub, ub
    matrix[:, 2, 2], matrix[:, 2, 3] = 1 - db, db
    matrix[:, 3, 3] = 1
    return matrix


def predict(theta, x, length, *, offsets=None):
    """Marginalize every unobserved state from the initial N at origin."""
    x = np.asarray(x, np.float64)
    if x.shape != (len(x), 12) or length not in range(1, 11):
        raise ValueError('EXACT_ORIGIN_FEATURES_AND_ORIGINAL_W_REQUIRED')
    pars = np.asarray(theta, np.float64).reshape(5, 22)
    base = x @ pars[:, 10:].T
    fixed = np.zeros((5, 10)) if offsets is None else np.asarray(offsets, np.float64)
    result = np.zeros((len(x), length + 1, 4), np.float64)
    result[:, 0, 0] = 1
    for day in range(length):
        eta = base + pars[:, day] + fixed[:, day]
        logits = np.column_stack([np.zeros(len(x)), eta[:, :3]])
        probs = np.exp(logits - logsumexp(logits, axis=1)[:, None])
        old, new = result[:, day], result[:, day + 1]
        new[:, 0] = old[:, 0] * probs[:, 0]
        new[:, 1] = old[:, 0] * probs[:, 1] + old[:, 1] * expit(-eta[:, 3])
        new[:, 2] = old[:, 0] * probs[:, 2] + old[:, 2] * expit(-eta[:, 4])
        new[:, 3] = (old[:, 0] * probs[:, 3] + old[:, 1] * expit(eta[:, 3])
                     + old[:, 2] * expit(eta[:, 4]) + old[:, 3])
    return result


def select(scores, stable_ids, *, lower=False, negative_only=False):
    indices = np.flatnonzero(scores < 0) if negative_only else np.arange(len(scores))
    order = scores[indices] if lower else -scores[indices]
    return indices[np.lexsort((np.asarray(stable_ids)[indices], order))[:4]]


def observed_path_loss(theta, x, states, length, *, offsets=None):
    """Evaluation only: labels select likelihood entries, never predictions."""
    previous = np.zeros(len(x), np.int8)
    values = np.zeros(len(x), np.float64)
    for day in range(length):
        matrix = transition_matrix(theta, x, day, offsets=offsets)
        actual = states[:, day]
        values -= np.log(np.maximum(matrix[np.arange(len(x)), previous, actual], np.finfo(float).tiny)) / length
        previous = actual
    return values
