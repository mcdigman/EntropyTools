# SPDX-FileCopyrightText: Copyright 2026 Matthew C. Digman
# SPDX-License-Identifier: Apache-2.0
"""Tests of the nearest neighbor estimators against the predicted bias for Gaussian samples with repeats."""

from functools import cache
from typing import Literal

import numpy as np
import pytest
import scipy.special
import scipy.stats

from EntropyTools.entropy_process import _N_DIGAMMA_SMALL, _nn_corrected_log_distance, entropy_nn1, entropy_nn2, kl_divergence_nn2

SEED = 31415

N_SAMP = 1000
N_TRIAL = 100
N_REPEAT_MAX = 4000

LogMode = Literal['log', 'geo']
LOG_MODES: tuple[LogMode, ...] = ('log', 'geo')

REPEATS = np.arange(1, N_REPEAT_MAX + 1)
CORRECTIONS = {
    'log': np.log(REPEATS),
    'geo': scipy.special.digamma(REPEATS) + np.euler_gamma,
}

# The cases marked coverage here and in GEO_MANY_CASES together reach every line and branch.
# n_par, n_unique, n_repeat
UNIFORM_EXACT_CASES = [
    (1, 50, 1),
    pytest.param(1, 50, 2, marks=pytest.mark.coverage),
    (1, 50, 5),
    (1, 50, 20),
    (3, 200, 1),
    (3, 200, 2),
    (3, 200, 5),
    (3, 200, 20),
    (10, 200, 1),
    (10, 200, 2),
    (10, 200, 5),
    (10, 200, 20),
]

# n_repeat, n_par
GEO_MANY_CASES = [
    (_N_DIGAMMA_SMALL - 1, 1),
    (_N_DIGAMMA_SMALL - 1, 3),
    pytest.param(_N_DIGAMMA_SMALL, 1, marks=pytest.mark.coverage),
    (_N_DIGAMMA_SMALL, 3),
    (_N_DIGAMMA_SMALL + 1, 1),
    (_N_DIGAMMA_SMALL + 1, 3),
    (2 * _N_DIGAMMA_SMALL, 1),
    (2 * _N_DIGAMMA_SMALL, 3),
]

# Tolerances are approximately the finite-sample bias plus four standard errors.
# family, parameters, n_par, entropy atol, cross-entropy atol, KL divergence atols for repeats in test, compare, and both
CASES = [
    ('uniform', (5,), 1, 0.05, 0.03, 0.07, 0.04, 0.06),
    ('uniform', (5,), 2, 0.07, 0.05, 0.07, 0.04, 0.07),
    ('poisson', (0.5,), 1, 0.03, 0.03, 0.04, 0.03, 0.04),
    ('poisson', (0.5,), 2, 0.04, 0.03, 0.04, 0.03, 0.04),
    ('poisson', (2.0,), 1, 0.05, 0.04, 0.05, 0.04, 0.06),
    ('poisson', (2.0,), 2, 0.06, 0.04, 0.05, 0.04, 0.06),
    ('poisson', (5.0,), 1, 0.06, 0.04, 0.07, 0.05, 0.08),
    ('poisson', (5.0,), 2, 0.09, 0.05, 0.08, 0.05, 0.07),
    ('poisson', (20.0,), 1, 0.11, 0.06, 0.12, 0.07, 0.14),
    ('geometric', (0.5,), 1, 0.05, 0.03, 0.06, 0.04, 0.07),
    ('geometric', (0.5,), 2, 0.06, 0.05, 0.06, 0.04, 0.06),
    ('geometric', (0.2,), 1, 0.08, 0.05, 0.10, 0.05, 0.11),
    ('geometric', (0.2,), 2, 0.11, 0.07, 0.11, 0.06, 0.11),
    ('mixed', (2.0, 0.5), 1, 0.09, 0.04, 0.10, 0.04, 0.11),
    ('mixed', (2.0, 0.5), 2, 0.10, 0.06, 0.09, 0.06, 0.10),
    ('mixed', (5.0, 0.5), 1, 0.11, 0.05, 0.12, 0.05, 0.14),
    ('mixed', (5.0, 0.5), 2, 0.11, 0.07, 0.12, 0.07, 0.11),
]

# family, parameters, log mode predicted to be less biased
PREFERRED_CASES = [
    ('uniform', (5,), 'log'),
    ('poisson', (5.0,), 'log'),
    ('poisson', (20.0,), 'log'),
    ('poisson', (0.5,), 'geo'),
    ('geometric', (0.5,), 'geo'),
    ('geometric', (0.2,), 'geo'),
    ('mixed', (2.0, 0.5), 'geo'),
    ('mixed', (5.0, 0.5), 'log'),
]

# cases where the predicted difference is large compared to the scatter of the entropy
PREFERRED_WIDE_CASES = [
    ('uniform', (5,), 'log'),
    ('poisson', (5.0,), 'log'),
    ('poisson', (20.0,), 'log'),
    ('geometric', (0.5,), 'geo'),
    ('geometric', (0.2,), 'geo'),
]

# n_par, n_base, entropy atol, cross-entropy atol, KL divergence atol
METROPOLIS_CASES = [
    (1, 20000, 0.07, 0.04, 0.08),
    (1, 1000, 0.08, 0.05, 0.10),
    (1, 200, 0.11, 0.06, 0.14),
    (1, 50, 0.18, 0.09, 0.24),
    (2, 20000, 0.18, 0.07, 0.18),
    (2, 2000, 0.15, 0.07, 0.15),
]

METROPOLIS_WIDTH = 10.0


def gaussian_entropy(n_par):
    """Entropy of a unit Gaussian."""
    return n_par / 2 * np.log(2 * np.pi * np.e)


def unique_draw(rng, n_par):
    """Draw Gaussian samples without repeats."""
    return rng.normal(size=(N_SAMP, n_par))


def uniform_draw(rng, n_par, n_repeat):
    """Draw Gaussian samples with the same number of repeats of each."""
    return np.repeat(rng.normal(size=(N_SAMP // n_repeat, n_par)), n_repeat, axis=0)


def uniform_pmf(n_repeat):
    """Distribution of a constant number of repeats."""
    return np.where(n_repeat == REPEATS, 1.0, 0.0)


def poisson_draw(rng, n_par, rate):
    """Draw with replacement from a finite set of Gaussian samples."""
    n_base = int(np.round(N_SAMP / rate))
    return rng.normal(size=(n_base, n_par))[rng.integers(n_base, size=N_SAMP)]


def poisson_pmf(rate):
    """Distribution of a Poisson number of repeats, given at least one."""
    return scipy.stats.poisson.pmf(REPEATS, rate) / -np.expm1(-rate)


def geometric_draw(rng, n_par, prob):
    """Draw Gaussian samples where each step moves to a new sample with the given probability."""
    return rng.normal(size=(N_SAMP + 1, n_par))[np.cumsum(rng.random(N_SAMP) < prob)]


def geometric_pmf(prob):
    """Distribution of a geometric number of repeats."""
    return scipy.stats.geom.pmf(REPEATS, prob)


def mixed_draw(rng, n_par, rate, prob):
    """Draw where each step moves to a sample from a finite set of Gaussian samples with the given probability."""
    n_base = int(np.round(N_SAMP * prob / rate))
    visits = rng.integers(n_base, size=N_SAMP + 1)
    return rng.normal(size=(n_base, n_par))[visits[np.cumsum(rng.random(N_SAMP) < prob)]]


def mixed_pmf(rate, prob):
    """Distribution of the sum of a Poisson number of geometric repeats, given at least one."""
    rate = np.asarray(rate, dtype=float)
    prob = np.asarray(prob, dtype=float)
    stay = 1 - prob
    pmf = np.zeros((N_REPEAT_MAX + 1, *rate.shape))
    pmf[0] = np.exp(-rate)
    pmf[1] = rate * prob * pmf[0]
    for itrn in range(2, N_REPEAT_MAX + 1):
        pmf[itrn] = ((rate * prob + 2 * (itrn - 1) * stay) * pmf[itrn - 1] - (itrn - 2) * stay**2 * pmf[itrn - 2]) / itrn
    return pmf[1:] / -np.expm1(-rate)


FAMILIES = {
    'uniform': (uniform_draw, uniform_pmf),
    'poisson': (poisson_draw, poisson_pmf),
    'geometric': (geometric_draw, geometric_pmf),
    'mixed': (mixed_draw, mixed_pmf),
}


def expected_bias(pmf, log_mode):
    """Bias from repeats, if they are independent of position."""
    return np.log(REPEATS @ pmf) - CORRECTIONS[log_mode] @ pmf


def metropolis_draw(rng, n_par, n_base):
    """Draw from a Metropolis chain with uniform independent proposals from a finite set and a Gaussian target."""
    base = rng.uniform(-METROPOLIS_WIDTH / 2, METROPOLIS_WIDTH / 2, (n_base, n_par))
    log_density = -np.sum(base**2, axis=1) / 2
    density = np.exp(log_density)
    current = rng.choice(n_base, p=density / density.sum())
    proposals = rng.integers(n_base, size=N_SAMP)
    log_uniform = np.log(rng.random(N_SAMP))
    chain = np.zeros(N_SAMP, dtype=np.int64)
    for itrs in range(N_SAMP):
        if log_density[proposals[itrs]] - log_density[current] > log_uniform[itrs]:
            current = proposals[itrs]
        chain[itrs] = current
    return base[chain]


def metropolis_expected_bias(n_par, n_base, log_mode):
    """Bias from repeats in the Metropolis chain, averaged over the position dependent acceptance."""
    nodes, weights = np.polynomial.legendre.leggauss(40)
    half_radius_sq = scipy.special.gammaincinv(n_par / 2, (nodes + 1) / 2)
    inner = np.exp(n_par / 2 * np.log(2 * np.pi * half_radius_sq) - scipy.special.loggamma(n_par / 2 + 1))
    outer = (2 * np.pi) ** (n_par / 2) * np.exp(half_radius_sq) * scipy.special.gammaincc(n_par / 2, half_radius_sq)
    acceptance = (inner + outer) / METROPOLIS_WIDTH**n_par
    density = np.exp(-half_radius_sq) / (2 * np.pi) ** (n_par / 2)
    rate = N_SAMP * METROPOLIS_WIDTH**n_par * density * acceptance / n_base
    return np.sum(weights / 2 * expected_bias(mixed_pmf(rate, acceptance), log_mode))


@cache
def mean_bias(family, parameters, n_par):
    """Average bias of the entropy, cross-entropy, and KL divergences for each log mode."""
    rng = np.random.default_rng([SEED, n_par])
    draw = metropolis_draw if family == 'metropolis' else FAMILIES[family][0]
    results = {log_mode: np.zeros((N_TRIAL, 5)) for log_mode in LOG_MODES}
    for itrt in range(N_TRIAL):
        samples_unique = unique_draw(rng, n_par)
        samples_compare = draw(rng, n_par, *parameters)
        samples_test = draw(rng, n_par, *parameters)
        for log_mode in LOG_MODES:
            kl_test, e1, _ = kl_divergence_nn2(samples_unique, samples_test, log_mode=log_mode)
            kl_compare, _, e2 = kl_divergence_nn2(samples_compare, samples_unique, log_mode=log_mode)
            kl_both, _, _ = kl_divergence_nn2(samples_compare, samples_test, log_mode=log_mode)
            results[log_mode][itrt] = e1 - gaussian_entropy(n_par), e2 - gaussian_entropy(n_par), kl_test, kl_compare, kl_both
    return {log_mode: results[log_mode].mean(axis=0) for log_mode in LOG_MODES}


def repeat_and_shuffle(rng, samples, n_repeat):
    """Repeat each sample the same number of times in random order."""
    return rng.permutation(np.repeat(samples, n_repeat, axis=0))


def uniform_shift(n_unique, n_repeat, log_mode):
    """Exact entropy and cross-entropy shifts caused by the same number of repeats of each sample."""
    n_samp = n_unique * n_repeat
    correction = CORRECTIONS[log_mode][n_repeat - 1]
    entropy_shift = scipy.special.digamma(n_samp) - scipy.special.digamma(n_unique) - correction + np.log(n_samp - n_repeat) - np.log(n_samp - 1)
    cross_entropy_shift = scipy.special.digamma(n_samp + 1) - scipy.special.digamma(n_unique + 1) - correction
    return entropy_shift, cross_entropy_shift


@pytest.mark.parametrize('log_mode', LOG_MODES)
@pytest.mark.parametrize(('n_par', 'n_unique', 'n_repeat'), UNIFORM_EXACT_CASES)
def test_uniform_repeats_exact(n_par, n_unique, n_repeat, log_mode):
    """Repeating each sample the same number of times shifts the estimates by an exactly predictable amount."""
    rng = np.random.default_rng([SEED, n_par, n_unique, n_repeat])
    unique_compare = rng.normal(0.0, 2.0, (n_unique, n_par))
    unique_test = rng.normal(0.0, 1.0, (n_unique, n_par))
    samples_compare = repeat_and_shuffle(rng, unique_compare, n_repeat)
    samples_test = repeat_and_shuffle(rng, unique_test, n_repeat)
    divergence, e1, e2 = kl_divergence_nn2(samples_compare, samples_test, log_mode=log_mode)

    entropy_shift, cross_entropy_shift = uniform_shift(n_unique, n_repeat, log_mode)
    expected_e1 = entropy_nn1(unique_test, log_mode=log_mode) + entropy_shift
    expected_e2 = entropy_nn2(unique_compare, unique_test, log_mode=log_mode) + cross_entropy_shift
    assert e1 == pytest.approx(expected_e1, rel=1e-10)
    assert e2 == pytest.approx(expected_e2, rel=1e-10)
    assert divergence == pytest.approx(expected_e2 - expected_e1, rel=1e-9)


@pytest.mark.parametrize('duplicate_mode', [0, 1])
@pytest.mark.parametrize(('n_repeat', 'n_par'), GEO_MANY_CASES)
def test_geo_many_repeats(n_repeat, n_par, duplicate_mode):
    """The geo correction is consistent on both sides of the largest number of repeats tabulated."""
    n_self = 1000
    n_far = 500
    sample = np.full(n_par, 0.5)
    offsets = np.repeat([0.0, 2.0, 5.0], [n_self, n_repeat, n_far])
    samples_compare = sample + offsets[:, None]

    # computed directly because the whole estimators are quadratic in the number of samples
    result = _nn_corrected_log_distance(sample, samples_compare, duplicate_mode, 'geo')

    correction = scipy.special.digamma(n_repeat) + np.euler_gamma
    if duplicate_mode == 0:
        correction += np.log(n_self + n_repeat + n_far - 1) - np.log(n_repeat + n_far)
    expected = np.log(4.0 * n_par) - 2 / n_par * correction
    # allow for an asymptotic expansion of digamma, which is accurate to 1 / (2 * n_repeat) at leading order
    assert result == pytest.approx(expected, rel=0, abs=1.1 / (n_par * n_repeat))


@pytest.mark.parametrize('log_mode', LOG_MODES)
@pytest.mark.parametrize('n_par', [1, 3, 10])
def test_repeated_test_cross_entropy(n_par, log_mode):
    """Repeats in the test samples do not change the cross-entropy contribution of each sample."""
    n_unique = 50
    n_repeat = 4
    rng = np.random.default_rng([SEED, n_par])
    samples_compare = rng.normal(size=(n_unique * n_repeat, n_par))
    unique_test = rng.normal(size=(n_unique, n_par))
    samples_test = repeat_and_shuffle(rng, unique_test, n_repeat)
    per_sample = [entropy_nn2(samples_compare, np.tile(sample, (n_unique * n_repeat, 1)), log_mode=log_mode) for sample in unique_test]
    assert entropy_nn2(samples_compare, samples_test, log_mode=log_mode) == pytest.approx(np.mean(per_sample), rel=1e-10)


@pytest.mark.parametrize('prob', [0.5, 0.2, 0.05])
def test_expected_bias_geometric(prob):
    """The geo correction is predicted to be unbiased for geometric repeats."""
    assert expected_bias(geometric_pmf(prob), 'geo') == pytest.approx(0.0, abs=1e-10)
    assert expected_bias(geometric_pmf(prob), 'log') > 0.1


@pytest.mark.parametrize('rate', [0.5, 2.0, 5.0])
def test_expected_bias_mixed(rate):
    """The predicted geo bias for mixed repeats depends only on the Poisson rate."""
    expected = expected_bias(poisson_pmf(rate), 'geo')
    for prob in (0.8, 0.5, 0.2):
        assert expected_bias(mixed_pmf(rate, prob), 'geo') == pytest.approx(expected, rel=1e-8)


@pytest.mark.parametrize('log_mode', LOG_MODES)
@pytest.mark.parametrize(
    ('family', 'parameters', 'n_par', 'entropy_atol', 'cross_entropy_atol', 'kl_test_atol', 'kl_compare_atol', 'kl_both_atol'),
    CASES,
)
def test_repeat_bias(family, parameters, n_par, entropy_atol, cross_entropy_atol, kl_test_atol, kl_compare_atol, kl_both_atol, log_mode):
    """Mean bias for samples with repeats matches the predicted bias."""
    expected = expected_bias(FAMILIES[family][1](*parameters), log_mode)
    entropy_bias, cross_entropy_bias, kl_test, kl_compare, kl_both = mean_bias(family, parameters, n_par)[log_mode]
    assert entropy_bias == pytest.approx(expected, rel=0, abs=entropy_atol)
    assert cross_entropy_bias == pytest.approx(expected, rel=0, abs=cross_entropy_atol)
    assert kl_test == pytest.approx(-expected, rel=0, abs=kl_test_atol)
    assert kl_compare == pytest.approx(expected, rel=0, abs=kl_compare_atol)
    assert kl_both == pytest.approx(0.0, rel=0, abs=kl_both_atol)


@pytest.mark.parametrize(('family', 'parameters', 'n_par'), [case[:3] for case in CASES])
def test_repeat_log_mode_difference(family, parameters, n_par):
    """Difference between log modes matches the predicted difference."""
    pmf = FAMILIES[family][1](*parameters)
    expected = expected_bias(pmf, 'log') - expected_bias(pmf, 'geo')
    results = mean_bias(family, parameters, n_par)
    difference = results['log'] - results['geo']
    assert difference[:2] == pytest.approx(expected, rel=0, abs=0.012)


@pytest.mark.parametrize(('family', 'parameters', 'preferred'), PREFERRED_CASES)
def test_compare_preferred_log_mode(family, parameters, preferred):
    """The log mode predicted to be less biased is less biased for repeats in the compare samples."""
    other = 'geo' if preferred == 'log' else 'log'
    pmf = FAMILIES[family][1](*parameters)
    results = mean_bias(family, parameters, 1)
    assert abs(expected_bias(pmf, preferred)) < abs(expected_bias(pmf, other))
    assert np.all(np.abs(results[preferred][[1, 3]]) < np.abs(results[other][[1, 3]]))


@pytest.mark.parametrize(('family', 'parameters', 'preferred'), PREFERRED_WIDE_CASES)
def test_test_preferred_log_mode(family, parameters, preferred):
    """The log mode predicted to be less biased is less biased for repeats in the test samples."""
    other = 'geo' if preferred == 'log' else 'log'
    results = mean_bias(family, parameters, 1)
    assert np.all(np.abs(results[preferred][[0, 2]]) < np.abs(results[other][[0, 2]]))


@pytest.mark.parametrize(('n_base', 'preferred'), [(20000, 'geo'), (50, 'log')])
def test_metropolis_preferred_log_mode(n_base, preferred):
    """The log mode predicted to be less biased is less biased for a Metropolis chain."""
    other = 'geo' if preferred == 'log' else 'log'
    results = mean_bias('metropolis', (n_base,), 1)
    assert abs(metropolis_expected_bias(1, n_base, preferred)) < abs(metropolis_expected_bias(1, n_base, other))
    assert np.all(np.abs(results[preferred][:4]) < np.abs(results[other][:4]))


@pytest.mark.parametrize('log_mode', LOG_MODES)
@pytest.mark.parametrize(('n_par', 'n_base', 'entropy_atol', 'cross_entropy_atol', 'kl_atol'), METROPOLIS_CASES)
def test_metropolis_repeat_bias(n_par, n_base, entropy_atol, cross_entropy_atol, kl_atol, log_mode):
    """Mean bias for a Metropolis chain with position dependent repeats matches the predicted bias."""
    expected = metropolis_expected_bias(n_par, n_base, log_mode)
    entropy_bias, cross_entropy_bias, kl_test, _, _ = mean_bias('metropolis', (n_base,), n_par)[log_mode]
    assert entropy_bias == pytest.approx(expected, rel=0, abs=entropy_atol)
    assert cross_entropy_bias == pytest.approx(expected, rel=0, abs=cross_entropy_atol)
    assert kl_test == pytest.approx(-expected, rel=0, abs=kl_atol)
