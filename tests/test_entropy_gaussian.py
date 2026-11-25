# SPDX-FileCopyrightText: Copyright 2026 Matthew C. Digman
# SPDX-License-Identifier: Apache-2.0
"""Tests of the nearest neighbor estimators against analytic Gaussian results."""

import numpy as np
import pytest

from EntropyTools.entropy_process import entropy_nn1, entropy_nn2, kl_divergence_nn2

SEED = 31415

# Tolerances are approximately the finite-sample bias plus four standard deviations.
ENTROPY_ATOLS = {
    (1, 100): 0.67,
    (1, 1000): 0.21,
    (1, 2000): 0.16,
    (1, 10000): 0.07,
    (2, 100): 0.76,
    (2, 500): 0.34,
    (2, 1000): 0.25,
    (2, 2000): 0.18,
    (2, 10000): 0.08,
    (3, 100): 0.88,
    (3, 500): 0.39,
    (3, 1000): 0.28,
    (3, 2000): 0.20,
    (3, 10000): 0.09,
    (5, 100): 0.98,
    (5, 500): 0.47,
    (5, 1000): 0.36,
    (5, 2000): 0.27,
    (5, 10000): 0.13,
    (10, 100): 1.36,
    (10, 1000): 0.39,
    (10, 2000): 0.27,
    (10, 10000): 0.16,
    (20, 1000): 1.38,
    (20, 10000): 0.67,
}

SAME_CASES = [
    (1, 100, 0.66, 0.83),
    (1, 1000, 0.20, 0.27),
    (1, 10000, 0.07, 0.09),
    (2, 100, 0.68, 0.81),
    (2, 1000, 0.24, 0.28),
    (2, 10000, 0.07, 0.09),
    (3, 100, 0.75, 0.84),
    (3, 1000, 0.26, 0.28),
    (3, 10000, 0.09, 0.09),
    (5, 100, 0.84, 0.87),
    (5, 1000, 0.31, 0.28),
    (5, 10000, 0.12, 0.09),
    (10, 100, 1.12, 1.00),
    (10, 1000, 0.32, 0.30),
    (10, 10000, 0.14, 0.10),
    (20, 1000, 1.27, 0.39),
    (20, 10000, 0.64, 0.11),
]

DIFFERENT_CASES = [
    # same width, offset mean
    (1, 2000, 1.0, 1.0, 1.0, 0.23, 0.25),
    (1, 10000, 1.0, 1.0, 1.0, 0.11, 0.12),
    (2, 2000, 1.0, 1.0, 1.0, 0.32, 0.33),
    (2, 10000, 1.0, 1.0, 1.0, 0.16, 0.17),
    (3, 10000, 1.0, 1.0, 0.5, 0.12, 0.12),
    (5, 10000, 1.0, 1.0, 0.5, 0.18, 0.17),
    # test narrower than compare
    (1, 2000, 0.5, 2.0, 0.0, 0.20, 0.25),
    (1, 10000, 0.5, 2.0, 0.0, 0.10, 0.11),
    (1, 10000, 1.0, 2.0, 0.0, 0.08, 0.10),
    (2, 500, 0.5, 2.0, 0.0, 0.59, 0.67),
    (2, 2000, 1.0, 2.0, 0.0, 0.17, 0.23),
    (2, 10000, 1.0, 2.0, 0.0, 0.08, 0.11),
    (3, 500, 0.5, 2.0, 0.0, 0.91, 1.00),
    (3, 2000, 1.0, 2.0, 0.0, 0.21, 0.27),
    (3, 10000, 1.0, 2.0, 0.0, 0.10, 0.11),
    (5, 500, 1.0, 2.0, 0.0, 0.63, 0.68),
    (5, 2000, 1.0, 2.0, 0.0, 0.33, 0.42),
    (5, 10000, 1.0, 2.0, 0.0, 0.15, 0.20),
    (10, 2000, 1.0, 2.0, 0.0, 0.97, 1.00),
    (10, 10000, 1.0, 2.0, 0.0, 0.60, 0.65),
    # test wider than compare
    (1, 10000, 1.5, 1.0, 0.0, 0.10, 0.10),
    (2, 10000, 1.5, 1.0, 0.0, 0.18, 0.17),
    (3, 10000, 1.5, 1.0, 0.0, 0.29, 0.28),
    # different width, offset mean
    (1, 2000, 1.0, 2.0, 1.0, 0.17, 0.22),
    (2, 2000, 1.0, 2.0, 1.0, 0.20, 0.25),
    (3, 2000, 1.0, 2.0, 1.0, 0.24, 0.28),
    (5, 2000, 1.0, 2.0, 1.0, 0.41, 0.51),
]


def gaussian_entropy(n_par, sigma):
    """Entropy of an isotropic Gaussian."""
    return n_par / 2 * np.log(2 * np.pi * np.e * sigma**2)


def gaussian_kl(n_par, sigma_test, sigma_compare, offset=0.0):
    """KL divergence of the test Gaussian from the compare Gaussian."""
    variance_ratio = (sigma_test / sigma_compare) ** 2
    return n_par / 2 * (variance_ratio - 1 - np.log(variance_ratio) + (offset / sigma_compare) ** 2)


def gaussian_cross_entropy(n_par, sigma_test, sigma_compare, offset=0.0):
    """Cross-entropy of the test Gaussian relative to the compare Gaussian."""
    return gaussian_entropy(n_par, sigma_test) + gaussian_kl(n_par, sigma_test, sigma_compare, offset)


def draw_pair(n_par, n_samp, sigma_test, sigma_compare, offset=0.0):
    """Draw independent compare and test samples."""
    rng = np.random.default_rng([SEED, n_par, n_samp])
    samples_compare = rng.normal(0.0, sigma_compare, (n_samp, n_par))
    samples_test = rng.normal(offset, sigma_test, (n_samp, n_par))
    return samples_compare, samples_test


@pytest.mark.parametrize('sigma', [0.1, 1.0, 3.0])
@pytest.mark.parametrize(('n_par', 'n_samp'), [case[:2] for case in SAME_CASES])
def test_entropy_nn1(n_par, n_samp, sigma):
    """One sample entropy matches the analytic entropy."""
    _, samples = draw_pair(n_par, n_samp, sigma, sigma)
    expected = gaussian_entropy(n_par, sigma)
    assert entropy_nn1(samples) == pytest.approx(expected, rel=0, abs=ENTROPY_ATOLS[n_par, n_samp])


@pytest.mark.parametrize('sigma', [0.1, 1.0, 3.0])
@pytest.mark.parametrize(('n_par', 'n_samp', 'cross_entropy_atol', '_kl_atol'), SAME_CASES)
def test_entropy_nn2_same(n_par, n_samp, cross_entropy_atol, _kl_atol, sigma):
    """Two sample entropy matches the analytic entropy for identical distributions."""
    samples_compare, samples_test = draw_pair(n_par, n_samp, sigma, sigma)
    expected = gaussian_entropy(n_par, sigma)
    assert entropy_nn2(samples_compare, samples_test) == pytest.approx(expected, rel=0, abs=cross_entropy_atol)


@pytest.mark.parametrize(
    ('n_par', 'n_samp', 'sigma_test', 'sigma_compare', 'offset', 'cross_entropy_atol', '_kl_atol'),
    DIFFERENT_CASES,
)
def test_entropy_nn2_different(n_par, n_samp, sigma_test, sigma_compare, offset, cross_entropy_atol, _kl_atol):
    """Two sample entropy matches the analytic cross-entropy for different distributions."""
    samples_compare, samples_test = draw_pair(n_par, n_samp, sigma_test, sigma_compare, offset)
    expected = gaussian_cross_entropy(n_par, sigma_test, sigma_compare, offset)
    assert entropy_nn2(samples_compare, samples_test) == pytest.approx(expected, rel=0, abs=cross_entropy_atol)


@pytest.mark.parametrize('sigma', [0.1, 1.0, 3.0])
@pytest.mark.parametrize(('n_par', 'n_samp', 'cross_entropy_atol', 'kl_atol'), SAME_CASES)
def test_kl_divergence_same(n_par, n_samp, cross_entropy_atol, kl_atol, sigma):
    """KL divergence vanishes for identical distributions."""
    samples_compare, samples_test = draw_pair(n_par, n_samp, sigma, sigma)
    divergence, e1, e2 = kl_divergence_nn2(samples_compare, samples_test)
    expected = gaussian_entropy(n_par, sigma)
    assert divergence == pytest.approx(0.0, rel=0, abs=kl_atol)
    assert e1 == pytest.approx(expected, rel=0, abs=ENTROPY_ATOLS[n_par, n_samp])
    assert e2 == pytest.approx(expected, rel=0, abs=cross_entropy_atol)


@pytest.mark.parametrize(
    ('n_par', 'n_samp', 'sigma_test', 'sigma_compare', 'offset', 'cross_entropy_atol', 'kl_atol'),
    DIFFERENT_CASES,
)
def test_kl_divergence_different(n_par, n_samp, sigma_test, sigma_compare, offset, cross_entropy_atol, kl_atol):
    """KL divergence matches the analytic value for different distributions."""
    samples_compare, samples_test = draw_pair(n_par, n_samp, sigma_test, sigma_compare, offset)
    divergence, e1, e2 = kl_divergence_nn2(samples_compare, samples_test)
    expected_kl = gaussian_kl(n_par, sigma_test, sigma_compare, offset)
    expected_entropy = gaussian_entropy(n_par, sigma_test)
    expected_cross_entropy = gaussian_cross_entropy(n_par, sigma_test, sigma_compare, offset)
    assert kl_atol < expected_kl
    assert divergence == pytest.approx(expected_kl, rel=0, abs=kl_atol)
    assert e1 == pytest.approx(expected_entropy, rel=0, abs=ENTROPY_ATOLS[n_par, n_samp])
    assert e2 == pytest.approx(expected_cross_entropy, rel=0, abs=cross_entropy_atol)


@pytest.mark.parametrize(
    ('n_par', 'entropy_atol', 'cross_entropy_atol', 'kl_atol'),
    [(1, 0.10, 0.09, 0.06), (2, 0.11, 0.11, 0.07), (3, 0.07, 0.08, 0.09)],
)
def test_small_sample_mean(n_par, entropy_atol, cross_entropy_atol, kl_atol):
    """Estimates averaged over many small sample sets match the analytic values."""
    n_samp = 5
    n_trial = 4000
    rng = np.random.default_rng([SEED, n_par])
    results = np.zeros((n_trial, 3))
    for itrt in range(n_trial):
        samples_compare = rng.normal(0.0, 1.0, (n_samp, n_par))
        samples_test = rng.normal(0.0, 1.0, (n_samp, n_par))
        results[itrt] = kl_divergence_nn2(samples_compare, samples_test)
    expected = gaussian_entropy(n_par, 1.0)
    divergence, e1, e2 = results.mean(axis=0)
    assert divergence == pytest.approx(0.0, rel=0, abs=kl_atol)
    assert e1 == pytest.approx(expected, rel=0, abs=entropy_atol)
    assert e2 == pytest.approx(expected, rel=0, abs=cross_entropy_atol)


@pytest.mark.parametrize('scale', [0.01, 7.0])
@pytest.mark.parametrize('n_par', [1, 3, 10, 50])
def test_scaling(n_par, scale):
    """Rescaling all samples shifts the entropies by a known constant and preserves the KL divergence."""
    samples_compare, samples_test = draw_pair(n_par, 500, 1.0, 2.0)
    divergence, e1, e2 = kl_divergence_nn2(samples_compare, samples_test)
    divergence_scaled, e1_scaled, e2_scaled = kl_divergence_nn2(scale * samples_compare, scale * samples_test)
    shift = n_par * np.log(scale)
    assert divergence_scaled == pytest.approx(divergence, rel=1e-9)
    assert e1_scaled - e1 == pytest.approx(shift, rel=1e-9)
    assert e2_scaled - e2 == pytest.approx(shift, rel=1e-9)


@pytest.mark.parametrize(('n_par', 'n_samp'), [(1, 100), (3, 1000), (10, 1000)])
def test_kl_divergence_components(n_par, n_samp):
    """KL divergence components match the standalone entropy estimates."""
    samples_compare, samples_test = draw_pair(n_par, n_samp, 1.0, 2.0)
    divergence, e1, e2 = kl_divergence_nn2(samples_compare, samples_test)
    assert e1 == pytest.approx(entropy_nn1(samples_test), rel=1e-12)
    assert e2 == pytest.approx(entropy_nn2(samples_compare, samples_test), rel=1e-12)
    assert divergence == pytest.approx(e2 - e1, rel=1e-12)


@pytest.mark.parametrize(('n_par', 'n_samp'), [(1, 100), (3, 1000), (10, 1000)])
def test_log_mode_no_repeats(n_par, n_samp):
    """The log mode has no effect without repeated samples."""
    samples_compare, samples_test = draw_pair(n_par, n_samp, 1.0, 2.0)
    result_geo = kl_divergence_nn2(samples_compare, samples_test, log_mode='geo')
    result_log = kl_divergence_nn2(samples_compare, samples_test, log_mode='log')
    np.testing.assert_allclose(result_geo, result_log, rtol=1e-12)
    assert entropy_nn1(samples_test, log_mode='geo') == pytest.approx(entropy_nn1(samples_test, log_mode='log'), rel=1e-12)
    assert entropy_nn2(samples_compare, samples_test, log_mode='geo') == pytest.approx(entropy_nn2(samples_compare, samples_test, log_mode='log'), rel=1e-12)
