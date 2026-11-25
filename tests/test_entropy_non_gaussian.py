# SPDX-FileCopyrightText: Copyright 2026 Matthew C. Digman
# SPDX-License-Identifier: Apache-2.0
"""Tests of the nearest neighbor estimators against analytic non-Gaussian results."""

import numpy as np
import pytest

from EntropyTools.entropy_process import entropy_nn1, entropy_nn2, kl_divergence_nn2

SEED = 31415

# Tolerances are approximately the finite-sample bias plus four standard deviations.
# family, n_par, n_samp, theta, entropy atol, cross-entropy atol, KL divergence atol
SAME_CASES = [
    ('exponential', 1, 2000, 0.5, 0.17, 0.17, 0.19),
    ('exponential', 1, 10000, 0.5, 0.07, 0.07, 0.09),
    ('exponential', 2, 2000, 0.5, 0.20, 0.18, 0.20),
    ('exponential', 2, 10000, 0.5, 0.09, 0.09, 0.09),
    ('exponential', 3, 2000, 0.5, 0.29, 0.26, 0.19),
    ('exponential', 3, 10000, 0.5, 0.14, 0.14, 0.10),
    ('banana', 2, 2000, 1.0, 0.17, 0.17, 0.19),
    ('banana', 2, 10000, 1.0, 0.08, 0.07, 0.09),
    ('banana', 3, 2000, 1.0, 0.18, 0.16, 0.19),
    ('banana', 3, 10000, 1.0, 0.09, 0.10, 0.10),
    ('banana', 5, 2000, 1.0, 0.32, 0.31, 0.21),
    ('banana', 5, 10000, 1.0, 0.14, 0.12, 0.09),
    ('two_box', 1, 2000, 0.3, 0.14, 0.13, 0.19),
    ('two_box', 1, 10000, 0.3, 0.07, 0.06, 0.09),
    ('two_box', 2, 2000, 0.3, 0.16, 0.15, 0.20),
    ('two_box', 2, 10000, 0.3, 0.08, 0.07, 0.08),
    ('two_box', 3, 2000, 0.3, 0.24, 0.20, 0.19),
    ('two_box', 3, 10000, 0.3, 0.12, 0.11, 0.09),
    ('two_box', 5, 10000, 0.3, 0.32, 0.31, 0.10),
]

# family, n_par, n_samp, test theta, compare theta, entropy atol, cross-entropy atol, KL divergence atol
DIFFERENT_CASES = [
    # test narrower than compare
    ('exponential', 1, 2000, 0.5, 2.0, 0.17, 0.19, 0.24),
    ('exponential', 1, 10000, 0.5, 2.0, 0.07, 0.08, 0.12),
    ('exponential', 2, 2000, 0.5, 2.0, 0.20, 0.27, 0.31),
    ('exponential', 2, 10000, 0.5, 2.0, 0.09, 0.14, 0.16),
    ('exponential', 3, 2000, 0.5, 2.0, 0.29, 0.53, 0.51),
    ('exponential', 3, 10000, 0.5, 2.0, 0.14, 0.24, 0.23),
    # test wider than compare
    ('exponential', 1, 2000, 2.0, 1.0, 0.17, 0.26, 0.24),
    ('exponential', 1, 10000, 2.0, 1.0, 0.07, 0.12, 0.12),
    ('exponential', 2, 2000, 2.0, 1.0, 0.20, 0.35, 0.33),
    ('exponential', 2, 10000, 2.0, 1.0, 0.09, 0.19, 0.17),
    ('exponential', 3, 2000, 2.0, 1.0, 0.29, 0.43, 0.43),
    ('exponential', 3, 10000, 2.0, 1.0, 0.14, 0.20, 0.21),
    # test narrower than compare
    ('banana', 2, 2000, 1.0, 2.0, 0.17, 0.18, 0.22),
    ('banana', 2, 10000, 1.0, 2.0, 0.08, 0.07, 0.10),
    ('banana', 3, 2000, 1.0, 2.0, 0.18, 0.23, 0.29),
    ('banana', 3, 10000, 1.0, 2.0, 0.09, 0.09, 0.13),
    ('banana', 5, 2000, 1.0, 2.0, 0.32, 0.46, 0.42),
    ('banana', 5, 10000, 1.0, 2.0, 0.14, 0.22, 0.21),
    # test more concentrated than compare
    ('two_box', 1, 2000, 0.1, 0.5, 0.15, 0.15, 0.22),
    ('two_box', 1, 10000, 0.1, 0.5, 0.07, 0.07, 0.10),
    ('two_box', 2, 2000, 0.1, 0.5, 0.17, 0.17, 0.21),
    ('two_box', 2, 10000, 0.1, 0.5, 0.08, 0.07, 0.10),
    ('two_box', 3, 2000, 0.1, 0.5, 0.24, 0.24, 0.25),
    ('two_box', 3, 10000, 0.1, 0.5, 0.11, 0.11, 0.10),
    ('two_box', 5, 10000, 0.1, 0.5, 0.31, 0.32, 0.11),
    # test less concentrated than compare
    ('two_box', 1, 2000, 0.5, 0.1, 0.14, 0.22, 0.26),
    ('two_box', 1, 10000, 0.5, 0.1, 0.06, 0.10, 0.12),
    ('two_box', 2, 2000, 0.5, 0.1, 0.16, 0.23, 0.25),
    ('two_box', 2, 10000, 0.5, 0.1, 0.08, 0.10, 0.10),
    ('two_box', 3, 2000, 0.5, 0.1, 0.23, 0.32, 0.27),
    ('two_box', 3, 10000, 0.5, 0.1, 0.12, 0.15, 0.13),
    ('two_box', 5, 10000, 0.5, 0.1, 0.32, 0.38, 0.15),
]


def exponential_draw(rng, n_par, n_samp, scale):
    """Draw independent exponential samples."""
    return rng.exponential(scale, (n_samp, n_par))


def exponential_entropy(n_par, scale):
    """Entropy of independent exponentials."""
    return n_par * (1 + np.log(scale))


def exponential_kl(n_par, scale_test, scale_compare):
    """KL divergence of the test exponentials from the compare exponentials."""
    ratio = scale_test / scale_compare
    return n_par * (ratio - 1 - np.log(ratio))


def banana_draw(rng, n_par, n_samp, sigma):
    """Draw Gaussian samples and apply a volume preserving quadratic shear."""
    samples = rng.normal(0.0, sigma, (n_samp, n_par))
    samples[:, 1:] += samples[:, :1] ** 2
    return samples


def banana_entropy(n_par, sigma):
    """Entropy of the sheared Gaussian, unchanged by the shear."""
    return n_par / 2 * np.log(2 * np.pi * np.e * sigma**2)


def banana_kl(n_par, sigma_test, sigma_compare):
    """KL divergence of the test sheared Gaussian from the compare, unchanged by the shear."""
    variance_ratio = (sigma_test / sigma_compare) ** 2
    return n_par / 2 * (variance_ratio - 1 - np.log(variance_ratio))


def two_box_draw(rng, n_par, n_samp, weight):
    """Draw samples from two separated unit boxes, with the given weight in the second box."""
    samples = rng.uniform(-0.5, 0.5, (n_samp, n_par))
    return samples + 10.0 * (rng.random((n_samp, 1)) < weight)


def two_box_entropy(_n_par, weight):
    """Entropy of two separated unit boxes."""
    return -weight * np.log(weight) - (1 - weight) * np.log(1 - weight)


def two_box_kl(_n_par, weight_test, weight_compare):
    """KL divergence of the test box weights from the compare box weights."""
    return weight_test * np.log(weight_test / weight_compare) + (1 - weight_test) * np.log((1 - weight_test) / (1 - weight_compare))


FAMILIES = {
    'exponential': (exponential_draw, exponential_entropy, exponential_kl),
    'banana': (banana_draw, banana_entropy, banana_kl),
    'two_box': (two_box_draw, two_box_entropy, two_box_kl),
}


def draw_pair(family, n_par, n_samp, theta_test, theta_compare):
    """Draw independent compare and test samples."""
    rng = np.random.default_rng([SEED, n_par, n_samp])
    draw, _, _ = FAMILIES[family]
    samples_compare = draw(rng, n_par, n_samp, theta_compare)
    samples_test = draw(rng, n_par, n_samp, theta_test)
    return samples_compare, samples_test


def expected_values(family, n_par, theta_test, theta_compare):
    """Get the analytic KL divergence, entropy, and cross-entropy."""
    _, entropy, kl = FAMILIES[family]
    expected_kl = kl(n_par, theta_test, theta_compare)
    expected_entropy = entropy(n_par, theta_test)
    return expected_kl, expected_entropy, expected_entropy + expected_kl


@pytest.mark.parametrize(('family', 'n_par', 'n_samp', 'theta', 'entropy_atol'), [case[:5] for case in SAME_CASES])
def test_entropy_nn1(family, n_par, n_samp, theta, entropy_atol):
    """One sample entropy matches the analytic entropy."""
    _, samples = draw_pair(family, n_par, n_samp, theta, theta)
    _, expected, _ = expected_values(family, n_par, theta, theta)
    assert entropy_nn1(samples) == pytest.approx(expected, rel=0, abs=entropy_atol)


@pytest.mark.parametrize(
    ('family', 'n_par', 'n_samp', 'theta', '_entropy_atol', 'cross_entropy_atol', '_kl_atol'),
    SAME_CASES,
)
def test_entropy_nn2_same(family, n_par, n_samp, theta, _entropy_atol, cross_entropy_atol, _kl_atol):
    """Two sample entropy matches the analytic entropy for identical distributions."""
    samples_compare, samples_test = draw_pair(family, n_par, n_samp, theta, theta)
    _, expected, _ = expected_values(family, n_par, theta, theta)
    assert entropy_nn2(samples_compare, samples_test) == pytest.approx(expected, rel=0, abs=cross_entropy_atol)


@pytest.mark.parametrize(
    ('family', 'n_par', 'n_samp', 'theta_test', 'theta_compare', '_entropy_atol', 'cross_entropy_atol', '_kl_atol'),
    DIFFERENT_CASES,
)
def test_entropy_nn2_different(family, n_par, n_samp, theta_test, theta_compare, _entropy_atol, cross_entropy_atol, _kl_atol):
    """Two sample entropy matches the analytic cross-entropy for different distributions."""
    samples_compare, samples_test = draw_pair(family, n_par, n_samp, theta_test, theta_compare)
    _, _, expected = expected_values(family, n_par, theta_test, theta_compare)
    assert entropy_nn2(samples_compare, samples_test) == pytest.approx(expected, rel=0, abs=cross_entropy_atol)


@pytest.mark.parametrize(
    ('family', 'n_par', 'n_samp', 'theta', 'entropy_atol', 'cross_entropy_atol', 'kl_atol'),
    SAME_CASES,
)
def test_kl_divergence_same(family, n_par, n_samp, theta, entropy_atol, cross_entropy_atol, kl_atol):
    """KL divergence vanishes for identical distributions."""
    samples_compare, samples_test = draw_pair(family, n_par, n_samp, theta, theta)
    divergence, e1, e2 = kl_divergence_nn2(samples_compare, samples_test)
    _, expected, _ = expected_values(family, n_par, theta, theta)
    assert divergence == pytest.approx(0.0, rel=0, abs=kl_atol)
    assert e1 == pytest.approx(expected, rel=0, abs=entropy_atol)
    assert e2 == pytest.approx(expected, rel=0, abs=cross_entropy_atol)


@pytest.mark.parametrize(
    ('family', 'n_par', 'n_samp', 'theta_test', 'theta_compare', 'entropy_atol', 'cross_entropy_atol', 'kl_atol'),
    DIFFERENT_CASES,
)
def test_kl_divergence_different(family, n_par, n_samp, theta_test, theta_compare, entropy_atol, cross_entropy_atol, kl_atol):
    """KL divergence matches the analytic value for different distributions."""
    samples_compare, samples_test = draw_pair(family, n_par, n_samp, theta_test, theta_compare)
    divergence, e1, e2 = kl_divergence_nn2(samples_compare, samples_test)
    expected_kl, expected_entropy, expected_cross_entropy = expected_values(family, n_par, theta_test, theta_compare)
    assert kl_atol < expected_kl
    assert divergence == pytest.approx(expected_kl, rel=0, abs=kl_atol)
    assert e1 == pytest.approx(expected_entropy, rel=0, abs=entropy_atol)
    assert e2 == pytest.approx(expected_cross_entropy, rel=0, abs=cross_entropy_atol)
