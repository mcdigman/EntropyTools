# SPDX-FileCopyrightText: Copyright 2026 Matthew C. Digman
# SPDX-License-Identifier: Apache-2.0
"""Tests of the nearest neighbor estimators against analytic results and predicted edge effects for uniform cubes and balls."""

from functools import cache

import numpy as np
import pytest
import scipy.integrate
import scipy.special

from EntropyTools.entropy_process import entropy_nn1, entropy_nn2, kl_divergence_nn2

SEED = 31415

# total number of draws averaged over when measuring edge effects
N_DRAW = 200000

# Single draw tolerances are approximately the predicted edge bias plus four standard deviations.
# shape, n_par, n_samp, size, entropy atol, cross-entropy atol, KL divergence atol
SAME_CASES = [
    ('cube', 1, 1000, 1.0, 0.19, 0.18, 0.26),
    ('cube', 1, 10000, 2.0, 0.07, 0.06, 0.09),
    ('cube', 2, 1000, 1.0, 0.20, 0.20, 0.27),
    ('cube', 2, 10000, 2.0, 0.07, 0.06, 0.08),
    ('cube', 3, 1000, 1.0, 0.29, 0.27, 0.27),
    ('cube', 3, 10000, 2.0, 0.11, 0.10, 0.09),
    ('ball', 1, 1000, 1.0, 0.18, 0.18, 0.25),
    ('ball', 1, 10000, 2.0, 0.06, 0.07, 0.09),
    ('ball', 2, 1000, 1.0, 0.21, 0.19, 0.25),
    ('ball', 2, 10000, 2.0, 0.06, 0.06, 0.08),
    ('ball', 3, 1000, 1.0, 0.29, 0.25, 0.28),
    ('ball', 3, 10000, 2.0, 0.11, 0.09, 0.09),
]

# test shape, compare shape, n_par, n_samp, test size, compare size, KL divergence atol
NESTED_CASES = [
    ('cube', 'cube', 1, 10000, 1.0, 2.0, 0.09),
    ('cube', 'cube', 2, 10000, 1.0, 2.0, 0.13),
    ('cube', 'cube', 3, 10000, 1.0, 2.0, 0.18),
    ('ball', 'ball', 1, 10000, 1.0, 2.0, 0.10),
    ('ball', 'ball', 2, 10000, 1.0, 2.0, 0.12),
    ('ball', 'ball', 3, 10000, 1.0, 2.0, 0.16),
    ('ball', 'cube', 2, 10000, 1.0, 2.0, 0.09),
    ('ball', 'cube', 3, 10000, 1.0, 2.0, 0.12),
    ('cube', 'ball', 2, 10000, 1.0, np.sqrt(2) / 2, 0.10),
    ('cube', 'ball', 3, 10000, 1.0, np.sqrt(3) / 2, 0.14),
]

# Averaged tolerances are approximately four standard errors.
# n_par, n_samp, n_trial, entropy atol, cross-entropy atol, KL divergence atol
BALL_CASES = [
    (1, 10, 5000, 0.027, 0.025, 0.037),
    (2, 100, 4000, 0.010, 0.009, 0.014),
    (3, 100, 4000, 0.011, 0.010, 0.014),
    (3, 1000, 400, 0.010, 0.009, 0.013),
    (5, 100, 4000, 0.012, 0.010, 0.015),
    (5, 2000, 100, 0.014, 0.014, 0.019),
    (10, 100, 2000, 0.019, 0.015, 0.023),
    (10, 2000, 100, 0.018, 0.015, 0.023),
    (20, 100, 2000, 0.022, 0.017, 0.027),
    (20, 1000, 100, 0.031, 0.022, 0.034),
    (50, 100, 1000, 0.040, 0.029, 0.047),
    (50, 1000, 50, 0.055, 0.036, 0.063),
]

# n_par, n_samp, n_trial, compare radius, KL divergence atol
NESTED_BALL_CASES = [
    (2, 100, 4000, 2.0, 0.017),
    (3, 1000, 400, 1.2, 0.015),
    (3, 1000, 400, 2.0, 0.022),
    (5, 1000, 400, 1.2, 0.016),
    (5, 1000, 400, 2.0, 0.029),
    (10, 1000, 200, 1.2, 0.026),
    (10, 1000, 200, 2.0, 0.067),
    (20, 1000, 100, 1.2, 0.039),
]

# n_par, small n_samp, large n_samp, exponent atol
SCALING_CASES = [
    (2, 30, 400, 0.14),
    (3, 100, 3200, 0.09),
    (5, 100, 6400, 0.05),
]


def log_ball_volume(n_par):
    """Log volume of a unit ball."""
    return n_par / 2 * np.log(np.pi) - scipy.special.loggamma(n_par / 2 + 1)


def cube_draw(rng, n_par, n_samp, width):
    """Draw uniformly from a centered cube."""
    return rng.uniform(-width / 2, width / 2, (n_samp, n_par))


def cube_log_volume(n_par, width):
    """Log volume of a cube."""
    return n_par * np.log(width)


def ball_draw(rng, n_par, n_samp, radius):
    """Draw uniformly from a centered ball."""
    directions = rng.normal(size=(n_samp, n_par))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return radius * directions * rng.random((n_samp, 1)) ** (1 / n_par)


def ball_log_volume(n_par, radius):
    """Log volume of a ball."""
    return log_ball_volume(n_par) + n_par * np.log(radius)


SHAPES = {
    'cube': (cube_draw, cube_log_volume),
    'ball': (ball_draw, ball_log_volume),
}


def cap_fraction(n_par, height):
    """Fraction of a unit ball beyond a plane at a signed distance from the center."""
    fraction = scipy.special.betainc((n_par + 1) / 2, 0.5, np.clip(1 - height**2, 0.0, 1.0)) / 2
    return np.where(height >= 0.0, fraction, 1 - fraction)


def ball_overlap(n_par, center, radius, radius_compare):
    """Fraction of the compare ball within the given radius of a point at the given distance from its center."""
    plane = (center**2 + radius_compare**2 - radius**2) / (2 * center)
    volume_ratio = (radius / radius_compare) ** n_par
    overlap = cap_fraction(n_par, plane / radius_compare) + volume_ratio * cap_fraction(n_par, (center - plane) / radius)
    overlap = np.where(radius <= radius_compare - center, volume_ratio, overlap)
    return np.where(radius >= radius_compare + center, 1.0, overlap)


def ball_edge_bias(n_par, n_neighbor, radius_test, radius_compare):
    """Exact expected bias of the estimator for a ball inside a ball."""
    nodes, weights = np.polynomial.legendre.leggauss(40)
    centers = radius_test * ((nodes + 1) / 2) ** (1 / n_par)
    log_radius = np.zeros(centers.size)
    for itrc, center in enumerate(centers):
        log_low = np.log(radius_compare) - (np.log(n_neighbor) + 35) / n_par
        log_grid = np.linspace(log_low, np.log(radius_compare + center), 4000)
        overlap = ball_overlap(n_par, center, np.exp(log_grid), radius_compare)
        survival = np.exp(n_neighbor * np.log1p(-np.minimum(overlap, 1 - 1e-16)))
        log_radius[itrc] = log_low + scipy.integrate.trapezoid(survival, log_grid)
    expected_log_volume = n_par * np.sum(weights / 2 * log_radius)
    return expected_log_volume - n_par * np.log(radius_compare) + scipy.special.digamma(n_neighbor + 1) + np.euler_gamma


def leading_edge_bias(shape, n_par, n_samp):
    """Leading order predicted bias from a flat edge, proportional to n_samp**(-1 / n_par)."""
    slope = np.exp(log_ball_volume(n_par - 1) - log_ball_volume(n_par))

    def integrand(height):
        interior = 1 - cap_fraction(n_par, height)
        return height * slope * (1 - height**2) ** ((n_par - 1) / 2) * interior ** (-1 - 1 / n_par)

    flat_edge = scipy.integrate.quad(integrand, 0.0, 1.0)[0]
    surface_to_volume = {'ball': n_par, 'cube': 2 * n_par * np.exp(-log_ball_volume(n_par) / n_par)}[shape]
    return surface_to_volume * scipy.special.gamma(1 + 1 / n_par) * flat_edge * n_samp ** (-1 / n_par)


def draw_pair(shape_test, shape_compare, n_par, n_samp, size_test, size_compare):
    """Draw independent compare and test samples."""
    rng = np.random.default_rng([SEED, n_par, n_samp])
    samples_compare = SHAPES[shape_compare][0](rng, n_par, n_samp, size_compare)
    samples_test = SHAPES[shape_test][0](rng, n_par, n_samp, size_test)
    return samples_compare, samples_test


@cache
def mean_bias(shape, n_par, n_samp, n_trial, size_compare=1.0):
    """Average bias of the KL divergence, entropy, and cross-entropy for a unit shape inside a compare shape."""
    rng = np.random.default_rng([SEED, n_par, n_samp, n_trial])
    draw, log_volume = SHAPES[shape]
    results = np.zeros((n_trial, 3))
    for itrt in range(n_trial):
        samples_compare = draw(rng, n_par, n_samp, size_compare)
        samples_test = draw(rng, n_par, n_samp, 1.0)
        results[itrt] = kl_divergence_nn2(samples_compare, samples_test)
    expected_entropy = log_volume(n_par, 1.0)
    expected_cross_entropy = log_volume(n_par, size_compare)
    return tuple(results.mean(axis=0) - [expected_cross_entropy - expected_entropy, expected_entropy, expected_cross_entropy])


def mean_entropy_bias(shape, n_par, n_samp):
    """Average entropy bias using a fixed total number of draws."""
    _, entropy_bias, cross_entropy_bias = mean_bias(shape, n_par, n_samp, int(np.ceil(N_DRAW / n_samp)))
    return (entropy_bias + cross_entropy_bias) / 2


@pytest.mark.parametrize(('shape', 'n_par', 'n_samp', 'size', 'entropy_atol'), [case[:5] for case in SAME_CASES])
def test_entropy_nn1(shape, n_par, n_samp, size, entropy_atol):
    """One sample entropy matches the log volume."""
    _, samples = draw_pair(shape, shape, n_par, n_samp, size, size)
    expected = SHAPES[shape][1](n_par, size)
    assert entropy_nn1(samples) == pytest.approx(expected, rel=0, abs=entropy_atol)


@pytest.mark.parametrize(
    ('shape', 'n_par', 'n_samp', 'size', '_entropy_atol', 'cross_entropy_atol', '_kl_atol'),
    SAME_CASES,
)
def test_entropy_nn2(shape, n_par, n_samp, size, _entropy_atol, cross_entropy_atol, _kl_atol):
    """Two sample entropy matches the log volume."""
    samples_compare, samples_test = draw_pair(shape, shape, n_par, n_samp, size, size)
    expected = SHAPES[shape][1](n_par, size)
    assert entropy_nn2(samples_compare, samples_test) == pytest.approx(expected, rel=0, abs=cross_entropy_atol)


@pytest.mark.parametrize(
    ('shape', 'n_par', 'n_samp', 'size', '_entropy_atol', '_cross_entropy_atol', 'kl_atol'),
    SAME_CASES,
)
def test_kl_divergence_same(shape, n_par, n_samp, size, _entropy_atol, _cross_entropy_atol, kl_atol):
    """KL divergence vanishes for identical shapes."""
    samples_compare, samples_test = draw_pair(shape, shape, n_par, n_samp, size, size)
    divergence, _, _ = kl_divergence_nn2(samples_compare, samples_test)
    assert divergence == pytest.approx(0.0, rel=0, abs=kl_atol)


@pytest.mark.parametrize(
    ('shape_test', 'shape_compare', 'n_par', 'n_samp', 'size_test', 'size_compare', 'kl_atol'),
    NESTED_CASES,
)
def test_kl_divergence_nested(shape_test, shape_compare, n_par, n_samp, size_test, size_compare, kl_atol):
    """KL divergence matches the log volume ratio for nested shapes."""
    samples_compare, samples_test = draw_pair(shape_test, shape_compare, n_par, n_samp, size_test, size_compare)
    divergence, _, _ = kl_divergence_nn2(samples_compare, samples_test)
    expected = SHAPES[shape_compare][1](n_par, size_compare) - SHAPES[shape_test][1](n_par, size_test)
    assert kl_atol < expected
    assert divergence == pytest.approx(expected, rel=0, abs=kl_atol)


@pytest.mark.parametrize(('n_par', 'n_samp', 'n_trial', 'entropy_atol', 'cross_entropy_atol', 'kl_atol'), BALL_CASES)
def test_ball_edge_bias(n_par, n_samp, n_trial, entropy_atol, cross_entropy_atol, kl_atol):
    """Mean bias for a ball matches the exact expected edge effect."""
    kl_bias, entropy_bias, cross_entropy_bias = mean_bias('ball', n_par, n_samp, n_trial)
    expected_entropy_bias = ball_edge_bias(n_par, n_samp - 1, 1.0, 1.0)
    expected_cross_entropy_bias = ball_edge_bias(n_par, n_samp, 1.0, 1.0)
    assert entropy_bias == pytest.approx(expected_entropy_bias, rel=0, abs=entropy_atol)
    assert cross_entropy_bias == pytest.approx(expected_cross_entropy_bias, rel=0, abs=cross_entropy_atol)
    assert kl_bias == pytest.approx(expected_cross_entropy_bias - expected_entropy_bias, rel=0, abs=kl_atol)


@pytest.mark.parametrize(('n_par', 'n_samp', 'n_trial', 'radius_compare', 'kl_atol'), NESTED_BALL_CASES)
def test_nested_ball_edge_bias(n_par, n_samp, n_trial, radius_compare, kl_atol):
    """Mean KL divergence bias for nested balls matches the exact expected edge effect."""
    kl_bias, _, _ = mean_bias('ball', n_par, n_samp, n_trial, radius_compare)
    expected = ball_edge_bias(n_par, n_samp, 1.0, radius_compare) - ball_edge_bias(n_par, n_samp - 1, 1.0, 1.0)
    assert kl_bias == pytest.approx(expected, rel=0, abs=kl_atol)


@pytest.mark.parametrize(('n_par', 'n_samp'), [(1, 100), (2, 2000), (3, 10**5), (5, 10**8)])
def test_leading_edge_bias_prediction(n_par, n_samp):
    """Leading order edge prediction approaches the exact ball prediction for many samples."""
    assert ball_edge_bias(n_par, n_samp, 1.0, 1.0) / leading_edge_bias('ball', n_par, n_samp) == pytest.approx(1.0, abs=0.03)


@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize(('n_par', 'n_samp', 'rtol'), [(2, 100, 0.23), (3, 400, 0.16), (4, 1600, 0.15), (5, 6400, 0.14)])
def test_leading_edge_bias(shape, n_par, n_samp, rtol):
    """Mean entropy bias matches the leading order edge prediction."""
    assert mean_entropy_bias(shape, n_par, n_samp) == pytest.approx(leading_edge_bias(shape, n_par, n_samp), rel=rtol)


@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize(('n_par', 'n_small', 'n_large', 'atol'), SCALING_CASES)
def test_edge_bias_sample_scaling(shape, n_par, n_small, n_large, atol):
    """Mean entropy bias scales as n_samp**(-1 / n_par)."""
    ratio = mean_entropy_bias(shape, n_par, n_large) / mean_entropy_bias(shape, n_par, n_small)
    assert np.log(ratio) / np.log(n_large / n_small) == pytest.approx(-1 / n_par, rel=0, abs=atol)


@pytest.mark.parametrize(('n_par', 'n_samp', 'rtol'), [(3, 400, 0.15), (4, 1600, 0.11), (5, 6400, 0.09)])
def test_edge_bias_shape_scaling(n_par, n_samp, rtol):
    """Ratio of cube and ball biases matches the ratio of surface areas at fixed volume."""
    ratio = mean_entropy_bias('cube', n_par, n_samp) / mean_entropy_bias('ball', n_par, n_samp)
    assert ratio == pytest.approx(2 * np.exp(-log_ball_volume(n_par) / n_par), rel=rtol)


@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('n_par', [3, 4, 5])
def test_edge_bias_dimension_scaling(shape, n_par):
    """Number of samples needed for a fixed bias grows exponentially with dimension as predicted."""
    target = 0.2
    n_samp = int(np.round((leading_edge_bias(shape, n_par, 1) / target) ** n_par))
    assert mean_entropy_bias(shape, n_par, n_samp) == pytest.approx(target, rel=0.15)
