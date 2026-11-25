# SPDX-FileCopyrightText: Copyright 2026 Matthew C. Digman
# SPDX-License-Identifier: Apache-2.0
"""Tests that the KL divergence vanishes for draws from the same complicated distribution."""

import numpy as np
import pytest

from EntropyTools.entropy_process import kl_divergence_nn2

SEED = 31415
PARAMETER_SEED = 27182

N_COMPONENT = 6
N_DOF = 4
N_HIDDEN = 32

# Tolerances are approximately four standard deviations.
# family, n_par, n_samp, KL divergence atol
CASES = [
    ('warped_mixture', 2, 1000, 0.28),
    ('warped_mixture', 2, 10000, 0.09),
    ('warped_mixture', 3, 1000, 0.27),
    ('warped_mixture', 3, 10000, 0.09),
    ('warped_mixture', 5, 1000, 0.28),
    ('warped_mixture', 5, 10000, 0.10),
    ('warped_mixture', 10, 1000, 0.36),
    ('warped_mixture', 10, 10000, 0.10),
    ('warped_mixture', 20, 10000, 0.14),
    ('random_network', 3, 1000, 0.27),
    ('random_network', 3, 10000, 0.10),
    ('random_network', 5, 1000, 0.29),
    ('random_network', 5, 10000, 0.09),
    ('random_network', 10, 1000, 0.37),
    ('random_network', 10, 10000, 0.12),
    ('random_network', 20, 10000, 0.15),
]

# family, n_par, mean KL divergence atol
MEAN_CASES = [
    ('warped_mixture', 2, 0.020),
    ('warped_mixture', 5, 0.020),
    ('warped_mixture', 10, 0.025),
    ('random_network', 3, 0.020),
    ('random_network', 5, 0.020),
    ('random_network', 10, 0.025),
]

CONTROL_CASES = [case for case in CASES if case[2] == 10000]


def warped_mixture_draw(rng, n_par, n_samp):
    """Draw from a mixture of Student-t distributions warped by volume preserving shears."""
    rng_par = np.random.default_rng([PARAMETER_SEED, n_par])
    weights = rng_par.dirichlet(np.full(N_COMPONENT, 3.0))
    means = rng_par.normal(0.0, 2.0, (N_COMPONENT, n_par))
    widths = np.exp(rng_par.uniform(-1.2, 0.5, (N_COMPONENT, n_par)))
    rotations = np.linalg.qr(rng_par.normal(size=(N_COMPONENT, n_par, n_par)))[0]

    components = rng.choice(N_COMPONENT, n_samp, p=weights)
    scales = np.sqrt(N_DOF / rng.chisquare(N_DOF, (n_samp, 1)))
    offsets = scales * widths[components] * rng.normal(size=(n_samp, n_par))
    samples = means[components] + np.einsum('nij,nj->ni', rotations[components], offsets)
    samples[:, 1:] += 0.3 * samples[:, :1] ** 2
    samples[:, 0] += np.sin(samples[:, 1])
    return samples


def random_network_draw(rng, n_par, n_samp):
    """Draw from the pushforward of a Gaussian through a random neural network."""
    rng_par = np.random.default_rng([PARAMETER_SEED, n_par])
    n_latent = n_par + 2
    weights_in = rng_par.normal(0.0, 2.0 / np.sqrt(n_latent), (n_latent, N_HIDDEN))
    weights_out = rng_par.normal(0.0, 2.0 / np.sqrt(N_HIDDEN), (N_HIDDEN, n_par))
    return np.tanh(rng.normal(size=(n_samp, n_latent)) @ weights_in) @ weights_out


DRAWS = {
    'warped_mixture': warped_mixture_draw,
    'random_network': random_network_draw,
}


def draw_pair(family, n_par, n_samp):
    """Draw two independent sample sets from the same distribution."""
    rng = np.random.default_rng([SEED, n_par, n_samp])
    return DRAWS[family](rng, n_par, n_samp), DRAWS[family](rng, n_par, n_samp)


def shuffle_columns(rng, samples):
    """Permute each column independently, preserving only the marginal distributions."""
    return rng.permuted(samples, axis=0)


def matched_gaussian_draw(rng, samples):
    """Draw from a Gaussian with the same mean and covariance as the samples."""
    return rng.multivariate_normal(samples.mean(axis=0), np.cov(samples, rowvar=False), samples.shape[0])


CONTROLS = {
    'shuffled': shuffle_columns,
    'matched_gaussian': matched_gaussian_draw,
}


def mean_kl_divergence(family, n_par):
    """Average the KL divergence over many pairs of draws from the same distribution."""
    n_samp = 1000
    n_trial = 300
    rng = np.random.default_rng([SEED, n_par])
    draw = DRAWS[family]
    return np.mean([kl_divergence_nn2(draw(rng, n_par, n_samp), draw(rng, n_par, n_samp))[0] for _ in range(n_trial)])


@pytest.mark.parametrize(('family', 'n_par', 'n_samp', 'kl_atol'), CASES)
def test_kl_divergence_same(family, n_par, n_samp, kl_atol):
    """KL divergence vanishes for draws from the same distribution."""
    samples_compare, samples_test = draw_pair(family, n_par, n_samp)
    divergence, _, _ = kl_divergence_nn2(samples_compare, samples_test)
    assert divergence == pytest.approx(0.0, rel=0, abs=kl_atol)


@pytest.mark.parametrize(('family', 'n_par', 'kl_atol'), MEAN_CASES)
def test_kl_divergence_same_mean(family, n_par, kl_atol):
    """KL divergence averaged over many draws from the same distribution vanishes."""
    assert mean_kl_divergence(family, n_par) == pytest.approx(0.0, rel=0, abs=kl_atol)


@pytest.mark.parametrize('control', CONTROLS)
@pytest.mark.parametrize(('family', 'n_par', 'n_samp', 'kl_atol'), CONTROL_CASES)
def test_kl_divergence_control(family, n_par, n_samp, kl_atol, control):
    """KL divergence is inconsistent with zero for similar but distinct distributions."""
    samples_compare, samples_test = draw_pair(family, n_par, n_samp)
    rng = np.random.default_rng([SEED, n_par, n_samp, 1])
    divergence, _, _ = kl_divergence_nn2(CONTROLS[control](rng, samples_compare), samples_test)
    assert divergence > kl_atol
