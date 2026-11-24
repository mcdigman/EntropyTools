"""Functions implementing the 1 and 2 sample nearest neigbor approximations of sample entropy and KL divergence."""

from typing import Literal

import numpy as np
import scipy.special
from numba import njit, prange
from numpy.typing import NDArray

_DIGAMMA_SMALL = scipy.special.digamma(np.arange(0, 100000)) + np.euler_gamma


@njit(fastmath={'nsz','arcp','contract','afn','reassoc'})
def _nn_corrected_log_distance(sample_test: NDArray[np.float64], samples_compare: NDArray[np.float64], identical_ignore: int, duplicate_mode: Literal[0, 1], log_mode: Literal['geo', 'log']) -> np.float64:
    """
    Compute the log distance from a sample to its nearest non-identical neighbor, with duplicate correction.

    For a given sample, this function finds the squared Euclidean distance to the nearest non-identical
    neighbor in the provided set of samples. If multiple identical points are present, a correction factor
    is applied to reduce bias, allowing up to `identical_ignore` exact duplicates to be ignored without correction.

    Parameters
    ----------
    sample_test : numpy.ndarray of shape (n_parameters,)
        The sample for which the nearest neighbor distance is computed.
    samples_compare : numpy.ndarray of shape (n_samples, n_parameters)
        Array of samples among which the nearest non-identical neighbor is searched.
    identical_ignore : int
        Number of exact duplicates to ignore before applying the correction factor.
    duplicate_mode : Literal[0, 1]
        Mode selector for handling duplicates

    Returns
    -------
    logd_nn_corr : float
        Logarithm of the squared distance to the nearest non-identical neighbor, corrected for duplicates.

    Notes
    -----
    The correction factor is based on the number of exact duplicates found and assumes
    duplicates distances are randomly distributed within an n-dimensional ball to the nearest neighbor radius.
    """
    assert len(sample_test.shape) == 1
    assert len(samples_compare.shape) == 2

    n_samp = samples_compare.shape[0]
    n_par = samples_compare.shape[1]

    assert sample_test.shape[0] == n_par

    dist_sq_min = np.inf
    present_count = 0  # track how many times the identical point is present
    repeat_count_nearest = 0  # track how many times the nearest non-identical point is repeated

    for itrn1 in range(n_samp):
        sample1 = samples_compare[itrn1]

        dist_sq = 0.0
        for itrp in range(n_par):
            dist_sq += (sample_test[itrp] - sample1[itrp]) ** 2

        if dist_sq == dist_sq_min:
            repeat_count_nearest += 1
        elif dist_sq == 0.0:
            # find the distance to the nearest non-identical point
            present_count += 1
        elif dist_sq < dist_sq_min:
            repeat_count_nearest = 1
            dist_sq_min = dist_sq


    # log distance to nearest non-identical neighbor
    logd_nn = np.log(dist_sq_min)

    if duplicate_mode == 0:
        if log_mode == 'geo' and repeat_count_nearest < _DIGAMMA_SMALL.size:
            log_correction = _DIGAMMA_SMALL[repeat_count_nearest] - np.log(n_samp - present_count) + np.log(n_samp - 1)
        else:
            log_correction = np.log(repeat_count_nearest) - np.log(n_samp - present_count) + np.log(n_samp - 1)
    else:
        # allow for up to a certain number of identical points to be ignored with no correction factor
        repeat_adjust = repeat_count_nearest + max(present_count - identical_ignore, 0)

        # correction factor to try to heuristically reduce bias caused by exact duplicates
        # possibly necessary because we are using the nearest non-identical neighbor
        # correction factor is larger the more times the point was duplicated
        # based on the assumption that duplicates are at distances randomly distributed
        # within an n-dimensional ball out to radius of the nearest non-identical neighbor
        if log_mode == 'geo' and repeat_count_nearest < _DIGAMMA_SMALL.size:
            log_correction = _DIGAMMA_SMALL[repeat_adjust]
        else:
            log_correction = np.log(repeat_adjust)

    # add the contribution and the correction factor
    return logd_nn - (2 / n_par) * log_correction


@njit(parallel=True, fastmath=True)
def _entropy_nn1_helper(samples: NDArray[np.float64], identical_ignore: int, log_mode: Literal['geo', 'log']) -> float:
    """
    Estimate the differential entropy of IID samples using the one-sample nearest neighbor estimator.

    This function implements the estimator based on the method described in Eq. 5 of
    https://hal.archives-ouvertes.fr/hal-02774953/document, which computes the entropy
    based on the distance to the nearest neighbor for each sample.
    log_mode: Literal['geo', 'log']
        Mode selector for log approximation

    Parameters
    ----------
    samples : numpy.ndarray of shape (n_samples, n_parameters)
        Array of IID samples drawn from the target distribution. Each row corresponds
        to a sample, and each column to a parameter dimension.
    identical_ignore : int
        Number of exact duplicate samples to ignore before applying a correction factor
        for bias due to identical values. Allows up to `identical_ignore` duplicates
        to be present without correction.
    log_mode: Literal['geo', 'log']
        Mode selector for log approximation

    Returns
    -------
    entropy : float
        Estimated differential entropy of the sample distribution.

    Notes
    -----
    The estimator is based on the average log-distance to the nearest neighbor in the sample set.
    The returned value does not include the shift factor correction for bias; see `_entropy_shift_factor`.
    The `identical_ignore` argument controls how many exact duplicates are ignored before a bias correction
    is applied, which helps mitigate the effect of repeated samples on the entropy estimate.
    It should be at least 1 for nn1, because the nearest neighbor of a sample is itself in the one-sample case.
    """
    assert len(samples.shape) == 2
    assert identical_ignore > 0
    assert log_mode in ('geo', 'log')

    n_par = samples.shape[1]
    n_samp = samples.shape[0]

    assert n_par > 0
    assert n_samp >= 2

    logd_tot = 0.0
    for itrn2 in prange(n_samp):
        # find the distance to the nearest neighbor to this sample
        sample2 = samples[itrn2, :]
        logd_tot += _nn_corrected_log_distance(sample2, samples, identical_ignore, 0, log_mode)

    return n_par / n_samp / 2 * logd_tot


@njit(parallel=True, fastmath=True)
def _entropy_nn2_helper(samples_compare: NDArray[np.float64], samples_test: NDArray[np.float64], identical_ignore: int, log_mode: Literal['geo', 'log']) -> float:
    """
    Estimate the differential entropy of a test sample set using the two-sample nearest neighbor estimator.

    This function implements the estimator described in Eq. 6 of
    https://hal.archives-ouvertes.fr/hal-02774953/document, which computes the entropy
    based on the distance from each sample in `samples_test` to its nearest neighbor in `samples_compare`.

    Parameters
    ----------
    samples_compare : numpy.ndarray of shape (n_samples, n_parameters)
        Reference set of IID samples drawn from the target distribution. Exact duplicates of points in both samples_compare
        from samples_test are handled using the `identical_ignore` parameter.
        Duplicates within samples_compare may bias the estimate.
    samples_test : numpy.ndarray of shape (n_samples, n_parameters)
        Test set of samples for which the entropy is estimated. Duplicates within this set are allowed
        and require no special handling, although excessive duplicates may bias the estimate.
    identical_ignore : int
        Number of exact duplicate samples in `samples_compare` to ignore before applying a correction factor
        for bias due to identical values. This does not affect duplicates in `samples_test`.
    log_mode: Literal['geo', 'log']
        Mode selector for log approximation

    Returns
    -------
    entropy : float
        Estimated differential entropy of the test sample distribution.

    Notes
    -----
    The estimator is based on the average log-distance from each sample in `samples_test` to its nearest
    neighbor in `samples_compare`. The `identical_ignore` argument only affects how duplicates in `samples_compare`
    are handled; duplicates in `samples_test` are permitted and do not require correction.
    The returned value does not include the shift factor correction for bias; see `_entropy_shift_factor`.
    """
    assert len(samples_compare.shape) == 2
    assert len(samples_test.shape) == 2
    assert samples_compare.shape == samples_test.shape
    assert identical_ignore > -1
    assert log_mode in ('geo', 'log')

    n_par = samples_compare.shape[1]
    n_samp = samples_compare.shape[0]

    assert n_par > 0
    assert n_samp >= 2

    logd_tot = 0.0
    for itrn2 in prange(n_samp):
        # find the distance to the nearest neighbor to this sample
        sample2 = samples_test[itrn2, :]
        logd_tot += _nn_corrected_log_distance(sample2, samples_compare, identical_ignore, 1, log_mode)
    return n_par / n_samp / 2 * logd_tot


def _entropy_shift_factor(n_par: int, n_samp: int) -> float:
    """
    Compute the analytic shift factor for the nearest neighbor entropy estimator.

    This function returns the bias correction term that should be added to the
    nearest neighbor entropy estimate to account for the finite sample size and
    dimensionality, as described in the literature (see Eq. 5 and 6 of
    https://hal.archives-ouvertes.fr/hal-02774953/document).

    Parameters
    ----------
    n_par : int
        Number of parameter dimensions (features) in the sample.
    n_samp : int
        Number of samples in the dataset.

    Returns
    -------
    shift_factor : float
        Analytic correction factor to be added to the raw nearest neighbor entropy estimate.

    Notes
    -----
    The shift factor includes terms for the volume of the n-dimensional unit ball,
    the log gamma function, and Euler's constant, and is required for unbiased entropy estimation.
    """
    half_n_par = n_par / 2
    logc1 = half_n_par * np.log(np.pi) - scipy.special.loggamma(half_n_par + 1)
    # paper calls for np.log(n_samp) but digamma is more accurate for small n_samp
    sample_shift = scipy.special.digamma(n_samp + 1) 
    return sample_shift + logc1 + np.euler_gamma


def entropy_nn2(samples_compare: NDArray[np.float64], samples_test: NDArray[np.float64], *, identical_ignore: int = 0, log_mode: Literal['geo', 'log']='geo') -> float:
    """
    Estimate the differential entropy of a test sample set using the two-sample nearest neighbor estimator.

    This function implements the estimator described in Eq. 6 of
    https://hal.archives-ouvertes.fr/hal-02774953/document, which computes the entropy
    based on the distance from each sample in `samples_test` to its nearest neighbor in `samples_compare`.

    Parameters
    ----------
    samples_compare : numpy.ndarray of shape (n_samples, n_parameters)
        Reference set of IID samples drawn from the target distribution. Exact duplicates of points in both samples_compare
        from samples_test are handled using the `identical_ignore` parameter.
        Duplicates within samples_compare may bias the estimate.
    samples_test : numpy.ndarray of shape (n_samples, n_parameters)
        Test set of samples for which the entropy is estimated. Duplicates within this set are allowed
        and require no special handling, although excessive duplicates may bias the estimate.
    identical_ignore : int
        Number of exact duplicate samples in `samples_compare` to ignore before applying a correction factor
        for bias due to identical values. This does not affect duplicates in `samples_test`.
    log_mode: Literal['geo', 'log']
        Mode selector for log approximation

    Returns
    -------
    entropy : float
        Estimated differential entropy of the test sample distribution.

    Notes
    -----
    The estimator is based on the average log-distance from each sample in `samples_test` to its nearest
    neighbor in `samples_compare`. The `identical_ignore` argument only affects how duplicates in `samples_compare`
    are handled; duplicates in `samples_test` are permitted and do not require correction.
    """
    assert len(samples_compare.shape) == 2
    assert len(samples_test.shape) == 2
    assert samples_compare.shape == samples_test.shape
    assert identical_ignore > -1
    assert log_mode in ('geo', 'log')

    n_par = samples_compare.shape[1]
    n_samp = samples_compare.shape[0]

    assert n_par > 0
    assert n_samp >= 2

    shift_factor = _entropy_shift_factor(n_par, n_samp)

    return _entropy_nn2_helper(samples_compare, samples_test, identical_ignore, log_mode) + shift_factor


def entropy_nn1(samples: NDArray[np.float64], *, identical_ignore: int = 1, log_mode: Literal['geo', 'log']='geo') -> float:
    """
    Estimate the differential entropy of IID samples using the one-sample nearest neighbor estimator.

    This function implements the estimator based on the method described in Eq. 5 of
    https://hal.archives-ouvertes.fr/hal-02774953/document, which computes the entropy
    based on the distance to the nearest neighbor for each sample.

    Parameters
    ----------
    samples : numpy.ndarray of shape (n_samples, n_parameters)
        Array of IID samples drawn from the target distribution. Each row corresponds
        to a sample, and each column to a parameter dimension.
    identical_ignore : int
        Number of exact duplicate samples to ignore before applying a correction factor
        for bias due to identical values. Allows up to `identical_ignore` duplicates
        to be present without correction.
    log_mode: Literal['geo', 'log']
        Mode selector for log approximation

    Returns
    -------
    entropy : float
        Estimated differential entropy of the sample distribution.

    Notes
    -----
    The estimator is based on the average log-distance to the nearest neighbor in the sample set.
    The `identical_ignore` argument controls how many exact duplicates are ignored before a bias correction
    is applied, which helps mitigate the effect of repeated samples on the entropy estimate.
    It should be at least 1 for nn1, because the nearest neighbor of a sample is itself in the one-sample case.
    """
    assert len(samples.shape) == 2
    assert log_mode in ('geo', 'log')

    n_par = samples.shape[1]
    n_samp = samples.shape[0]

    assert n_par > 0
    assert n_samp >= 2

    shift_factor = _entropy_shift_factor(n_par, n_samp - 1)
    entropy_base = _entropy_nn1_helper(samples, identical_ignore, log_mode)

    return entropy_base + shift_factor


def kl_divergence_nn2(samples_compare: NDArray[np.float64], samples_test: NDArray[np.float64], *, identical_ignore: int = 0, log_mode: Literal['geo', 'log']='geo') -> tuple[float, float, float]:
    """
    Estimate the Kullback-Leibler (KL) divergence between two distributions using the two-sample nearest neighbor estimator.

    This function computes the relative entropy (KL divergence) between the distributions represented by `samples_compare`
    (reference, e.g., posterior samples) and `samples_test` (test samples), using the nearest neighbor estimators for entropy
    as described in Eq. 7 of https://hal.archives-ouvertes.fr/hal-02774953/document.
    note Eq. 7 has the sign wrong

    Parameters
    ----------
    samples_compare : numpy.ndarray of shape (n_samples, n_parameters)
        Reference set of IID samples drawn from the first distribution (e.g., posterior).
    samples_test : numpy.ndarray of shape (n_samples, n_parameters)
        Test set of IID samples drawn from the second distribution.
    identical_ignore : int, optional
        Number of exact duplicate samples in `samples_compare` to ignore before applying a correction factor
        for bias due to identical values. This does not affect duplicates in `samples_test` unless they are also present in `samples_compare`.
    log_mode: Literal['geo', 'log']
        Mode selector for log approximation

    Returns
    -------
    divergence_k : float
        Estimated KL divergence D_KL(samples_test || samples_compare).
    e1 : float
        Estimated entropy of `samples_test`.
    e2 : float
        Estimated cross-entropy between `samples_compare` and `samples_test`.

    Notes
    -----
    The KL divergence is estimated as the difference between the cross-entropy between `samples_compare` and 
    `samples_test` and the entropy of `samples_test`, both computed using nearest neighbor estimators.
    """
    assert len(samples_compare.shape) == 2
    assert len(samples_test.shape) == 2
    assert samples_compare.shape == samples_test.shape
    assert identical_ignore > -1
    assert log_mode in ('geo', 'log')

    e1 = entropy_nn1(samples_test, identical_ignore=identical_ignore + 1, log_mode=log_mode)
    e2 = entropy_nn2(samples_compare, samples_test, identical_ignore=identical_ignore, log_mode=log_mode)
    divergence_k = e2 - e1
    return divergence_k, e1, e2
