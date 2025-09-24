# This file is part of "probitlcmlongit" which is released under GPL v3.
#
# Copyright (c) 2022-2025 Eric Alan Wayman <ericwaymanpublications@mailworks.org>.
#
# This program is FLO (free/libre/open) software: you can redistribute
# it and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


## integrated_autocorr2

import numpy as np
from scipy.linalg import toeplitz

# The following two functions, integrated_autocorr2 and yule_walker,
# were taken from 
#     https://github.com/rmcgibbo/pyhmc/blob/master/pyhmc/autocorr2.py
# That file was licensed initially under the BSD License.
# The copyrights for that file are as follows:
# Copyright (c) 1996-2001 Ian T. Nabney
# Copyright (c) 1998-2000 Aki Vehtari
# Copyright (c) 2008-2009 Kilian Koepsell
# Copyright (c) 2014-2014 Robert T. McGibbon

def integrated_autocorr2(x):
    r"""Estimate the integrated autocorrelation time, :math:`\tau_{int}` of a
    time series.

    This method estimates the spectral density at zero frequency by fitting
    an AR(p) model, with p selected by AIC.

    Parameters
    ----------
    x : ndarray, shape=(n_samples, n_dims)
        The time series, with time along axis 0.

    References
    ----------
    .. [1] Plummer, M., Best, N., Cowles, K., and Vines, K. (2006). CODA:
        Convergence diagnosis and output analysis for MCMC. R News, 6(1):7-11.

    Returns
    -------
    tau_int : ndarray, shape=(n_dims,)
        The estimated integrated autocorrelation time of each dimension in
        ``x``, considered independently.
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    process_var = np.var(x, axis=0, ddof=1)

    tau_int = np.zeros(x.shape[1])
    for j in range(x.shape[1]):
        # fit an AR(p) model, with p selected by AIC
        rho, sigma2 = yule_walker(x[:,j], order_max=10)
        # power spectral density at zero frequency
        spec0 = sigma2 / (1 - np.sum(rho))**2
        # divide by the variance
        tau_int[j] = spec0 / process_var[j]

    return tau_int


def yule_walker(X, aic=True, order_max=None, demean=True):
    """Estimate AR(p) parameters from a sequence X using Yule-Walker equation.

    Parameters
    ----------
    X : array-like
        1d array
    aic: bool
        If ``True``, then the Akaike Information Criterion is used to choose
        the order of the autoregressive model. If ``False``, the model of order
        ``order.max`` is fitted.
    order_max : integer, optional
        Maximum order of model to fit. Defaults to the smaller of N-1 and
        10*log10(N) where N is the length of the sequence.
    demean : bool
        True, the mean is subtracted from `X` before estimation.

    Returns
    -------
    rho : array, shape=(order,)
        The autoregressive coefficients
    sigma2 : float
        Variance of the nosie term
    aic : float
        Akaike Information Criterion
    """
    # this code is adapted from https://github.com/statsmodels/statsmodels.
    # changes are made to increase compability with R's ``ar.yw``.
    X = np.array(X)
    if demean:
        X -= X.mean()
    n = X.shape[0]

    if X.ndim > 1 and X.shape[1] != 1:
        raise ValueError("expecting a vector to estimate AR parameters")

    if order_max is None:
        order_max = min(n - 1, int(10 * np.log10(n)))

    r = np.zeros(order_max+1, np.float64)
    r[0] = (X**2).sum() / n

    for k in range(1, order_max+1):
        r[k] = (X[0:-k]*X[k:]).sum() / n

    orders = np.arange(1, order_max+1) if aic else [order_max]
    aics = np.zeros(len(orders))
    sigmasqs = np.zeros(len(orders))
    rhos = [None for i in orders]

    for i, order in enumerate(orders):
        r_left = r[:order]
        r_right = r[1:order+1]

        # R = toeplitz(r[:-1])
        R = toeplitz(r_left)
        # rho = np.linalg.solve(R, r[1:])
        rho = np.linalg.solve(R, r_right)
        # sigmasq = r[0] - (r[1:]*rho).sum()
        sigmasq = r[0] - (r_right*rho).sum()
        aic = len(X) * (np.log(sigmasq) + 1) + 2*order + 2*demean
        # R compability
        sigmasq = sigmasq * len(X)/(len(X) - (order + 1))

        aics[i] = aic
        sigmasqs[i] = sigmasq
        rhos[i] = rho

    index = np.argmin(aics)
    return rhos[index], sigmasqs[index]


## geweke

# This function "geweke" was copied from PyMC v3.7.
# That software was distributed under the Apache License, Version 2.0.
# The copyrights for that release are as follows:
# Copyright (c) 2006 Christopher J. Fonnesbeck (Academic Free License)
# Copyright (c) 2007-2008 Christopher J. Fonnesbeck, Anand Prabhakar Patil, David Huard (Academic Free License)
# Copyright (c) 2009-2017 The PyMC developers (see contributors to pymc-devs on GitHub)
def geweke(x, first=.1, last=.5, intervals=20):
    R"""Return z-scores for convergence diagnostics.

    Compare the mean of the first % of series with the mean of the last % of
    series. x is divided into a number of segments for which this difference is
    computed. If the series is converged, this score should oscillate between
    -1 and 1.

    Parameters
    ----------
    x : array-like
      The trace of some stochastic parameter.
    first : float
      The fraction of series at the beginning of the trace.
    last : float
      The fraction of series at the end to be compared with the section
      at the beginning.
    intervals : int
      The number of segments.

    Returns
    -------
    scores : list [[]]
      Return a list of [i, score], where i is the starting index for each
      interval and score the Geweke score on the interval.

    Notes
    -----

    The Geweke score on some series x is computed by:

      .. math:: \frac{E[x_s] - E[x_e]}{\sqrt{V[x_s] + V[x_e]}}

    where :math:`E` stands for the mean, :math:`V` the variance,
    :math:`x_s` a section at the start of the series and
    :math:`x_e` a section at the end of the series.

    References
    ----------
    Geweke (1992)
    """

    if np.ndim(x) > 1:
        return [geweke(y, first, last, intervals) for y in np.transpose(x)]

    # Filter out invalid intervals
    for interval in (first, last):
        if interval <= 0 or interval >= 1:
            raise ValueError(
                "Invalid intervals for Geweke convergence analysis",
                (first,
                 last))
    if first + last >= 1:
        raise ValueError(
            "Invalid intervals for Geweke convergence analysis",
            (first,
             last))

    # Initialize list of z-scores
    zscores = []

    # Last index value
    end = len(x) - 1

    # Start intervals going up to the <last>% of the chain
    last_start_idx = (1 - last) * end

    # Calculate starting indices
    start_indices = np.arange(0, int(last_start_idx), step=int(
        (last_start_idx) / (intervals - 1)))

    # Loop over start indices
    for start in start_indices:
        # Calculate slices
        first_slice = x[start: start + int(first * (end - start))]
        last_slice = x[int(end - last * (end - start)):]

        z = first_slice.mean() - last_slice.mean()
        z /= np.sqrt(first_slice.var() + last_slice.var())

        zscores.append([start, z])

    if intervals is None:
        return np.array(zscores[0])
    else:
        return np.array(zscores)
