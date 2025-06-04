#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 08:53:03 2025

@author: mseifer1
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import os

from simulationImport import importCSV


def loglike(wc: pt.TensorVariable, wt: pt.TensorVariable) -> pt.TensorVariable:
    # wt = pt.vector("wt", dtype="float64")
    # wc = pt.scalar("wc", dtype="float64")
    wt_squared = pt.pow(wt, 2)
    wc_squared = pt.pow(wc, 2)
    sum_squares = wc_squared + wt_squared
    square_root = pt.sqrt(sum_squares - 1)
    at = pt.arctan(square_root)
    numer = (wc_squared * square_root) + ((wt_squared - wc_squared) * at)
    denum = pt.pow(sum_squares, 2)

    expr = pt.log(numer / denum)

    wt_times_wc = wc * wt
    at2 = pt.arctan(wt / wc)
    numer2 = 2 * wt_times_wc * (wc - 1) + wt_times_wc + (wt_squared - wc_squared) * at2
    denum2 = pt.pow(sum_squares, 2)
    expr2 = pt.log(numer2 / denum2)

<<<<<<< HEAD
    condition = pt.lt(wc, 1)

    result = pt.switch(condition, expr, expr2)

    return pytensor.function([wt,wc],expr2)

    return result
=======
>>>>>>> origin/PyTensorLogLike

# TODO check how this behaves, still not getting negative plots with either data
    expit2 = pm.math.sigmoid((wc-1)*1000)
    expit1 = pm.math.sigmoid(1000*(1-wc))
    final_expression = expr*expit2 + expr2*expit1
    return final_expression

# Get the directory this code file is stored in
dir_path = os.path.dirname(os.path.realpath(__file__))

# find the path for the data source;  this should work on everyone's system now
dataSource = (
    dir_path + "/isotropic_sims/10000/data_3957522615600_xx_1.2_yy_1.2_zz_1.2.csv"
)

print(f"Running on PyMC v{pm.__version__}")

# Import data from file
dataAll = importCSV(dataSource)
radec_data = [sublist[1:3] for sublist in dataAll]
wt_data = [sublist[3] for sublist in dataAll]
# wt_data = pytensor.shared(wt_dataOLD, name="wt_shared")

model = pm.Model()

with model:
    # Priors for unknown model parameters.  I defined q to be wc - 1, to avoid
    # confusion between the model parameter and the inverse speed of light as a
    # function of the parameters.
    q = pm.TruncatedNormal("q", mu=0, sigma=1, lower=-1,upper=1)
    #wc = pm.TruncatedNormal("wc", mu=0, sigma=1, lower=0)
    # TODO: this should be replaced with a truncated distribution, so that
    # q < -1 (wc < 0) is impossible

    # Expected value of wc, in terms of unknown model parameters and observed "X" values.
    # Right now this is very simple.  Eventually it will need to accept more parameter
    # values, along with RA & declination.
    wc = q + 1

    # Likelihood (sampling distribution) of observations
    wt_obs = pm.CustomDist("wt_obs", wc, observed=wt_data, logp=loglike)

    trace = pm.sample(1000)

az.plot_trace(trace, show=True)
