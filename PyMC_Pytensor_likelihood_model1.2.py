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

# only pass wc and sigma
pytensor.config.exception_verbosity = "high"


# Get the directory this code file is stored in
dir_path = os.path.dirname(os.path.realpath(__file__))

# find the path for the data source;  this should work on everyone's system now
dataset = "/isotropic_sims/10000/data_3957522615761_xx_0.8_yy_0.8_zz_0.8.csv"
# dataset = "/isotropic_sims/10000/data_3957522615600_xx_1.2_yy_1.2_zz_1.2.csv"

dataSource = dir_path + dataset

print(f"Running on PyMC v{pm.__version__}")

# Import data from file
dataAll = importCSV(dataSource)
radec_data = [sublist[1:3] for sublist in dataAll]
wt_data = [sublist[3] for sublist in dataAll]
wt_min = np.min(wt_data)
wc_min = np.sqrt(1 - wt_min**2)


model = pm.Model()
size = 1

n_hatDUM = pt.constant([1.0, 0.0, 0.0], dtype=pytensor.config.floatX)


# TODO removing v and vhat from the calling shenanigans does create a new error. Look into that perchance
def true_wt(
    n_hat: pt.TensorVariable,
    wc: pt.TensorVariable,
    sigma: pt.TensorVariable,
    size: pt.TensorVariable,
    v: pt.TensorVariable,
    v_hat: pt.TensorVariable,
) -> pt.TensorVariable:

    # v = pt.random.normal(0, 1, size=(3,))
    # norm_vec = pt.sqrt(pt.sum(v**2))
    #
    # # Define the unit vector on the sphere
    # v_hat = v / norm_vec
    if n_hat is None or v_hat is None:
        raise ValueError("One of the input variables is None")
    dot_product = pt.dot(n_hat, v_hat)
    numer = (1 / v) + wc * n_hat * dot_product
    denom = pt.sqrt(1 - pt.pow(dot_product, 2))
    expr = numer / denom
    return pm.Normal.dist(expr, sigma, size=size)


size = 1

with model:
    # n_hatDUM = pm.Normal("n_hat", mu=0)
    wc = pm.TruncatedNormal("wc", mu=0, sigma=10, lower=0)
    sigma = pm.Normal("sigma", mu=0, sigma=1)
    v = pm.TruncatedNormal("v", mu=0, sigma=1, shape=3, lower=0, upper=1)
    norm_vec = pt.sqrt(pt.sum(v**2))

    # unit vector on the sphere
    v_hat = pm.Deterministic("v_hat", v / norm_vec)
    # Likelihood (sampling distribution) of observations
    wt_obs = pm.CustomDist(
        "wt_obs", wc, sigma, size, v, v_hat, dist=true_wt, observed=n_hatDUM
    )

    trace = pm.sample(1000, target_accept=0.80)
    wt_draws = pm.draw(wt_obs, draws=1000)
plt.hist(wt_draws, bins=30, density=True)
plt.xlabel("wt")
plt.ylabel("Pwt")
plt.title("Histogram")
plt.show()


summ = az.summary(trace)
print(summ)
az.plot_trace(trace, show=True)
az.plot_posterior(trace, round_to=3, figsize=[8, 4], textsize=10)
