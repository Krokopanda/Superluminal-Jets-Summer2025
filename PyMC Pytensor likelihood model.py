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

# TODO: make this more human-readable by renaming d->wt, c->wc.
def loglike(c_input: pt.TensorVariable, d_input: pt.TensorVariable) -> pt.TensorVariable:
    d = pt.vector("d", dtype="float64")
    c = pt.scalar("c", dtype="float64")
    d_squared = pt.pow(d, 2)
    c_squared = pt.pow(c, 2)
    sum_squares = c_squared + d_squared
    square_root = pt.sqrt(sum_squares - 1)
    at = pt.arctan(square_root)
    numer = (c_squared * square_root) + ((d_squared - c_squared) * at)
    denum = pt.pow(sum_squares, 2)

    expr = pt.log(numer / denum)
    # print(expr.type)
    likelihood = pytensor.function(
        [d, c], expr, mode="FAST_RUN", on_unused_input="ignore"
    )
    answer = likelihood(d_input, c_input)
    return answer

# Get the directory this code file is stored in
dir_path = os.path.dirname(os.path.realpath(__file__))

#find the path for the data source;  this should work on everyone's system now
dataSource = dir_path + '/isotropic_sims/10000/data_3957522615761_xx_0.8_yy_0.8_zz_0.8.csv'

print(f"Running on PyMC v{pm.__version__}")

# Import data from file
dataAll = importCSV(dataSource)
radec_data = [sublist[1:3] for sublist in dataAll]
wt_data = [sublist[3] for sublist in dataAll]

model = pm.Model()

with model:
    
    # Priors for unknown model parameters.  I defined q to be wc - 1, to avoid 
    # confusion between the model parameter and the inverse speed of light as a 
    # function of the parameters.
    q = pm.Normal("q", mu=0, sigma=1)
    # TODO: this should be replaced with a truncated distribution, so that 
    # q < -1 (wc < 0) is impossible

    # Expected value of wc, in terms of unknown model parameters and observed "X" values.  
    # Right now this is very simple.  Eventually it will need to accept more parameter 
    # values, along with RA & declination.
    wc = q + 1
    
    # Likelihood (sampling distribution) of observations
    wt_obs = pm.CustomDist("wt_obs", wc, observed=wt_data, logp=loglike)
    
    trace = pm.sample(1000)
    
az.plot_trace(trace)
