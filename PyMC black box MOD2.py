#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 14:39:30 2025

@author: mseifer1

Code-only version of https://github.com/pymc-devs/pymc-examples/blob/main/examples/howto/blackbox_external_likelihood_numpy.ipynb
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from pytensor.graph import Apply, Op
from scipy.optimize import approx_fprime

print(f"Running on PyMC v{pm.__version__}")

# %config InlineBackend.figure_format = 'retina'
az.style.use("arviz-darkgrid")

def my_model():
    return 1

def my_loglike(m, c, sigma, x, data):
    # We fail explicitly if inputs are not numerical types for the sake of this tutorial
    # As defined, my_loglike would actually work fine with PyTensor variables!
    for param in (x,z):
        if not isinstance(param, (float, np.ndarray)):
            raise TypeError(f"Invalid input type to loglike: {type(param)}")
    model = my_model(m, c, x)
    return ((z**2)*np.sqrt(x**2+z**2-1)+(x**2-z**2)*\
            np.arctan(np.sqrt(x**2+z**2-1)))/((x**2+z**2)**2)
    # This is the log-likelihood for a Gaussian; we'd want to change this accordingly
    

# define a pytensor Op for our likelihood function

class LogLike(Op):
    def make_node(self, x,z) -> Apply:
        # Convert inputs to tensor variables
        x = pt.as_tensor(x)
        z = pt.as_tensor(z)

        inputs = [x,z]
        # Define output type, in our case a vector of likelihoods
        # with the same dimensions and same data type as data
        # If data must always be a vector, we could have hard-coded
        outputs = [pt.vector()]
        #outputs = [data.type()]

        # Apply is an object that combines inputs, outputs and an Op (self)
        return Apply(self, inputs, outputs)

    def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
        # This is the method that compute numerical output
        # given numerical inputs. Everything here is numpy arrays
        x,z = inputs  # this will contain my variables

        # call our numpy log-likelihood function
        loglike_eval = my_loglike(x,z)

        # Save the result in the outputs list provided by PyTensor
        # There is one list per output, each containing another list
        # pre-populated with a `None` where the result should be saved.
        outputs[0][0] = np.asarray(loglike_eval)
        
# set up our data
N = 10  # number of data points
x = np.linspace(0.0, 3.0, N)

xtrue = 0.4  # true gradient
ztrue = 3.0  # true y-intercept

#truemodel = my_model(xtrue, ztrue)
truemodel = my_model()
# make data
rng = np.random.default_rng(716743)
data = rng.normal(size=N) + truemodel

# create our Op
loglike_op = LogLike()

test_out = loglike_op(xtrue, ztrue)

def custom_dist_loglike(data,x,z):
    # data, or observed is always passed as the first input of CustomDist
    return loglike_op(x,z)

# use PyMC to sampler from log-likelihood
with pm.Model() as no_grad_model:
    # uniform priors on m and c
    x = pm.Uniform("x", lower=-10.0, upper=10.0, initval=xtrue)
    z = pm.Uniform("z", lower=-10.0, upper=10.0, initval=ztrue)

    # use a CustomDist with a custom logp function
    likelihood = pm.CustomDist(
        "likelihood", x, z, logp=custom_dist_loglike
    )
    
ip = no_grad_model.initial_point()

with no_grad_model:
    # Use custom number of draws to replace the HMC based defaults
    idata_no_grad = pm.sample(3000, tune=1000)

# plot the traces
az.plot_trace(idata_no_grad, lines=[("x", {}, xtrue), ("z", {}, ztrue)]);
