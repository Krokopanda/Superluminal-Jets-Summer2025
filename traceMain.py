import pymc as pm
import pytensor
import pytensor.tensor as pt
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from logpTensor import loglike
from simulationImport import importCSV

# config InlineBackend.figure_format = 'retina'
az.style.use("arviz-darkgrid")

# Example data input
dataSource = "isotropic_sims/300/data_3957593320521_xx_1.2_yy_1.2_zz_1.2.csv"

dataAll = importCSV(dataSource)
data = [sublist[3] for sublist in dataAll]
data_input = np.array(data)

log_fast_likelihood, log_slow_likelihood = loglike()


with pm.Model() as model:
    # Define the variable c
    c = pm.TruncatedNormal("c", lower=0, sigma=0.5)

    idata_grad = pm.sample(10000, tuning=1000)
az.plot_trace(idata_grad, lines=[("c", {}, ctrue)], show=True)
