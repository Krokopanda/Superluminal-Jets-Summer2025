
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import os

from simulationImport import importCSV


def loglike() -> pt.TensorVariable:
    wt = pt.vector("wt", dtype="float64")
    wc = pt.scalar("wc", dtype="float64")
    wt_squared = pt.pow(wt, 2)
    wc_squared = pt.pow(wc, 2)
    sum_squares = wc_squared + wt_squared

    # First expression (used when wc < 1)
    square_root = pt.sqrt(sum_squares - 1)
    at = pt.arctan(square_root)

    numer1 = (wc_squared * square_root) + ((wt_squared - wc_squared) * at)
    denum1 = pt.pow(sum_squares, 2)
    expr1 = pt.log(numer1 / denum1)

    # Second expression (used when wc >= 1)
    wt_times_wc = wt * wc
    at_ratio = pt.arctan(wt / wc)
    # Single-argument arctan is OK here because wt & wc are always positive

    numer2 = 2 * wt_times_wc * (wc - 1) + wt_times_wc + (wt_squared - wc_squared) * at_ratio
    denum2 = pt.pow(sum_squares, 2)
    expr2 = pt.log(numer2 / denum2)
   
    condition = pt.lt(wc, 1)
    
    result = pt.switch(condition, expr1, expr2)

    return pytensor.function([wt,wc],result)

function = loglike()

# Get the directory this code file is stored in
dir_path = os.path.dirname(os.path.realpath(__file__))

# find the path for the data source;  this should work on everyone's system now
#dataset = "/isotropic_sims/10000/data_3957522615761_xx_0.8_yy_0.8_zz_0.8.csv"
dataset = "/isotropic_sims/10000/data_3957522615600_xx_1.2_yy_1.2_zz_1.2.csv"

dataSource = (
    dir_path + dataset
)

# Import data from file
dataAll = importCSV(dataSource)
radec_data = [sublist[1:3] for sublist in dataAll]
wt_data = [sublist[3] for sublist in dataAll]
# wt_data = pytensor.shared(wt_dataOLD, name="wt_shared")

npts=400
wcvals = np.linspace(0.85,1.2,npts)
loglikevals = np.zeros(npts)
for i in range(npts):
    loglikevals[i] = np.sum(function(wt_data,wcvals[i]))

plt.style.use('_mpl-gallery')

# plot
fig, ax = plt.subplots()

ax.plot(wcvals, loglikevals)

plt.show()
    
