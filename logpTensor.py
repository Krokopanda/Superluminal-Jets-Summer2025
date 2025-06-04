import pytensor
import pytensor.tensor as pt


# import pytensor.tensor as pt

from pytensor.graph import Apply, Op
from scipy.optimize import approx_fprime


def loglike():
    d = pt.vector("d", dtype="float64")
    c = pt.scalar("c", dtype="float64")
    d_squared = pt.pow(d, 2)
    c_squared = pt.pow(c, 2)
    sum_squares = c_squared + d_squared
    # fast light case
    square_root = pt.sqrt(sum_squares - 1)
    at = pt.arctan(square_root)
    numer = (c_squared * square_root) + ((d_squared - c_squared) * at)
    denum = pt.pow(sum_squares, 2)
    expr = pt.log(numer / denum)
    log_fast_likelihood = pytensor.function(
        [d, c], expr, mode="FAST_RUN", on_unused_input="ignore"
    )
    # slow light case
    dtimesc = c * d
    at2 = pt.arctan(d / c)
    numer2 = 2 * dtimesc * (c - 1) + dtimesc + (d_squared - c_squared) * at2
    denum2 = pt.pow(sum_squares, 2)
    expr2 = pt.log(numer2 / denum2)
    # log likelihood function demands format like log_slow_likelihood(data,c)
    log_slow_likelihood = pytensor.function(
        [d, c], expr2, mode="FAST_RUN", on_unused_input="ignore"
    )
    return pt.sum(expr), pt.sum(expr2)
    # return expr,expr2


def main():

    vals = []

    for i in range(2, 10):
        vals.append(i)
    for i in range(1, 45, 1):
        vals.append(i / 50)
    function1, function2 = loglike()
    print(function1([0.5, 0.9], 3))
    print(function2([1], 0.5))


if __name__ == "__main__":
    main()
# grad_c = pt.grad(pt.sum(expr), c)
# grad_func_c = pytensor.function([d, c], grad_c)
# answer2 = grad_func_c([1, 2, 3, 4], 3)
