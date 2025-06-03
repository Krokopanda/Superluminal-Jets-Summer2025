import pytensor
import pytensor.tensor as pt


# import pytensor.tensor as pt

from pytensor.graph import Apply, Op
from scipy.optimize import approx_fprime


def loglike(d_input, c_input):
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


def main():

    vals = []

    for i in range(1, 50000, 50):
        vals.append(i)
    answer = loglike(vals, 3)
    print(answer)


main()
# grad_c = pt.grad(pt.sum(expr), c)
# grad_func_c = pytensor.function([d, c], grad_c)
# answer2 = grad_func_c([1, 2, 3, 4], 3)
