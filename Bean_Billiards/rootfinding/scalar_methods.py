import numpy as np
from scipy.optimize.zeros import RootResults
# RootResults is a class for representing the results and convergence of
#  rootfinding methods. I'll use it for compatibility with scipy.optimize.

def bisect(f, a, b, xtol=1e-12, maxiter=50, full_output=False, disp=True):
    assert f(a) * f(b) < 0
    flag = 0
    est_err = 1
    for iteration in range(maxiter):
        mid = (a + b) / 2
        if f(mid)*f(a) < 0:
            b = mid
        elif f(mid)*f(b) < 0:
            a = mid
        est_err = abs(a - b) / mid
        if est_err <= xtol:
            break
    else:  # executed if loop didn't break
        flag = -2  # CONVERR
        if disp: raise RuntimeError('Failed to converge to tolerance {0}\
                                     after {1} iterations. Found {2} with\
                                     estimated error {3}.'
                                     .format(tol, maxiter, x, est_err))       
    if full_output:
        r = RootResults(root=mid,
                        iterations=iteration+1,
                        function_calls=2*(iteration+1 + 1),
                        flag=flag)
        return [mid, r]
    return mid


def newton(f, a0, a1=None, fprime=None, fprime2=None,
           maxiter=50, tol=1e-8, full_output=False, disp=True):
    flag = 0
    if fprime is None:
        # use secant method
        def secant_iter_gen(xprev, x, f):
            f_calls = 0
            est_err = 1
            while True:
                yield x, xprev, est_err, f_calls
                xnext = x - f(x) * (x - xprev) / (f(x) - f(xprev))
                xprev = x; x = xnext;
                est_err = abs((x - xprev) / x)
                f_calls += 3
        if a1 is None:  # make one up. Using the method from scipy.optimize:
            delta = 1e-4 if a0 >= 0 else -1e-4
            a1 = a0 * (1 + delta) + delta
        method = secant_iter_gen(a0, a1, f)
    else:
        # use Newton-Raphson method
        def NR_iter_gen(x, f, fprime):
            f_calls = 0
            est_err = 1
            xprev = x
            while True:
                yield x, xprev, est_err, f_calls
                xprev = x
                x = xprev - f(xprev) / fprime(xprev)
                est_err = abs((x - xprev) / x)
                f_calls += 2
        method = NR_iter_gen(a0, f, fprime)
    # TODO: if fprime2 is not None, use Halley's method.

    x, xprev, est_err, f_calls = next(method)
    for iteration in range(maxiter):
        x, xprev, est_err, f_calls = next(method) #.next()
        if est_err <= tol:
            break
    else:  # executed if loop didn't break
        flag = -2  # CONVERR
        if disp: raise RuntimeError('Failed to converge to tolerance {0}\
                                     after {1} iterations. Found {2} with\
                                     estimated error {3}.'
                                     .format(tol, maxiter, x, est_err))
    if full_output:
        r = RootResults(root=x,
                        iterations=iteration+1,
                        function_calls=f_calls,
                        flag=flag)
        return [x, r]
    return x

