# CONTROLS/simple_pendulum/CLF

note that this error returned by the rk.py in scipy.integrate.solve_ivp is circumvented by changing the method rk_step(fun, t, y, f, h, A, B, C, K) as follows:

```python
def rk_step(fun, t, y, f, h, A, B, C, K):
    """Perform a single Runge-Kutta step.

    This function computes a prediction of an explicit Runge-Kutta method and
    also estimates the error of a less accurate method.

    Notation for Butcher tableau is as in [1]_.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system.
    t : float
        Current time.
    y : ndarray, shape (n,)
        Current state.
    f : ndarray, shape (n,)
        Current value of the derivative, i.e., ``fun(x, y)``.
    h : float
        Step to use.
    A : ndarray, shape (n_stages, n_stages)
        Coefficients for combining previous RK stages to compute the next
        stage. For explicit methods the coefficients at and above the main
        diagonal are zeros.
    B : ndarray, shape (n_stages,)
        Coefficients for combining RK stages for computing the final
        prediction.
    C : ndarray, shape (n_stages,)
        Coefficients for incrementing time for consecutive RK stages.
        The value for the first stage is always zero.
    K : ndarray, shape (n_stages + 1, n)
        Storage array for putting RK stages here. Stages are stored in rows.
        The last row is a linear combination of the previous rows with
        coefficients

    Returns
    -------
    y_new : ndarray, shape (n,)
        Solution at t + h computed with a higher accuracy.
    f_new : ndarray, shape (n,)
        Derivative ``fun(t + h, y_new)``.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4.
    """
    K[0] = f.reshape(np.size(f),)
    for s, (a, c) in enumerate(zip(A[1:], C[1:]), start=1):
        dy = np.dot(K[:s].T, a[:s]) * h
        K[s] = fun(t + c * h, y + dy).reshape(np.size(f),)

    y_new = y + h * np.dot(K[:-1].T, B)
    f_new = fun(t + h, y_new)

    K[-1] = f_new.reshape(np.size(f),)

    return y_new, f_new
```