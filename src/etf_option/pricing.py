"""BSM with continuous implied carry q; prices and Greeks per ETF share.

Vega/rho are per unit (1.00) change, theta per calendar year.
Newton is safeguarded by a bracket; Brent is the reference solver.
"""
from math import exp, log, sqrt, isfinite
from scipy.special import ndtr
from scipy.optimize import brentq


def _validate(S, K, T, r, q, sigma=0):
    if not all(isfinite(x) for x in [S, K, T, r, q, sigma]):
        raise ValueError("Non-finite pricing input")
    if S <= 0 or K <= 0 or T < 0 or sigma < 0:
        raise ValueError("Require S,K > 0 and T,sigma >= 0")


def _sign(cp):
    if cp not in ("C", "P"):
        raise ValueError("cp must be C or P")
    return 1 if cp == "C" else -1


def bounds(S, K, T, r, q, cp):
    _validate(S, K, T, r, q)
    z = _sign(cp)
    a, b = S * exp(-q*T), K * exp(-r*T)
    return max(z*(a-b), 0), a if z == 1 else b


def price(S, K, T, r, sigma, q, cp="C"):
    _validate(S, K, T, r, q, sigma)
    z = _sign(cp)
    if T == 0 or sigma == 0:
        return bounds(S, K, T, r, q, cp)[0]
    d1 = (log(S/K)+(r-q+sigma*sigma/2)*T)/(sigma*sqrt(T))
    d2 = d1-sigma*sqrt(T)
    return float(z*(S*exp(-q*T)*ndtr(z*d1)-K*exp(-r*T)*ndtr(z*d2)))


def greeks(S, K, T, r, sigma, q, cp="C"):
    _validate(S, K, T, r, q, sigma)
    z = _sign(cp)
    if T == 0 or sigma == 0:
        raise ValueError("Greeks undefined at T=0 or sigma=0; handle expiry separately")
    d1 = (log(S/K)+(r-q+sigma*sigma/2)*T)/(sigma*sqrt(T))
    d2 = d1-sigma*sqrt(T)
    pdf = exp(-d1*d1/2)/sqrt(2*3.141592653589793)
    discq, discr = exp(-q*T), exp(-r*T)
    return dict(delta=z*discq*ndtr(z*d1), gamma=discq*pdf/(S*sigma*sqrt(T)),
                vega=S*discq*pdf*sqrt(T),
                theta=-S*discq*pdf*sigma/(2*sqrt(T))-z*r*K*discr*ndtr(z*d2)
                      +z*q*S*discq*ndtr(z*d1),
                rho=z*K*T*discr*ndtr(z*d2))


def parity(S, K, T, r, q):
    _validate(S, K, T, r, q)
    return S*exp(-q*T)-K*exp(-r*T)


def implied_vol(value, S, K, T, r, q, cp="C", method="brent", tol=1e-10):
    if method not in ("brent", "newton"):
        raise ValueError("Unknown solver")
    lo_price, hi_price = bounds(S, K, T, r, q, cp)
    if not isfinite(value) or T <= 0 or value < lo_price-tol or value >= hi_price:
        raise ValueError("Price outside invertible no-arbitrage interval")
    if abs(value-lo_price) <= tol:
        return 0.0
    f = lambda vol: price(S, K, T, r, vol, q, cp)-value
    low, high = 1e-9, 1.0
    while f(high) < 0 and high < 16:
        high *= 2
    if f(high) < 0:
        raise ValueError("IV not bracketed below 1600%; inspect price")
    if method == "brent":
        return float(brentq(f, low, high, xtol=tol))
    x = min(.25, high)
    for _ in range(100):
        fx = f(x)
        if abs(fx) < tol:
            return x
        if fx > 0:
            high = x
        else:
            low = x
        v = greeks(S, K, T, r, x, q, cp)["vega"]
        trial = x-fx/v if v > 1e-12 else float("nan")
        x = trial if low < trial < high else (low+high)/2
    raise RuntimeError("Newton did not converge")

