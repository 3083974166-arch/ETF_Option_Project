import pytest
from etf_option.pricing import price, greeks, implied_vol, parity, bounds


def test_known_bsm_and_parity():
    assert price(100, 100, 1, .05, .2, 0, "C") == pytest.approx(10.450583572, abs=1e-8)
    c = price(100, 105, .7, .03, .28, .018, "C")
    p = price(100, 105, .7, .03, .28, .018, "P")
    assert c-p == pytest.approx(parity(100, 105, .7, .03, .018), abs=1e-12)


@pytest.mark.parametrize("cp", ["C", "P"])
@pytest.mark.parametrize("method", ["brent", "newton"])
@pytest.mark.parametrize("K,T,sigma", [(90,.2,.2), (110,1,.6), (100,.01,.3), (80,2,.1), (100,1,1.4)])
def test_iv_recovery(cp, method, K, T, sigma):
    p = price(100, K, T, .02, sigma, .013, cp)
    assert implied_vol(p, 100, K, T, .02, .013, cp, method) == pytest.approx(sigma, abs=2e-7)


@pytest.mark.parametrize("cp", ["C", "P"])
def test_finite_difference_greeks(cp):
    S, K, T, r, sig, q = 100., 103., .8, .025, .24, .012
    f = lambda s=S, t=T, rate=r, vol=sig: price(s,K,t,rate,vol,q,cp)
    g = greeks(S,K,T,r,sig,q,cp)
    h = .001
    assert g["delta"] == pytest.approx((f(s=S+h)-f(s=S-h))/(2*h), rel=1e-6)
    assert g["gamma"] == pytest.approx((f(s=S+h)-2*f()+f(s=S-h))/h**2, rel=2e-5)
    h = 1e-5
    assert g["vega"] == pytest.approx((f(vol=sig+h)-f(vol=sig-h))/(2*h), rel=1e-6)
    assert g["rho"] == pytest.approx((f(rate=r+h)-f(rate=r-h))/(2*h), rel=1e-6)
    assert g["theta"] == pytest.approx(-(f(t=T+h)-f(t=T-h))/(2*h), rel=1e-6)


def test_boundaries():
    assert price(110,100,0,.02,.2,.01,"C") == 10
    assert price(90,100,0,.02,.2,.01,"P") == 10
    assert price(100,100,1,.03,0,.01) == bounds(100,100,1,.03,.01,"C")[0]
    with pytest.raises(ValueError):
        implied_vol(150,100,100,1,.02,.01)
    with pytest.raises(ValueError):
        implied_vol(0,100,100,0,.02,.01)
    with pytest.raises(ValueError):
        price(-1,100,1,.02,.2,.01)
    with pytest.raises(ValueError):
        greeks(100,100,0,.02,.2,.01)

