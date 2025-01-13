import pytest
import math
from src.black_scholes.model import price_put_down_and_in

def test_price_down_and_in():
    S0, K, B, r, sigma, T = 100, 100, 90, 0.01, 0.2, 1.0
    price = price_put_down_and_in(S0, K, B, r, sigma, T, method="mc")
    assert price > 0, "Le prix doit être > 0"
    assert not math.isnan(price), "Le prix ne doit pas être NaN"
