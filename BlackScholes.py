#implementing the black-scholes formula in python

import numpy as np 
from scipy.stats import norm

#define variables
#intrest rate
r=0.01 
#underlying 
s=30
#strike price
K=40
#time 
T=240/365
#volatility
sigma=0.30

#first function to calculate black scholes call option price
def blackScholes(r, s, K, T, sigma, type="c"):
    "Calculate BS option price for a call/put"
    d1 = (np.log(s / K) + (r + sigma**2/2) *T ) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    try:
        if type == "c":
            price = s * norm.cdf(d1,0,1) - K * np.exp(-r * T) * norm.cdf(d2,0,1)
        elif type == "p":
            price = K * np.exp(-r * T) * norm.cdf(-d2,0,1) - s * norm.cdf(-d1,0,1)
        return price
    except: 
        print("pleace confirm all option parameters are correct!!!")

print("option price is: ", round(blackScholes(r, s, K, T, sigma, type="p"),2))