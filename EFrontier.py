
import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import scipy.optimize as sc
import plotly.graph_objects as go

#import data
def getData(stocks, start, end):
   stockData = yf.download(stocks, start=start, end=end)
   stockData = stockData['Close']

   returns = stockData.pct_change()
   meanReturns = returns.mean()
   covMatrix = returns.cov()
   return meanReturns, covMatrix


def portfolioPerformance(weights, meanReturns, covMatrix):
    returns = np.sum(meanReturns*weights)*252
    std = np.sqrt( np.dot(weights.T, np.dot(covMatrix, weights))) * np.sqrt(252)
    return returns, std

#finds the negative sharp ratio of a portfolio
def negativeSR(weights, meanReturns, covMatix, riskFreeRate = 0 ):
   pReturns, pStd= portfolioPerformance(weights, meanReturns, covMatrix)
   return -(pReturns - riskFreeRate)/pStd

def maxSR(meanReturns, covMatrix, riskFreeRate = 0, constraintSet =(0,1)):
   "Minimize the negative sharp ratio, by altering the weights of the portfolio"
   numAssets = len(meanReturns)
   args = (meanReturns, covMatrix, riskFreeRate)
   constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) -1})
   bound = constraintSet
   bounds = tuple(bound for asset in range(numAssets) )
   result = sc.minimize(negativeSR, numAssets*[1./numAssets], args=args,
                        method= 'SLSQP', bounds=bounds, constraints=constraints)
   return result


def portfolioVariance(weights, meanReturns, covMatrix):
   return portfolioPerformance(weights, meanReturns, covMatrix) [1]


def minimizeVariance(meanReturns, covMatrix, constraintSet=(0,1)):
   """Minimize the portfolio variance by altering the weights of the portfolio"""
   numAssets = len(meanReturns)
   args = (meanReturns, covMatrix)
   constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) -1})
   bound = constraintSet
   bounds = tuple(bound for asset in range(numAssets) )
   result = sc.minimize(portfolioVariance, numAssets*[1./numAssets], args=args,
                        method= 'SLSQP', bounds=bounds, constraints=constraints)
   return result

#Define stocks and get data
stocklist = ['APPL', 'TSLA', 'MSFT',]
stocks = [stock for stock in stocklist]

#Define time period
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=365)

#Get mean returns and covariance matrix
meanReturns, covMatrix = getData(stocks, start=startDate, end=endDate)

#Define target returns for efficient frontier
def portfolioReturns(weights, meanReturns, covMatrix):
      return portfolioPerformance(weights, meanReturns, covMatrix) [0]

# Calculate Efficient Frontier
def efficientFrontier(meanReturns, covMatrix, returnTargets, constraintSet=(0,1) ):
   """For each target return, find the portfolio with the minimum variance"""
   numAssets = len(meanReturns)
   args = (meanReturns, covMatrix)

   constraints = ({'type': 'eq', 'fun': lambda x: portfolioReturns(x, meanReturns, covMatrix) - returnTargets},
                     {'type': 'eq', 'fun': lambda x: np.sum(x) -1})
   bound = constraintSet
   bounds = tuple(bound for asset in range(numAssets) )
   efficientFrontier = sc.minimize(portfolioVariance, numAssets*[1./numAssets], args=args,
                                   method= 'SLSQP', bounds=bounds, constraints=constraints)
   return efficientFrontier



# Calculate and output results
def calculatedResults(meanReturns, covMatrix, riskFreeRate=0, contraintSet=(0,1) ):
   """read in mean, cov matrix, and other financial information 
      out put, max sharp ratio, mean volatility portfolio, efficient fronteir """
   # Calculate Max Sharp Ratio Portfolio
   maxSR_Portfolio = maxSR(meanReturns, covMatrix)
   maxSR_returns, maxSR_std = portfolioPerformance(maxSR_Portfolio['x'], meanReturns, covMatrix)
   
   maxSR_allocation = pd.DataFrame (maxSR_Portfolio['x'], index=meanReturns.index, columns= ['allocation'] )
   maxSR_allocation.allocation = [round(i*100,2) for i in maxSR_allocation.allocation]



   # Calculate Min Variance Portfolio
   minVol_Portfolio = minimizeVariance(meanReturns, covMatrix)
   minVol_returns, minVol_std = portfolioPerformance(minVol_Portfolio['x'], meanReturns, covMatrix)
   
   minVol_allocation = pd.DataFrame (minVol_Portfolio['x'], index=meanReturns.index, columns= ['allocation'] )
   minVol_allocation.allocation = [round(i*100,2) for i in minVol_allocation.allocation]

   # Efficient Frontier
   efficientList = []
   targetReturns = np.linspace(minVol_returns, maxSR_returns, 20)
   for target in targetReturns: 
         efficientList.append(efficientFrontier(meanReturns, covMatrix, target)['fun'])
   
   maxSR_returns = round(maxSR_returns*100, 2), round (maxSR_std*100,2)
   minVol_returns = round(minVol_returns*100, 2), round (minVol_std*100,2)

   return maxSR_returns, maxSR_std, maxSR_allocation, minVol_returns, minVol_std, minVol_allocation, efficientList, targetReturns

print (calculatedResults(meanReturns, covMatrix),)


def EF_graph(meanReturns, covMatrix, riskFreeRate=0, contraintSet=(0,1) ):
   """ Return a graph ploting the min vol, max sr and efficient frontier"""
   maxSR_returns, maxSR_std, maxSR_allocation, minVol_returns, minVol_std, minVol_allocation, efficientList, targetReturns = calculatedResults(meanReturns, covMatrix, riskFreeRate, contraintSet)

   fig = go.Figure()

   # Max SR point
   MaxSharpRatio = go.Scatter(
       name='Max Sharp Ratio',
       mode = 'markers',
       x=[maxSR_std],
       y=[maxSR_returns],
       marker = dict(color='red', size=14, line = dict(width =3, color='black')),
   )
      # Min vol point
   MinVol = go.Scatter(
       name='Min Volatility',
       mode = 'markers',
       x=[minVol_std],
       y=[minVol_returns],
       marker = dict(color='red', size=14, line = dict(width =3, color='black')),
   )
      # Efficient Frontier line
   EF_Curve = go.Scatter(
       name='Efficient Frontier',
       mode = 'lines',
       x=[round(ef_std*100,2) for ef_std in efficientList],
       y=[round(target*100,2) for target in targetReturns],
       line = dict(color='black', width =4, dash='dashdot' ),
   )

   data = [MaxSharpRatio, MinVol, EF_Curve]

   layout = go.Layout(
       title = 'Portfolio Optomization with the Efficient Frontier',
       xaxis = dict(title='Annualized Volatility (%)'),
       yaxis = dict(title='Annualized Returns (%)'),
       showlegend = True,
       legend = dict (
           x=0.75, y=0, traceorder='normal', 
           bgcolor="#C2B6B6", 
           bordercolor='black', 
           borderwidth=2),
      width =800,
      height=600)
   fig = go.Figure(data=data, layout=layout)
   return fig.show()


EF_graph(meanReturns, covMatrix)