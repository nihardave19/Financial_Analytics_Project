


Explanation of Columns:
Count: Number of observations (daily returns) for each asset.
Mean: Average daily return.
Std Dev: Standard deviation of daily returns (a measure of volatility).
Min: Minimum daily return over the period.
25th Percentile: 25th percentile of daily returns.
Median: Median (50th percentile) daily return.
75th Percentile: 75th percentile of daily returns.
Max: Maximum daily return over the period.
Skew: Skewness, indicating asymmetry in the return distribution.
Kurtosis: Kurtosis, indicating the "tailedness" of the distribution (higher values imply more extreme values).


Explanation of Each Part:
Data Download: Retrieves the last three years of adjusted closing prices for each stock.
Returns Calculation: Calculates daily returns using percentage changes.
Descriptive Statistics: Provides comprehensive descriptive statistics for each asset, including skewness and kurtosis.
Visualization:
Price & Returns Histogram: Plots the time series of closing prices and histograms for daily returns for each asset.
Correlation Heatmap: Displays a heatmap for the correlation matrix, with color coding for easy interpretation.




Explanation of the Code:
Portfolio Performance Calculation: The portfolio_performance function calculates the annualized return and risk (standard deviation) of a portfolio given a set of weights.

Random Portfolios for Efficient Frontier: We generate 5,000 random portfolios and calculate their returns, risks, and Sharpe ratios. These are used to plot the efficient frontier.

Maximum Sharpe Portfolio (Market Portfolio): We identify the portfolio with the highest Sharpe ratio, which we assume to be the market portfolio.

Capital Market Line (CML): The CML is plotted from the risk-free rate to the market portfolio.

Security Market Line (SML): Using the market portfolio as the benchmark, we calculate the beta and expected return of RELIANCE.NS and plot it on the SML.

Output:
Efficient Frontier and CML Plot: Shows the Markowitz curve with the CML extending from the risk-free rate to the market portfolio.
SML Plot: Displays the Security Market Line, with the selected asset (RELIANCE.NS) plotted according to its beta and expected return.



Explanation:
Return: We calculate the weighted average return of the combined index, annualized by multiplying by 252 (the typical number of trading days in a year).

Risk: The portfolio's risk (volatility) is calculated using the covariance matrix of returns, scaled to annualized risk by multiplying by 
25
2
1
/
2
252 
1/2
 .

Output:
The Combined Index Return is the weighted average annual return of all the assets in the portfolio.
The Combined Index Risk is the annualized standard deviation of the combined portfolio’s returns, representing its volatility.
This will give you both the return and risk of your combined index, providing a summary of the risk-return profile of the synthetic portfolio you created.


