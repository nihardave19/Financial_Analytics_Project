import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

# Define the Indian assets to analyze
assets = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "HINDUNILVR.NS", 
          "ITC.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "LT.NS", "BAJFINANCE.NS"]

# Download the last 3 years of daily price data
data = yf.download(assets, start="2021-01-01", end="2024-01-01")['Adj Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Function to plot time series and histogram of returns
def plot_asset_data(asset):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Plot time series of closing prices
    ax[0].plot(data[asset])
    ax[0].set_title(f"{asset} Closing Prices (Last 3 Years)")
    ax[0].set_xlabel("Date")
    ax[0].set_ylabel("Price (INR)")

    # Plot histogram of returns
    sns.histplot(returns[asset], kde=True, ax=ax[1], color="skyblue", edgecolor="black")
    ax[1].set_title(f"{asset} Daily Returns Histogram")
    ax[1].set_xlabel("Daily Return")
    ax[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


# Calculate descriptive statistics for each asset
desc_stats = returns.describe(percentiles=[0.25, 0.5, 0.75 ,0.95 ,0.99]).T
desc_stats['skew'] = returns.skew()
desc_stats['kurtosis'] = returns.kurtosis()

# Rename columns for better clarity
desc_stats = desc_stats.rename(columns={'count': 'Count', 'mean': 'Mean', 'std': 'Std Dev',
                                        'min': 'Min', '25%': '25th Percentile', '50%': 'Median',
                                        '75%': '75th Percentile', 'max': 'Max'})

# Print descriptive statistics
print("Descriptive Statistics for Each Asset:\n", desc_stats)



# # Calculate descriptive statistics for each asset
# desc_stats = returns.describe().T
# desc_stats["skew"] = returns.skew()
# desc_stats["kurtosis"] = returns.kurt()

# # Print descriptive statistics
# print("Descriptive Statistics for Each Asset:\n", desc_stats)

# Plot data for each asset
for asset in assets:
    plot_asset_data(asset)



# Correlation Analysis
correlation_matrix = returns.corr()

# Print the correlation matrix
print("Correlation Matrix of Returns:\n", correlation_matrix)

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix of Returns")
plt.show()



##### Come up with a combined index which is the weighted average of all the picked assets

# Define weights for each asset (assuming equal weights for simplicity)
weights = np.array([1/len(assets)] * len(assets))  # Equal weights for all assets

# Calculate weighted daily returns of the combined index
weighted_returns = returns.dot(weights)

# Calculate cumulative returns to get the index level over time
cumulative_returns = (1 + weighted_returns).cumprod()

# Plot the combined index
plt.figure(figsize=(10, 6))
plt.plot(cumulative_returns, label="Combined Index")
plt.title("Combined Index Based on Weighted Average of Selected Assets")
plt.xlabel("Date")
plt.ylabel("Index Level")
plt.legend()
plt.show()

# Print descriptive statistics for the combined index
index_desc_stats = weighted_returns.describe(percentiles=[0.25, 0.5, 0.75,0.95 ,0.99])
index_desc_stats['skew'] = weighted_returns.skew()
index_desc_stats['kurtosis'] = weighted_returns.kurtosis()

print("Descriptive Statistics for the Combined Index:\n", index_desc_stats)



#####Present the Markowitz curve, CML and SML for a security that you may like out of 10 assets selected.


# Define risk-free rate
risk_free_rate = 0.05 / 252  # Annualized risk-free rate, adjusted to daily

# Step 1: Calculate portfolio returns, risks, and Sharpe ratios for random portfolios
def portfolio_performance(weights, returns):
    port_return = np.sum(weights * returns.mean()) * 252
    port_stddev = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return port_return, port_stddev

# Generate random portfolios
num_portfolios = 5000
all_weights = np.zeros((num_portfolios, len(assets)))
returns_array = np.zeros(num_portfolios)
stddev_array = np.zeros(num_portfolios)
sharpe_array = np.zeros(num_portfolios)

for i in range(num_portfolios):
    weights = np.random.random(len(assets))
    weights /= np.sum(weights)
    all_weights[i, :] = weights
    
    port_return, port_stddev = portfolio_performance(weights, returns)
    returns_array[i] = port_return
    stddev_array[i] = port_stddev
    sharpe_array[i] = (port_return - risk_free_rate) / port_stddev

# Step 2: Find the portfolio with the maximum Sharpe ratio (Market Portfolio)
max_sharpe_idx = sharpe_array.argmax()
max_sharpe_return = returns_array[max_sharpe_idx]
max_sharpe_stddev = stddev_array[max_sharpe_idx]
max_sharpe_weights = all_weights[max_sharpe_idx]

# Step 3: Plot Efficient Frontier (Markowitz Curve) and CML
plt.figure(figsize=(12, 8))
plt.scatter(stddev_array, returns_array, c=sharpe_array, cmap='viridis', marker="o")
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Portfolio Risk (Standard Deviation)')
plt.ylabel('Portfolio Return')
plt.title('Markowitz Efficient Frontier and Capital Market Line')

# Plot CML (Capital Market Line)
x = np.linspace(0, max_sharpe_stddev, 100)
y = risk_free_rate + (max_sharpe_return - risk_free_rate) / max_sharpe_stddev * x
plt.plot(x, y, 'r--', label="Capital Market Line (CML)")
plt.scatter(max_sharpe_stddev, max_sharpe_return, c='red', marker="*", s=200, label="Market Portfolio")
plt.legend()

# Step 4: Calculate SML (Security Market Line) for chosen asset (e.g., RELIANCE.NS)
chosen_asset = "RELIANCE.NS"
market_portfolio_returns = returns.dot(max_sharpe_weights)
beta = np.cov(returns[chosen_asset], market_portfolio_returns)[0, 1] / np.var(market_portfolio_returns)
expected_return = risk_free_rate + beta * (max_sharpe_return - risk_free_rate)

# Plot SML
plt.figure(figsize=(12, 6))
plt.plot([0, 2], [risk_free_rate, risk_free_rate + 2 * (max_sharpe_return - risk_free_rate)], label="Security Market Line (SML)", color="purple")
plt.scatter(beta, expected_return, color="blue", s=100, label=f"{chosen_asset}")
plt.xlabel("Beta")
plt.ylabel("Expected Return")
plt.title("Security Market Line (SML)")
plt.legend()
plt.show()

# Print portfolio statistics for the selected asset and market portfolio
print(f"Market Portfolio Expected Return: {max_sharpe_return:.4f}, Risk: {max_sharpe_stddev:.4f}")
print(f"Selected Asset ({chosen_asset}) - Beta: {beta:.4f}, Expected Return: {expected_return:.4f}")

# Print portfolio statistics for the selected asset and market portfolio in percentage terms
print(f"Market Portfolio Expected Return: {max_sharpe_return * 100:.2f}%, Risk: {max_sharpe_stddev * 100:.2f}%")
print(f"Selected Asset ({chosen_asset}) - Beta: {beta:.4f}, Expected Return: {expected_return * 100:.2f}%")

#####################Obtain the return and the risk of combined index.

# Calculate the weighted average return of the combined index
combined_index_return = weighted_returns.mean() * 252  # Annualize the return

# Calculate the risk (volatility) of the combined index
combined_index_risk = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))  # Annualized risk

# Print the return and risk of the combined index (annualized)
print(f"Combined Index Return: {combined_index_return:.4f} or {combined_index_return * 100:.2f}%")
print(f"Combined Index Risk: {combined_index_risk:.4f} or {combined_index_risk * 100:.2f}%")
