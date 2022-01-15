# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 20:48:44 2022

@author: James Clark
"""
import pandas as pd
import math
import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Part I
# 1

bitcoin = pd.read_csv('BTC_USD_2014-11-03_2021-12-31-CoinDesk.csv')
snp = pd.read_csv('SP500_DailyIndex_2010-2021.csv')

# Set a datetime index
bitcoin['Datetime'] = pd.to_datetime(bitcoin['Date'])
bitcoin = bitcoin.set_index(['Datetime'])
del bitcoin['Date']

snp['Datetime'] = pd.to_datetime(snp['Date'])
snp = snp.set_index(['Datetime'])
del snp['Date']

btc_price = bitcoin['Closing Price (USD)']
snp_price = snp['Close']

# Compute daily returns
btc_daily_rets = btc_price.pct_change()
snp_daily_rets = snp_price.pct_change()

# 2

# Compute monthly returns
btc_cum_rets = btc_daily_rets.resample('M').agg(lambda r: (r + 1).prod()- 1)
snp_cum_rets = snp_daily_rets.resample('M').agg(lambda r: (r + 1).prod()- 1)

# 3

famafrench_rf = pd.read_csv('FF3Factors_Monthly.csv')

# Set a datetime index
famafrench_rf['Datetime'] = pd.to_datetime(famafrench_rf['Date'])
famafrench_rf = famafrench_rf.set_index(['Datetime'])
del famafrench_rf['Date']

famafrench_rf = famafrench_rf.div(100)

# 4

# Merging all 3 dataframes & renaming columns
first_merge = famafrench_rf.merge(btc_cum_rets, how = 'left', 
                                  left_index = True, right_index = True)
merged = first_merge.merge(snp_cum_rets, how = 'left', 
                                  left_index = True, right_index = True)
merged.columns = ['Mkt-RF', 'SMB', 'HML', 'RF', 'BTC', 'S&P']

# 5

final = merged.loc['2014-11-30':'2021-11-30']

# Add columms for excess returns
final['BTC Excess'] = final['BTC'] - final['RF']
final['S&P Excess'] = final['S&P'] - final['RF']

# Part II
# 1

btc_excess = final['BTC'] - final['RF']
snp_excess = final['S&P'] - final['RF']

btc_mean = btc_excess.mean()
btc_mean = (1 + btc_mean)**12-1

snp_mean = snp_excess.mean()
snp_mean = (1 + snp_mean)**12-1

btc_stdev = btc_excess.std()*math.sqrt(12)
snp_stdev = snp_excess.std()*math.sqrt(12)

btc_sharpe = btc_mean/btc_stdev
snp_sharpe = snp_mean/snp_stdev

correlation = np.corrcoef(btc_excess, snp_excess)[0,1]

btc_skew = skew(btc_excess)
snp_skew = skew(snp_excess)

btc_kurtosis = kurtosis(btc_excess)
snp_kurtosis = kurtosis(snp_excess)


# 2

final.plot(y = ['BTC', 'S&P'])
plt.title('BTC & S&P Cumulative Returns')
plt.show()

# 3

final.hist(column = 'BTC', bins = 10)
plt.xlabel('Excess Return')
plt.ylabel('Number of Occurrences')

# 4

final.plot.scatter(x = 'S&P', y = 'BTC')
plt.title('BTC & S&P Excess Returns')
plt.show()

# 5

capm_regression = smf.ols('Q("BTC Excess") ~ Q("S&P Excess")',
                          data = final).fit()

# 6

ff_regression = smf.ols('Q("BTC Excess") ~ Q("Mkt-RF") + Q("SMB") + Q("HML")', 
                        data = final).fit()

# Part III
# 1

hf_rets = pd.read_csv('HFR_CrpytoCurrency_IndexReturns.csv')

# Set a datetime index
hf_rets['Datetime'] = pd.to_datetime(hf_rets['Date'])
hf_rets = hf_rets.set_index(['Datetime'])
del hf_rets['Date']

# 2

complete_df = final.merge(hf_rets,how = 'left', 
                                  left_index = True, right_index = True)


complete_df.columns = ['Mkt-RF', 'SMB', 'HML', 'RF', 'BTC', 'S&P',
                       'BTC Excess','S&P Excess', 'HFR']

# Add column for HFR excess returns
complete_df['HFR Excess'] = complete_df['HFR'] - complete_df['RF']

# Drop first 2 rows of dataframe since there is no HFR data
complete_df = complete_df.iloc[2: , :]

# 3

hfr_excess = complete_df['HFR'] - complete_df['RF']

hfr_mean = hfr_excess.mean()
hfr_mean = (1 + hfr_mean)**12-1

hfr_stdev = hfr_excess.std()*math.sqrt(12)
hfr_sharpe = hfr_mean/hfr_stdev
hfr_skew = skew(hfr_excess)
hfr_kurtosis = kurtosis(hfr_excess)

# Value at Risk

var = hfr_excess.quantile(0.05)

# 4

hfr_vs_btc = smf.ols('Q("HFR Excess") ~ Q("BTC Excess")', 
                      data = complete_df).fit()

alpha = hfr_vs_btc.params['Intercept']
alpha_t = hfr_vs_btc.tvalues['Intercept']
beta = hfr_vs_btc.params['Q("BTC Excess")']
beta_t = hfr_vs_btc.tvalues['Q("BTC Excess")']

x = complete_df['BTC Excess']
y = complete_df['HFR Excess']
x1 = sm.add_constant(x)
reg = sm.OLS(y, x1).fit()
reg.summary()
predict_vals = reg.predict()
resid = complete_df['HFR Excess'] - predict_vals
IR = reg.params['const']/resid.std()

# Compute stdev of residuals
# hmm = hfr_vs_btc.get_influence()
# residuals = hmm.resid_studentized_internal
# res_stdev = residuals.std()

# info_ratio = alpha/res_stdev

# 5

complete_df['Max BTC Excess'] = np.maximum(complete_df["BTC Excess"], 0)

timing_regress = smf.ols(
    'Q("HFR Excess") ~ Q("BTC Excess")+ Q("Max BTC Excess")', 
                      data = complete_df).fit()

# Outputs

print('')
print('')
print('Part II')
print(f'BTC mean annual return = {btc_mean}')
print(f'S&P mean annual return = {snp_mean}')
print('')
print(f'BTC annual standard deviation = {btc_stdev}')
print(f'S&P annual standard deviation = {snp_stdev}')
print('')
print(f'BTC Sharpe Ratio = {btc_sharpe}')
print(f'S&P Sharpe Ratio = {snp_sharpe}')
print('')
print(f'Correlation between BTC and S&P = {correlation}')
print('')
print(f'BTC skew = {btc_skew}')
print(f'S&P skew = {snp_skew}')
print('')
print(f'BTC kurtosis = {btc_kurtosis}')
print(f'S&P kurtosis = {snp_kurtosis}')
print('')
print(capm_regression.summary())
print('')
print(ff_regression.summary())
print('')
print('Part III')
print('')
print(f'HFR mean annual return = {hfr_mean}')
print('')
print(f'HFR annual standard deviation = {hfr_stdev}')
print('')
print(f'HFR Sharpe Ratio = {hfr_sharpe}')
print('')
print(f'HFR skew = {hfr_skew}')
print('')
print(f'HFR kurtosis = {hfr_kurtosis}')
print('')
print(f'Value at Risk = {var}')
print('')
print(hfr_vs_btc.summary())
print('')
print(f'Information Ratio = {IR}')
print('')
print(timing_regress.summary())
      


