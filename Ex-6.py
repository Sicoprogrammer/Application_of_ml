import numpy as np
from scipy import stats

# Generate synthetic data
np.random.seed(0)  
data1 = np.random.normal(loc=50, scale=10, size=100)  
data2 = np.random.normal(loc=55, scale=10, size=100)  
data3 = np.random.normal(loc=60, scale=10, size=100)  

# Z-test

z_stat, p_value = stats.test(data1, data2)
print("Z-test:")
print("Z-statistic:", z_stat)
print("P-value:", p_value)
if p_value < 0.05:
    print("Reject Null Hypothesis")
else:
    print("Fail to Reject Null Hypothesis")

# t-test

t_stat, p_value = stats.ttest_ind(data1, data3)
print("\nT-test:")
print("T-statistic:", t_stat)
print("P-value:", p_value)
if p_value < 0.05:
    print("Reject Null Hypothesis")
else:
    print("Fail to Reject Null Hypothesis")

# ANOVA

f_stat, p_value = stats.f_oneway(data1, data2, data3)
print("\nANOVA:")
print("F-statistic:", f_stat)
print("P-value:", p_value)
if p_value < 0.05:
    print("Reject Null Hypothesis")
else:
    print("Fail to Reject Null Hypothesis")
