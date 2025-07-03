import pandas as pd
import numpy as np
from scipy import stats

def segment_data(data, feature, value=None, exclude_values=None):
    """
    Segment the data based on a feature. Optionally filter by value or exclude certain values.
    """
    if exclude_values is not None:
        data_segment = data[~data[feature].isin(exclude_values)]
    else:
        data_segment = data.copy()

    if value is not None:
        data_segment = data_segment[data_segment[feature] == value]
    
    return data_segment

def check_identical_values(data, metric):
    """
    Check if all values for a metric are identical.
    """
    unique_values = data[metric].dropna().unique()
    return len(unique_values) == 1

def chi_squared_test(data, feature, metric):
    """
    Perform chi-squared test for categorical data.
    """
    contingency_table = pd.crosstab(data[feature], data[metric])
    chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
    return chi2, p_value

def t_test(group_a, group_b, metric):
    """
    Perform a t-test between two groups on a given metric.
    """
    if check_identical_values(group_a, metric) or check_identical_values(group_b, metric):
        print(f"Warning: All values for {metric} are identical in one of the groups. Skipping t-test.")
        return None, None

    t_stat, p_value = stats.ttest_ind(group_a[metric].dropna(), group_b[metric].dropna(), nan_policy='omit')
    return t_stat, p_value

def z_test(group_a, group_b, metric):
    """
    Perform a z-test between two groups if sample size is large (>30).
    """
    mean_a, mean_b = group_a[metric].mean(), group_b[metric].mean()
    std_a, std_b = group_a[metric].std(), group_b[metric].std()
    n_a, n_b = group_a[metric].count(), group_b[metric].count()

    z_stat = (mean_a - mean_b) / np.sqrt((std_a**2 / n_a) + (std_b**2 / n_b))
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    return z_stat, p_value

def interpret_p_value(p_value, alpha=0.05):
    """
    Interpret the null hypothesis based on the p-value.
    """
    if p_value is None:
        return "Test skipped due to identical values."
    return "Reject the null hypothesis." if p_value < alpha else "Fail to reject the null hypothesis."

def risk_across_provinces(data):
    """
    Test for risk differences across provinces using Chi-Squared test on TotalPremium.
    """
    chi2, p_value = chi_squared_test(data, 'Province', 'TotalPremium')
    return f"Chi-squared test on Province and TotalPremium: chi2 = {chi2}, p-value = {p_value}\n" + interpret_p_value(p_value)

def risk_between_postalcodes(data):
    """
    Test for risk differences between postal codes using Chi-Squared test.
    """
    chi2, p_value = chi_squared_test(data, 'PostalCode', 'TotalPremium')
    return f"Chi-squared test on PostalCode and TotalPremium: chi2 = {chi2}, p-value = {p_value}\n" + interpret_p_value(p_value)

def margin_between_postalcodes(data):
    """
    Test for margin differences between postal codes using t-test or z-test on TotalPremium.
    """
    postal_codes = data['PostalCode'].unique()
    if len(postal_codes) < 2:
        return "Not enough unique postal codes for testing."

    group_a = segment_data(data, 'PostalCode', value=postal_codes[0])
    group_b = segment_data(data, 'PostalCode', value=postal_codes[1])
    
    if len(group_a) > 30 and len(group_b) > 30:
        z_stat, p_value = z_test(group_a, group_b, 'TotalPremium')
        return f"Z-test on TotalPremium: Z-statistic = {z_stat}, p-value = {p_value}\n" + interpret_p_value(p_value)
    else:
        t_stat, p_value = t_test(group_a, group_b, 'TotalPremium')
        return f"T-test on TotalPremium: T-statistic = {t_stat}, p-value = {p_value}\n" + interpret_p_value(p_value)

def risk_between_genders(data):
    """
    Test for risk differences between Men and Women using t-test on TotalPremium.
    """
    data = segment_data(data, 'Gender', exclude_values=['Not Specified'])

    group_a = segment_data(data, 'Gender', value='Male')
    group_b = segment_data(data, 'Gender', value='Female')

    if group_a.empty or group_b.empty:
        return "One of the gender groups is empty. Test cannot be performed."

    t_stat, p_value = t_test(group_a, group_b, 'TotalPremium')
    return f"T-test on TotalPremium: T-statistic = {t_stat}, p-value = {p_value}\n" + interpret_p_value(p_value)

def result_hypothesis(data):
    """
    Run all hypothesis tests and return the results.
    """
    results = {
        'Risk Differences Across Provinces': risk_across_provinces(data),
        'Risk Differences Between Postal Codes': risk_between_postalcodes(data),
        'Margin Differences Between Postal Codes': margin_between_postalcodes(data),
        'Risk Differences Between Women and Men': risk_between_genders(data),
    }
    return results
