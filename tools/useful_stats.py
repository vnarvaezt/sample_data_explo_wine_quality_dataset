import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def find_outliers_by_std(df, std_dev_thr=3):
    std_dev_threshold = 3
    mean = np.mean(df)
    std_dev = np.std(df)
    r = f"Outliers Check: More than {std_dev_thr} std far from mean"
    for var in df:
        outliers = df[
            (df[var] < std_dev.loc[var])
            | (df[var] > (mean.loc[var] + std_dev_threshold))
        ]
        if len(outliers) > 0:
            r += f"""
            {var} .... {len(outliers)} observations ({(len(outliers) / len(df)):.1%})
            """
    print(r)


def find_outliers_by_iqr(df):
    r = f"Outliers Check: IQR method"
    for var in df:
        # Calculate the first quartile (Q1) and third quartile (Q3)
        q1 = np.percentile(df[var], 25)
        q3 = np.percentile(df[var], 75)
        # Calculate the interquartile range (IQR)
        iqr = q3 - q1
        # Set the threshold for outliers based on the IQR method
        iqr_threshold_lower = q1 - 1.5 * iqr
        iqr_threshold_upper = q3 + 1.5 * iqr
        # Find outliers using the threshold values
        outliers_iqr = df[
            (df[var] < iqr_threshold_lower) | (df[var] > (iqr_threshold_upper))
        ]
        # Print the outliers
        if len(outliers_iqr) > 0:
            r += f"""
            {var} .... {len(outliers_iqr)} observations ({(len(outliers_iqr) / len(df)):.1%})
            """
    print(r)


def compute_khi2_test(df, target, alpha):
    # Create a contingency table from the dataset
    x = df.drop(target, axis=1)
    dependent_list = []
    for var in x:
        contingency_table = pd.crosstab(df[target], df[var])
        # Chi-Square test
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        # Check if the result is statistically significant at alpha
        if p < alpha:
            dependent_list.append(var)
    return dependent_list
