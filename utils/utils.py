import pandas as pd
from scipy.stats import chi2_contingency

def quantify_chi2_target(df, categorical_columns, target_column):
    """
    Quantify the relationship between categorical features and the target variable using chi-square test.

    Parameters:
    - df: DataFrame containing the data.
    - categorical_columns: List of categorical column names to quantify.
    - target_column: Name of the target variable column.

    Returns:
    - results: DataFrame containing chi-square statistic and p-value for each feature.
    """
    results = []

    for column in categorical_columns:
        contingency_table = pd.crosstab(df[column], df[target_column])
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        results.append({'Feature': column, 'Chi-square': chi2, 'p-value': p_value})

    results_df = pd.DataFrame(results)
    return results_df


def print_description(table, path):
    df_description = pd.read_csv(path)
    for _,row in df_description[df_description.Table == table][[ 'Row', 'Description']].iterrows():
        col = row.Row
        description = row.Description
        print( f'{col}: {description}')