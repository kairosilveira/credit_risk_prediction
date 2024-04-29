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



def get_column_types(df, n_cat = 10):
    """
    Determine the data types of columns in the DataFrame and categorize them into numerical and categorical.

    Parameters:
    - df: DataFrame for which column types need to be determined.
    - n_cat: Num max of unique values in a numeric column to be considered categoric 

    Returns:
    - numerical_columns: List of numerical column names.
    - categorical_columns: List of categorical column names.
    """
    df = df.copy()
    if 'SK_ID_CURR' in df.columns and 'TARGET' in df.columns:
        df = df.drop(columns=['SK_ID_CURR', 'TARGET'], axis=1)
    col_types = df.dtypes

    numerical_columns = []
    categorical_columns = []

    for column in col_types.index:
        if col_types[column] == 'object':
            categorical_columns.append(column)
        elif len(df[column].unique()) <= n_cat:  # Check cardinality for numerical columns
            categorical_columns.append(column)
        else:
            numerical_columns.append(column)

    print("Numerical columns:", len(numerical_columns), numerical_columns)
    print("Categorical columns:", len(categorical_columns), categorical_columns)
    
    return numerical_columns, categorical_columns