import pandas as pd

def aggregate_categorical(df, categorical_columns, id_column='SK_ID_CURR'):
    agg_results = []
    for col in categorical_columns:
        grouped = df.groupby([id_column, col]).size().unstack(fill_value=0).reset_index()
        new_cols = [f'{col}_{cat}_COUNT' for cat in grouped.columns[1:]]
        grouped.columns = [grouped.columns[0]] + new_cols
        grouped.set_index(id_column, inplace=True)

        agg_results.append(grouped)

    final_result = pd.concat(agg_results, axis=1)

    return final_result