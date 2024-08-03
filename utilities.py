from pandas import DataFrame
import pandas as pd

conversion_functions = {
    'int': lambda series: pd.to_numeric(series, errors='coerce').astype('Int64'),
    'float': lambda series: pd.to_numeric(series, errors='coerce'),
    'bool': lambda series: series.map(
        {'Yes': True, 'No': False, 'yes': True, 'no': False, 'true': True, 'false': False, 'True': True,
         'False': False}).astype('bool'),
    'category': lambda series: series.astype('category'),
    'datetime': lambda series: pd.to_datetime(series, errors='coerce'),
    'string': lambda series: series.astype('str')
}


def set_column_datatype(df: DataFrame, column: str, datatype: str) -> DataFrame:
    """
    Set the datatype of a specific column in a DataFrame.

    Parameters:
    df (DataFrame): The DataFrame containing the column.
    column (str): The name of the column to convert.
    datatype (str): The datatype to convert the column to. Supported values are 'int', 'float', 'bool', 'category', 'datetime'.

    Returns:
    DataFrame: The DataFrame with the converted column.
    """
    if datatype in conversion_functions:
        df[column] = conversion_functions[datatype](df[column])
    else:
        raise ValueError(f"Unsupported datatype: {datatype}")
    return df


def set_multiple_columns_datatype(df: DataFrame, columns: dict) -> DataFrame:
    """
    Set the datatype of multiple columns in a DataFrame.

    Parameters:
    df (DataFrame): The DataFrame containing the columns.
    columns (dict): A dictionary where keys are column names and values are the datatypes to convert to.

    Returns:
    DataFrame: The DataFrame with the converted columns.
    """
    for column, datatype in columns.items():
        df = set_column_datatype(df, column, datatype)
    return df


def group_rare_categories(df, column, threshold=50):
    """
    Groups less frequent categories into a common category 'Others'.

    :param df: DataFrame with the data.
    :param column: Name of the categorical column.
    :param threshold: Frequency threshold for considering a category as 'Others'.
    :return: DataFrame with the updated column.
    """
    # Ensure the column is categorical
    df[column] = df[column].astype('category')

    # Calculate the frequency of each category
    value_counts = df[column].value_counts()

    # Print value counts for debugging
    print(f"Value counts for column '{column}':\n{value_counts}")

    # Identify categories that appear less than the threshold
    less_frequent = value_counts[value_counts < threshold].index
    print(f"Less frequent categories for column '{column}':\n{less_frequent}")

    # Replace less frequent categories with 'Others'
    df[column] = df[column].apply(lambda x: 'Others' if x in less_frequent else x).astype('category')
    return df



def extract_date_features(df, date_column):
    """
    Extracts temporal features from a date column.

    :param df: DataFrame with the data.
    :param date_column: Name of the date column.
    :return: DataFrame with new temporal features.
    """
    df[date_column] = pd.to_datetime(df[date_column])
    df['year_' + date_column] = df[date_column].dt.year
    df['month_' + date_column] = df[date_column].dt.month

    return df


