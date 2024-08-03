from pandas import DataFrame
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

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


class Notebook:
    """
    A class to handle data preprocessing for a machine learning task.

    Attributes
    ----------
    test : pandas.DataFrame
        The test dataset.
    X_train : pandas.DataFrame
        The training features.
    y_train : pandas.Series
        The training labels.
    X_test : pandas.DataFrame
        The testing features.
    y_test : pandas.Series
        The testing labels.
    X_true_test : pandas.DataFrame
        The preprocessed test features ready for prediction.
    """

    def __init__(self, train_file='data/train.csv', test_file='data/test.csv'):
        """
        Initializes the Notebook class by loading and preprocessing data.

        Parameters
        ----------
        train_file : str, optional
            Path to the training data file (default is 'data/train.csv').
        test_file : str, optional
            Path to the test data file (default is 'data/test.csv').

        Sets
        ----------
        self. Test : pandas.DataFrame
            The test dataset.
        self.X_train : pandas.DataFrame
            The training features.
        self.y_train : pandas.Series
            The training labels.
        self.X_test : pandas.DataFrame
            The testing features.
        self.y_test : pandas.Series
            The testing labels.
        self.X_true_test : pandas.DataFrame
            The preprocessed test features ready for prediction.
        """
        self.columns = {"Pclass": 'category', 'Embarked': 'category', "Sex": 'category'}
        self.test = pd.read_csv(test_file)
        self.train = pd.read_csv(train_file)

        # Preprocess the training data
        self.train_dummies = pd.get_dummies(self.train.drop(['Cabin', 'Name', 'Ticket'], axis=1))
        self.X = self.train_dummies.drop(['Survived', 'PassengerId'], axis=1)
        self.y = self.train['Survived']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        # Preprocess the test data
        self.test_new = set_multiple_columns_datatype(self.test, self.columns)
        self.X_true_test = pd.get_dummies(self.test_new.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1))


class SimpleNNPyTorch(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleNNPyTorch, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
