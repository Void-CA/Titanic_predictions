import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utilities import set_multiple_columns_datatype

class CorrelationMatrix:
    def __init__(self, dataframe):
        self.correlation_matrix = None
        self.numeric_df = None
        self.dataframe = dataframe

    def filter_numeric_columns(self):
        """
        Filtra solo las columnas numéricas del dataframe.
        """
        self.numeric_df = self.dataframe.select_dtypes(include=['number', 'bool'])

    def compute_correlation_matrix(self):
        """
        Calcula la matriz de correlación para las columnas numéricas.
        """
        self.correlation_matrix = self.numeric_df.corr()

    def plot_correlation_matrix(self):
        """
        Grafica la matriz de correlación utilizando un mapa de calor.
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
        plt.title('Matriz de Correlación')
        plt.show()


df = pd.read_csv('data/train.csv')
columns = {"Pclass":'category', 'Embarked':'category', "Sex":'category'}
train = set_multiple_columns_datatype(df, columns)
df_dummies = pd.get_dummies(df.drop(['Cabin', 'Name', 'Ticket'], axis=1))
correlation_matrix = CorrelationMatrix(df_dummies)
correlation_matrix.filter_numeric_columns()
correlation_matrix.compute_correlation_matrix()
correlation_matrix.plot_correlation_matrix()
