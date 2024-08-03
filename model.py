from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler

class TitanicModelPipeline:
    def __init__(self):
        self.pipeline = Pipeline(steps=[
            ('preprocessor', ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_features),
                    ('cat', OneHotEncoder(), categorical_features)
                ])),
            ('feature_selection', SelectKBest(chi2)),
            ('classifier', RandomForestClassifier())
        ])
        self.grid_search = None

    def fit(self, X, y):
        param_grid = {
            'feature_selection__k': [2, 3],
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20]
        }
        self.grid_search = GridSearchCV(self.pipeline, param_grid, cv=5, scoring='accuracy')
        self.grid_search.fit(X, y)

    def evaluate(self, X_test, y_test):
        return {
            'accuracy': self.grid_search.score(X_test, y_test)
        }

    def best_params(self):
        return self.grid_search.best_params()
