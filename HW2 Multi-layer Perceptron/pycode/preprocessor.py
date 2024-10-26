# preprocessor.py
import numpy as np
import pandas as pd

class Preprocessor:
    """ TODO """ 
    def __init__(self, df):
        # Initialize the preprocessor with a DataFrame
        self.df = df

    def _drop_index(self):
        if 'Unnamed: 0' in self.df.columns:
            self.df = self.df.drop(['Unnamed: 0'], axis=1)
        return self

    def _standardize_scalar(self):
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                self.df[col] = pd.Categorical(self.df[col]).codes
        return self

    def _mice_imputation(self, iterations=10):
        for column in self.df.columns:
            if self.df[column].isnull().sum() > 0:
                missing_indices = self.df[self.df[column].isnull()].index
                for idx in missing_indices:
                    self.df.loc[idx, column] = self.df[column].mean()
        for _ in range(iterations):
            for column in self.df.columns:
                if self.df[column].isnull().sum() > 0:
                    correlations = self.df.corr()[column].abs().sort_values(ascending=False)
                    top_correlated = correlations[1:11].index.tolist()
                    missing_indices = self.df[self.df[column].isnull()].index
                    for idx in missing_indices:
                        row_values = self.df.loc[idx, top_correlated].dropna()
                        weights = correlations[row_values.index]
                        imputed_value = (row_values * weights).sum() / weights.sum()
                        self.df.loc[idx, column] = imputed_value
        return self

    def _remove_outliers(self, columns, threshold=3):
        for col in columns:
            if self.df[col].dtype in ['int64', 'float64']:
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                self.df = self.df[z_scores < threshold]
        return self

    def _add_polynomial_features(self, columns, degree=2):
        for col in columns:
            for d in range(2, degree + 1):
                new_col_name = f"{col}^2"
                self.df[new_col_name] = self.df[col] ** d
        return self

    def _normalize(self, numeric_columns):
        self.df[numeric_columns] = (
            (self.df[numeric_columns] - self.df[numeric_columns].mean())
            / self.df[numeric_columns].std()
        )
        return self

    def _pca(self, X):
        X_centered = X - np.mean(X, axis=0)
        covariance_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        return np.dot(X_centered, eigenvectors[:, :8])

    def _balance_data(self):
        majority = self.df[self.df['y'] == 0]
        minority = self.df[self.df['y'] == 1]

        num_samples_to_add = len(majority) - len(minority)

        if num_samples_to_add > 0:
            minority_oversampled = minority.sample(n=num_samples_to_add, replace=True, random_state=42)
            self.df = pd.concat([self.df, minority_oversampled])
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)

        return self

    def preprocess(self):
        self._drop_index()
        self._standardize_scalar()
        self._mice_imputation()
        self._balance_data()
        numeric_columns = self.df.select_dtypes(include=['int64', 'float64']).columns
        self._remove_outliers(numeric_columns)
        self._add_polynomial_features(numeric_columns)
        updated_numeric_columns = self.df.select_dtypes(include=['int64', 'float64']).columns
        self._normalize(updated_numeric_columns)
        y = (self.df['y'] > 0.5).astype(int).to_numpy()
        X = self.df.drop('y', axis=1).to_numpy(dtype=float)
        X_pca = self._pca(X)
        return X_pca, y
