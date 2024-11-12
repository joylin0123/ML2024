# preprocessor.py
import numpy as np
import pandas as pd

class Preprocessor:
    """ TODO """ 
    def __init__(self, df, variance_threshold=0.1, correlation_threshold=0.8):
        # Initialize the preprocessor with a DataFrame
        self.df = df
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold

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

    def _add_polynomial_features(self, columns, degree=3):
        new_columns = []

        for col in columns:
            for d in range(2, degree + 1):
                new_col_name = f"{col}^{d}"
                new_columns.append((new_col_name, self.df[col] ** d))

        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                new_col_name = f"{columns[i]}*{columns[j]}"
                new_columns.append((new_col_name, self.df[columns[i]] * self.df[columns[j]]))
        
        new_columns_df = pd.DataFrame(dict(new_columns))
        self.df = pd.concat([self.df, new_columns_df], axis=1)

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

    def _variance_threshold_selector(self, X):
        high_variance_cols = X.columns[X.var() > self.variance_threshold]
        return X[high_variance_cols]

    def _correlation_threshold_selector(self, X):
        correlation_matrix = X.corr().abs()
        upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > self.correlation_threshold)]
        return X.drop(columns=to_drop)

    def preprocess(self, train=True):
        self._drop_index()
        self._standardize_scalar()
        self._mice_imputation()
        numeric_columns = self.df.select_dtypes(include=['int64', 'float64']).columns
        if train:
            self._remove_outliers(numeric_columns)
            self._add_polynomial_features(numeric_columns)
            self._balance_data()
            self.df = self._variance_threshold_selector(self.df)
            self.df = self._correlation_threshold_selector(self.df)
            updated_numeric_columns = self.df.select_dtypes(include=['int64', 'float64']).columns
            self._normalize(updated_numeric_columns)
            y = (self.df['y'] > 0.5).astype(int).to_numpy()
            X = self.df.drop('y', axis=1).to_numpy(dtype=float)
            X_pca = self._pca(X)
            return X_pca, y
        self._normalize(numeric_columns)
        return self.df.to_numpy(dtype=float)
