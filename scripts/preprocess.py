"""Load competition data and define fold-local preprocessing."""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from scripts.feature_engineering import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]


class NativeCategories(TransformerMixin, BaseEstimator):
    """Learn categorical vocabularies per training fold, preserving pandas categories."""

    def fit(self, X, y=None):
        """Store sorted category levels from the supplied training rows only."""
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.categories_ = {column: sorted(X[column].astype(str).unique()) for column in X.columns}
        return self

    def transform(self, X):
        """Convert known levels to categoricals and unseen levels to missing values."""
        check_is_fitted(self, 'categories_')
        return pd.DataFrame({column: pd.Categorical(X[column].astype(str), categories=levels)
                             for column, levels in self.categories_.items()}, index=X.index)

    def get_feature_names_out(self, input_features=None):
        """Expose unchanged feature names for ColumnTransformer pandas output."""
        check_is_fitted(self, 'feature_names_in_')
        return self.feature_names_in_.copy()


def make_native_preprocessor(use_id_features=True):
    """Preserve V2 column order and native missing/category handling in ColumnTransformer."""
    transformers = [('numeric', 'passthrough', NUMERIC_FEATURES),
                    ('categorical', NativeCategories(), CATEGORICAL_FEATURES)]
    if use_id_features:
        transformers.extend([('id_categories', NativeCategories(), ['id_prefix_6','id_prefix_8']),
                             ('id_numeric', 'passthrough', ['id_suffix_6','id_suffix_4'])])
    return ColumnTransformer(transformers, verbose_feature_names_out=False).set_output(transform='pandas')


def load_training(path=None):
    """Read training rows and map Grade 1–5 labels to integers 1–5."""
    frame = pd.read_csv(path or ROOT / 'data/train.csv')
    target = pd.to_numeric(frame['damage_grade'].astype(str).str.replace('Grade ', '', regex=False), errors='raise')
    if not target.isin(range(1, 6)).all():
        raise ValueError('Training labels must be Grade 1–5 or integers 1–5.')
    if not frame.building_id.is_unique:
        raise ValueError('Training building IDs must be unique.')
    return frame.drop(columns='damage_grade'), target.astype(int)


def make_preprocessor():
    """Build median imputation/scaling and categorical imputation/one-hot encoding.

    All learned statistics are fitted inside the outer model pipeline, separately
    for each cross-validation split. Rare categories limit feature dimensionality.
    """
    numeric = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler(with_mean=False)),
    ])
    categorical = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', min_frequency=5,
                                  max_categories=128, sparse_output=True)),
    ])
    return ColumnTransformer([
        ('numeric', numeric, make_column_selector(dtype_include='number')),
        ('categorical', categorical, make_column_selector(dtype_exclude='number')),
    ], sparse_threshold=1.0)
