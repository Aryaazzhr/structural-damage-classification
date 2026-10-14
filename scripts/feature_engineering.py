"""Stateless building features compatible with scikit-learn pipelines."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class BuildingFeatures(TransformerMixin, BaseEstimator):
    """Create structural ratios and optional ID patterns without learning target statistics."""

    def __init__(self, use_id_features=False):
        """Configure whether empirical building-ID patterns are included."""
        self.use_id_features = use_id_features

    def fit(self, X, y=None):
        """Record input columns; no validation or target statistics are estimated."""
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        """Return a copy with ratios, missing-age flags, and categorical geography."""
        check_is_fitted(self, 'feature_names_in_')
        frame = X.copy()
        frame['age_unknown'] = (frame['age_building'] >= 995).astype(int)
        frame.loc[frame['age_building'] >= 995, 'age_building'] = np.nan
        frame['height_per_floor'] = frame['height_ft_pre_eq'] / frame['count_floors_pre_eq'].clip(lower=1)
        frame['area_per_floor'] = frame['plinth_area_sq_ft'] / frame['count_floors_pre_eq'].clip(lower=1)
        frame['volume_proxy'] = frame['plinth_area_sq_ft'] * frame['height_ft_pre_eq']
        for column in ['district_id', 'vdcmun_id', 'ward_id']:
            frame[column] = frame[column].astype(str)
        if self.use_id_features:
            frame['id_prefix_6'] = frame['building_id'].astype(str).str[:6]
            frame['id_suffix_6'] = frame['building_id'] % 1_000_000
        return frame.drop(columns=['building_id', 'damage_grade'], errors='ignore')


NUMERIC_FEATURES = ['count_floors_pre_eq', 'age_building', 'plinth_area_sq_ft',
                    'height_ft_pre_eq', 'age_unknown', 'height_per_floor',
                    'area_per_floor', 'volume_proxy', 'area_height_ratio',
                    'log_plinth_area', 'log_age', 'log_height']
CATEGORICAL_FEATURES = ['district_id', 'vdcmun_id', 'ward_id',
    'land_surface_condition', 'foundation_type', 'roof_type', 'ground_floor_type',
    'other_floor_type', 'position', 'plan_configuration', 'superstructure',
    'age_bin', 'floor_bin']


class NativeBuildingFeatures(BuildingFeatures):
    """Recreate V2 structural and empirical ID features as an sklearn transformer."""

    def transform(self, X):
        """Create stateless V2 features; leave unknown age missing for LightGBM."""
        frame = super().transform(X)
        frame['area_height_ratio'] = frame.plinth_area_sq_ft / frame.height_ft_pre_eq.clip(lower=1)
        frame['log_plinth_area'] = np.log1p(frame.plinth_area_sq_ft)
        frame['log_age'] = np.log1p(frame.age_building)
        frame['log_height'] = np.log1p(frame.height_ft_pre_eq)
        frame['age_bin'] = pd.cut(frame.age_building, bins=[0,5,10,20,30,50,100,999],
            labels=['0-5','5-10','10-20','20-30','30-50','50-100','100+'], include_lowest=True).astype(str)
        frame['floor_bin'] = frame.count_floors_pre_eq.clip(upper=5).astype(str)
        columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        if self.use_id_features:
            frame['id_prefix_8'] = X.building_id.astype(str).str[:8]
            frame['id_suffix_4'] = X.building_id % 10000
            columns = columns + ['id_prefix_6','id_prefix_8','id_suffix_6','id_suffix_4']
        return frame[columns]


class RegionalBuildingFeatures(NativeBuildingFeatures):
    """Add fold-local regional summaries and stateless material/ID interactions."""

    def fit(self, X, y=None):
        """Learn ward medians/counts from training features only; never use targets."""
        super().fit(X, y)
        clean = X.copy()
        clean.loc[clean.age_building >= 995, 'age_building'] = np.nan
        self.regional_columns_ = ['age_building', 'plinth_area_sq_ft', 'height_ft_pre_eq']
        self.global_medians_ = clean[self.regional_columns_].median()
        self.ward_medians_ = clean.groupby('ward_id')[self.regional_columns_].median()
        self.ward_counts_ = clean.ward_id.value_counts()
        return self

    def transform(self, X):
        """Add regional ratios, ID prefixes and material combinations without target leakage."""
        frame = super().transform(X)
        check_is_fitted(self, 'ward_medians_')
        for column in self.regional_columns_:
            median = X.ward_id.map(self.ward_medians_[column]).fillna(self.global_medians_[column])
            frame[f'ward_median_{column}'] = median
            frame[f'ward_relative_{column}'] = frame[column] / median.clip(lower=1)
        frame['ward_training_count'] = X.ward_id.map(self.ward_counts_).fillna(0)
        if self.use_id_features:
            identifier = X.building_id.astype(str)
            frame['id_prefix_9'] = identifier.str[:9]
            frame['id_prefix_10'] = identifier.str[:10]
        for first, second in [('district_id', 'superstructure'),
                              ('foundation_type', 'superstructure'),
                              ('roof_type', 'ground_floor_type')]:
            frame[f'{first}__{second}'] = X[first].astype(str) + '|' + X[second].astype(str)
        return frame
