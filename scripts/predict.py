"""Generate a Kaggle submission using the serialized complete sklearn pipeline."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import gc
from scripts.preprocess import ROOT


class FoldPipelineEnsemble:
    """Average five independently fitted sklearn pipelines with macro-F1 decision weights."""

    def __init__(self, model_files, weights):
        """Store portable filenames relative to models/ and class-ordered decision weights."""
        self.model_files = model_files
        self.weights = weights
        self.classes_ = np.arange(1, 6)

    def predict_proba(self, X):
        """Load one fold at a time and return unweighted average class probabilities."""
        probabilities = np.zeros((len(X), 5))
        for filename in self.model_files:
            pipeline = joblib.load(ROOT / 'models' / filename)
            if not np.array_equal(pipeline.classes_, self.classes_):
                raise ValueError('Unexpected fold class order.')
            probabilities += pipeline.predict_proba(X) / len(self.model_files)
            del pipeline
            gc.collect()
        return probabilities

    def predict(self, X):
        """Apply frozen class decision weights and return integer grades 1–5."""
        return self.classes_[(self.predict_proba(X) * np.asarray(self.weights)).argmax(1)]


def generate_submission(model_path=None, output=None, test_path=None, sample_path=None):
    """Predict test features, align to sample IDs, and write validated integer grades."""
    model = joblib.load(model_path or ROOT / 'models/model.pkl')
    test = pd.read_csv(test_path or ROOT / 'data/test.csv')
    sample = pd.read_csv(sample_path or ROOT / 'data/sample_submission.csv')
    if not test.building_id.is_unique or not sample.building_id.is_unique:
        raise ValueError('Test and sample IDs must be unique.')
    if set(test.building_id) != set(sample.building_id):
        raise ValueError('Test and sample must contain the same IDs.')
    if sample.columns.tolist() != ['building_id', 'damage_grade']:
        raise ValueError('Unexpected sample submission schema.')
    mapping = pd.Series(model.predict(test), index=test.building_id)
    sample['damage_grade'] = sample.building_id.map(mapping)
    if sample.isna().any().any() or not sample.damage_grade.isin(range(1, 6)).all():
        raise ValueError('Predictions must be nonmissing integer grades 1–5.')
    sample['damage_grade'] = sample.damage_grade.astype('int64')
    destination = Path(output or ROOT / 'submissions/submission.csv')
    destination.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(destination, index=False)
    return sample


def main():
    """Parse model/output paths and write the submission."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT / 'models/model.pkl')
    parser.add_argument('--output', type=Path, default=ROOT / 'submissions/submission.csv')
    args = parser.parse_args()
    result = generate_submission(args.model, args.output)
    print(f'Saved {len(result):,} rows to {args.output}')


class WeightedFoldPipelineEnsemble(FoldPipelineEnsemble):
    """Combine differently sized sets of fold pipelines with explicit model weights."""

    def __init__(self, model_files, weights, model_weights):
        """Store class decision multipliers and positive per-pipeline mixing weights."""
        super().__init__(model_files, weights)
        coefficients = np.asarray(model_weights, dtype=float)
        if len(coefficients) != len(model_files) or not np.isfinite(coefficients).all() or (coefficients <= 0).any():
            raise ValueError('Each model requires one finite positive mixing weight.')
        self.model_weights = coefficients / coefficients.sum()

    def predict_proba(self, X):
        """Load each pipeline once and combine probabilities in integer class order."""
        probabilities = np.zeros((len(X), 5))
        for filename, coefficient in zip(self.model_files, self.model_weights):
            pipeline = joblib.load(ROOT / 'models' / filename)
            if not np.array_equal(pipeline.classes_, self.classes_):
                raise ValueError('Unexpected fold class order.')
            probabilities += coefficient * pipeline.predict_proba(X)
            del pipeline
            gc.collect()
        return probabilities


if __name__ == '__main__':
    main()
