"""Train/tune traditional classifiers using one leakage-safe sklearn pipeline."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import gc
import hashlib
import shutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from scripts.feature_engineering import BuildingFeatures, NativeBuildingFeatures
from scripts.preprocess import ROOT, load_training, make_preprocessor, make_native_preprocessor


def build_pipeline(model_name='random_forest', use_id_features=False):
    """Create feature engineering → ColumnTransformer → traditional classifier."""
    if model_name == 'random_forest':
        estimator = RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
            class_weight='balanced_subsample', n_jobs=4, random_state=42)
    elif model_name == 'logistic_regression':
        estimator = LogisticRegression(class_weight='balanced', max_iter=1500, random_state=42)
    elif model_name == 'native_lightgbm':
        from lightgbm import LGBMClassifier
        return Pipeline([
            ('features', NativeBuildingFeatures(use_id_features=use_id_features)),
            ('preprocess', make_native_preprocessor(use_id_features)),
            ('model', LGBMClassifier(objective='multiclass', num_class=5,
                n_estimators=500, learning_rate=0.04, num_leaves=127,
                min_child_samples=80, colsample_bytree=0.9, reg_lambda=3,
                cat_smooth=30, cat_l2=20, random_state=42, verbosity=-1,
                n_jobs=8, histogram_pool_size=128, force_col_wise=True)),
        ])
    elif model_name == 'lightgbm':
        from lightgbm import LGBMClassifier
        estimator = LGBMClassifier(n_estimators=400, learning_rate=0.05,
            num_leaves=63, class_weight='balanced', n_jobs=4, verbosity=-1,
            histogram_pool_size=128, force_col_wise=True, random_state=42)
    else:
        raise ValueError(f'Unsupported traditional classifier: {model_name}')
    return Pipeline([
        ('features', BuildingFeatures(use_id_features=use_id_features)),
        ('preprocess', make_preprocessor()),
        ('model', estimator),
    ])


def parameter_space(model_name):
    """Return pipeline-qualified parameters for macro-F1 tuning."""
    if model_name == 'random_forest':
        return {'model__max_depth': [12, 20, None], 'model__min_samples_leaf': [2, 5, 10],
                'model__max_features': ['sqrt', 0.5]}
    if model_name == 'logistic_regression':
        return {'model__C': [0.01, 0.1, 1.0, 10.0]}
    return {'model__num_leaves': [31, 63, 127], 'model__reg_lambda': [0.1, 1, 5],
            'model__min_child_samples': [30, 60, 100]}


def run_training(model_name='random_forest', tune=False, sample_size=None,
                 use_id_features=False, output=None):
    """Tune on training folds, evaluate a reserved holdout, then refit all supplied rows.

    The holdout is never passed to the hyperparameter search. The reported score
    belongs to the pre-refit estimator, while the serialized pipeline uses all
    available training rows for final inference.
    """
    X, y = load_training()
    if sample_size is not None and sample_size < len(X):
        X, _, y, _ = train_test_split(X, y, train_size=sample_size, stratify=y, random_state=42)
    X_fit, X_valid, y_fit, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipeline = build_pipeline(model_name, use_id_features)
    best_params = {}
    print(f'Training {model_name}: {len(X_fit):,} training rows, {len(X_valid):,} holdout rows; tuning={tune}', flush=True)
    if tune:
        search = RandomizedSearchCV(pipeline, parameter_space(model_name), n_iter=4,
            scoring='f1_macro', cv=StratifiedKFold(3, shuffle=True, random_state=42),
            random_state=42, n_jobs=1, refit=True, error_score='raise')
        search.fit(X_fit, y_fit)
        pipeline, best_params = search.best_estimator_, search.best_params_
    else:
        pipeline.fit(X_fit, y_fit)
    predicted = pipeline.predict(X_valid)
    print(f'Holdout F1-macro: {f1_score(y_valid, predicted, average="macro"):.6f}', flush=True)
    report = {'model': model_name, 'metric': 'F1-macro',
              'holdout_macro_f1': f1_score(y_valid, predicted, average='macro'),
              'rows_used': len(X), 'holdout_rows': len(X_valid), 'tuning': tune,
              'use_id_features': use_id_features, 'best_params': best_params,
              'classification_report': classification_report(y_valid, predicted, output_dict=True),
              'note': 'Held-out score before refit; distinct from historical V2 cross-validation.'}
    destination = output or ROOT / 'models/model.pkl'
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f'Refitting on {len(X):,} rows and saving to {destination}', flush=True)
    pipeline.fit(X, y)
    joblib.dump(pipeline, destination)
    report_path = destination.with_suffix('.metrics.json')
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if sample_size is None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        figure_dir = ROOT / 'reports/figures'
        figure_dir.mkdir(parents=True, exist_ok=True)
        ConfusionMatrixDisplay(confusion_matrix(y_valid, predicted, labels=np.arange(1, 6)),
                               display_labels=np.arange(1, 6)).plot()
        plt.savefig(figure_dir / 'holdout_confusion_matrix.png', bbox_inches='tight', dpi=150)
        plt.close()
        (ROOT / 'reports/model_metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def macro_f1(y, predicted):
    """Compute five-class macro-F1 efficiently for integer labels 1–5."""
    matrix = np.bincount(5*(np.asarray(y, dtype=int)-1)+np.asarray(predicted)-1,
                         minlength=25).reshape(5, 5)
    return float(np.mean(2*np.diag(matrix)/np.maximum(matrix.sum(0)+matrix.sum(1), 1)))


def optimize_decisions(probabilities, y):
    """Tune four relative class weights on selection fold 0 only; Grade 5 is the anchor."""
    weights = np.ones(5)
    best = macro_f1(y, probabilities.argmax(1)+1)
    for span in [0.6, 0.25, 0.1]:
        for column in range(4):
            center = weights[column]
            winning = weights.copy()
            for value in np.linspace(max(0.3, center-span), center+span, 25):
                trial = weights.copy()
                trial[column] = value
                score = macro_f1(y, (probabilities*trial).argmax(1)+1)
                if score > best:
                    best, winning = score, trial.copy()
            weights = winning
    return weights


def restored_manifest():
    """Fingerprint data, code and library versions to prevent stale checkpoint reuse."""
    import lightgbm
    import sklearn
    paths = ['data/train.csv','data/test.csv','data/sample_submission.csv',
             'scripts/train.py','scripts/preprocess.py','scripts/feature_engineering.py']
    hashes = {}
    for relative in paths:
        with (ROOT/relative).open('rb') as handle:
            hashes[relative] = hashlib.file_digest(handle, 'sha256').hexdigest()
    return {'sha256': hashes, 'seed':42, 'folds':5, 'n_estimators':500,
            'lightgbm':lightgbm.__version__, 'sklearn':sklearn.__version__}


def run_restored_cv():
    """Retrain five native-category Pipelines and confirm macro-F1 on legacy fold indices.

    Fixed 500-tree settings are inherited from the historical experiment range,
    not tuned on confirmation folds. Category vocabularies are now fold-local.
    Decision multipliers are tuned only on fold 0 and frozen for folds 1–4.
    """
    from scripts.predict import FoldPipelineEnsemble
    models = ROOT/'models'
    reports = ROOT/'reports'
    models.mkdir(exist_ok=True)
    reports.mkdir(exist_ok=True)
    manifest_path = models/'native_manifest.json'
    manifest = restored_manifest()
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Native checkpoint inputs/code changed. Move native_* artifacts before starting a new experiment.')
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    X, y_series = load_training()
    y = y_series.to_numpy()
    test = pd.read_csv(ROOT/'data/test.csv')
    sample = pd.read_csv(ROOT/'data/sample_submission.csv')
    if not test.building_id.is_unique or set(test.building_id) != set(sample.building_id):
        raise ValueError('Invalid test/sample IDs.')
    splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X,y))
    oof = np.zeros((len(X),5))
    test_probabilities = np.zeros((len(test),5))
    rows = []
    for fold, (training, validation) in enumerate(splits):
        checkpoint = models/f'native_fold_{fold}.pkl'
        prediction_path = reports/f'native_fold_{fold}.npz'
        if prediction_path.exists() and checkpoint.exists():
            saved = np.load(prediction_path)
            valid_probability, test_probability = saved['validation'], saved['test']
            print(f'Resumed fold {fold+1}/5', flush=True)
        else:
            if checkpoint.exists():
                pipeline = joblib.load(checkpoint)
                print(f'Loaded trained fold {fold+1}/5; resuming inference', flush=True)
            else:
                pipeline = build_pipeline('native_lightgbm', use_id_features=True)
                counts = np.bincount(y[training], minlength=6)[1:]
                training_weights = np.sqrt(counts.mean()/counts)[y[training]-1]
                print(f'Training native Pipeline fold {fold+1}/5 on {len(training):,} rows', flush=True)
                pipeline.fit(X.iloc[training], y[training], model__sample_weight=training_weights)
                joblib.dump(pipeline, checkpoint)
            print(f'Predicting validation/test fold {fold+1}/5', flush=True)
            valid_probability = pipeline.predict_proba(X.iloc[validation])
            test_probability = pipeline.predict_proba(test)
            np.savez_compressed(prediction_path, validation=valid_probability, test=test_probability)
            del pipeline
            gc.collect()
        oof[validation] = valid_probability
        test_probabilities += test_probability/5
        if fold == 0:
            decision_weights = optimize_decisions(valid_probability, y[validation])
        score = macro_f1(y[validation], (valid_probability*decision_weights).argmax(1)+1)
        rows.append({'fold':fold, 'role':'selection' if fold==0 else 'confirmation',
                     'macro_f1':score, 'raw_macro_f1':macro_f1(y[validation],valid_probability.argmax(1)+1)})
        pd.DataFrame(rows).to_csv(reports/'native_fold_scores.csv',index=False)
        print(f'Fold {fold+1}/5 F1-macro={score:.6f}', flush=True)
    predictions = (oof*decision_weights).argmax(1)+1
    confirmation = np.concatenate([indices for _,indices in splits[1:]])
    report = {'model':'native LightGBM, ensemble of five sklearn Pipelines',
        'metric':'F1-macro', 'cv_macro_f1':macro_f1(y,predictions),
        'confirmation_macro_f1':macro_f1(y[confirmation],predictions[confirmation]),
        'decision_weights':decision_weights.tolist(), 'rows':len(X),
        'classification_report':classification_report(y,predictions,output_dict=True),
        'protocol':'StratifiedKFold(5, shuffle=True, random_state=42); decision weights selected on fold 0; fold-local categories; fixed 500 trees. Historical hyperparameters were previously selected on this dataset; this is not a new unbiased benchmark.',
        'historical_cv_macro_f1':0.5828382922377816}
    (reports/'native_model_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    np.savez_compressed(reports/'native_predictions.npz', oof=oof, test=test_probabilities, y=y,
                        decision_weights=decision_weights)
    ensemble = FoldPipelineEnsemble([f'native_fold_{fold}.pkl' for fold in range(5)], decision_weights.tolist())
    joblib.dump(ensemble, models/'native_ensemble.pkl')
    mapping = pd.Series((test_probabilities*decision_weights).argmax(1)+1,index=test.building_id)
    submission = sample.copy()
    submission['damage_grade'] = submission.building_id.map(mapping)
    if submission.isna().any().any() or not submission.damage_grade.isin(range(1,6)).all():
        raise ValueError('Invalid predictions.')
    submission.damage_grade = submission.damage_grade.astype('int64')
    submission.to_csv(ROOT/'submissions/submission_native.csv',index=False)
    print(json.dumps(report,indent=2),flush=True)
    return report


def main():
    """Parse CLI options and train the requested pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['random_forest', 'logistic_regression', 'lightgbm'], default='random_forest')
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--sample-size', type=int)
    parser.add_argument('--use-id-features', action='store_true')
    parser.add_argument('--output', type=str)
    parser.add_argument('--restore-v2', action='store_true', help='Retrain native-category pipelines using the historical five-fold split.')
    args = parser.parse_args()
    if args.restore_v2:
        if args.tune or args.sample_size is not None or args.output is not None or args.use_id_features:
            parser.error('--restore-v2 uses its own fixed settings and cannot be combined with these options.')
        run_restored_cv()
        return
    if args.sample_size is not None and args.output is None:
        parser.error('Provide --output for sample runs to protect the full-data model.')
    print(json.dumps(run_training(args.model, args.tune, args.sample_size,
                                  args.use_id_features, args.output), indent=2))


if __name__ == '__main__':
    main()
