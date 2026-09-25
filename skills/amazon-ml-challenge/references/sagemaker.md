# Amazon scientist briefing and SageMaker workflow

The intro session walked through Amazon's official example `xgboost_customer_churn`
(github.com/aws/amazon-sagemaker-examples, introduction_to_applying_machine_learning). Its tips,
mapped to action:

| Said in the session | Meaning | Do this |
|---|---|---|
| Data may be skewed to one side; look before touching | skewed target or imbalanced classes | eda.py; log target (reg) or stratify + threshold tuning (clf) |
| Noisy data; EDA is very important | label and input noise | worst-OOF review, curation (techniques.md B) |
| Missing values will exist; handle or drop | NaN handling | GBDT native NaN + is_missing flags; drop TRAIN rows only if validation agrees; NEVER drop test rows |
| float64 / int16 mix; use everything as numbers | numeric matrix | cast consistently, float32, parse numbers out of text |
| Phone number is a unique id | id columns | drop (eda.py flags ID-like columns); keep group ids like product_type_id |
| Charges carry the same info as minutes | redundancy | drop exact functions of other columns (eda.py flags r > 0.995) |
| yes → 1, no → 0; each state its own column | encoding | binary map; one-hot low cardinality; target-encode high cardinality |
| Target at the front, no headers | SageMaker built-in XGBoost CSV rule | only when training with the SageMaker container; inference CSV has no label |
| Upload to S3; bucket in a variable; channels | SageMaker plumbing | see code below |
| train / validation / test; test never shown; seed | holdout discipline | fixed K folds (make_folds.py) + seeds; organiser's test is the real test |
| num_round 100; objective binary | demo hyperparameters | early stopping instead of fixed rounds; objective matched to metric |
| Play with hyperparameters; SageMaker tuning | HPO | Optuna locally (gbdt_cv.py trials=) or SageMaker AMT |
| Autopilot helps with a quick baseline and feature engineering | AutoML | hour-1 tabular baseline + feature ideas; not the final model |
| Golden rule: start simple, layer on it | iterative improvement | one lever per experiment, keep only if OOF improves |
| Threshold 0.5; 0.934 = confident; confidence matters | decision threshold | tune on OOF (blend.py does it); abstain on low confidence for exact-match tasks |
| Precision, recall, F1 is the important parameter | metric choice | understand which errors the metric punishes (metrics.md) |
| S3 5 GB free; stop Jupyter when done | cost | confirm your free tier; stop notebook instances; delete endpoints |
| Live prediction only after deploy; endpoints via API | deployment | NOT needed for a CSV submission; predict locally or Batch Transform |
| "we use patch something" (user's note) | most likely batch inference (Batch Transform) | confirm with organisers |

## Sanitise for SageMaker built-in XGBoost

```python
df = pd.read_csv('raw.csv')
df = df.drop(columns=['phone'] + [c for c in df if c.endswith('_charge')])   # id + redundant
for c in ['intl_plan', 'vmail_plan']:
    df[c] = df[c].map({'yes': 1, 'no': 0})
df = pd.get_dummies(df, columns=['state'], dtype=int)
y = df.pop('churn'); df.insert(0, 'y', y)                                     # target first
df.to_csv('train.csv', header=False, index=False)                             # no header
```
Do the target-first reorder AFTER splitting train / validation / test (Amazon's notebooks do this to
avoid misalignment). Inference CSVs must not contain the target column.

## Minimal SageMaker training + tuning (SDK v2 style; verify against current docs)

```python
import sagemaker, boto3
from sagemaker.inputs import TrainingInput
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, IntegerParameter
sess, role = sagemaker.Session(), sagemaker.get_execution_role()
bucket, prefix = sess.default_bucket(), 'amlc'
boto3.Session().resource('s3').Bucket(bucket).Object(f'{prefix}/train/train.csv').upload_file('train.csv')
boto3.Session().resource('s3').Bucket(bucket).Object(f'{prefix}/validation/validation.csv').upload_file('validation.csv')
img = sagemaker.image_uris.retrieve('xgboost', sess.boto_region_name, '1.7-1')
est = sagemaker.estimator.Estimator(img, role, instance_count=1, instance_type='ml.m5.xlarge',
                                    output_path=f's3://{bucket}/{prefix}/out', sagemaker_session=sess)
est.set_hyperparameters(objective='binary:logistic', num_round=2000, early_stopping_rounds=100,
                        eval_metric='auc', max_depth=5, eta=0.1, subsample=0.8)
ch = {'train': TrainingInput(f's3://{bucket}/{prefix}/train', content_type='text/csv'),
      'validation': TrainingInput(f's3://{bucket}/{prefix}/validation', content_type='text/csv')}
tun = HyperparameterTuner(est, 'validation:auc',
        {'eta': ContinuousParameter(0.01, 0.3), 'max_depth': IntegerParameter(3, 10),
         'min_child_weight': ContinuousParameter(1, 20), 'alpha': ContinuousParameter(0, 5)},
        max_jobs=20, max_parallel_jobs=4)
tun.fit(ch)
# offline scoring for a submission: est.transformer(...).transform(s3_test_csv, content_type='text/csv', split_type='Line')
```

## Autopilot (quick baseline)

Autopilot analyses the data, handles missing values, selects features, tries algorithms with
cross-validation and tunes them. Ensembling mode stacks models; HPO mode uses Bayesian or multi-fidelity
search. It generates data-exploration and candidate notebooks: copy the feature ideas into your own
pipeline. Mostly for tabular data; recent problems were text/image heavy, so treat it as a baseline.

## Cost hygiene

Stop notebook instances when idle; delete endpoints the same day; keep data in one bucket/prefix;
check the free-tier terms on your account before relying on them.
