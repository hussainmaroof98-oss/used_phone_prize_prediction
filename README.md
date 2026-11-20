# Used Phone Price Prediction

A regression project that predicts the normalized resale price of a used phone from its specs, built step by step to practice the full machine learning cycle.

## Dataset

- Kaggle "Used Handheld Device Data" (3,454 phones, 15 columns)
- Target: `normalized_used_price`
- Features: brand, OS, 4G/5G, screen size, cameras, RAM, storage, battery, weight, release year, days used, normalized new price

## Workflow

1. **EDA**: checked shapes, data types, missing values, skewness, and correlations with price.
2. **Train/test split first** (80/20, `random_state=42`) so nothing from the test set leaks into preprocessing.
3. **Missing values**: filled with the median of the same release year, with medians learned from the training set only. 179 rear-camera values were missing, mostly from 2019-2020.
4. **Encoding**: one-hot encoding for brand, OS, 4G, 5G (`drop_first=True`), then aligned test columns to train.
5. **Scaling**: `StandardScaler` on numeric columns, fit on train and applied to test.
6. **Models**: Linear, Ridge, Lasso, Polynomial (degree 2) + Ridge, Random Forest (tuned with `GridSearchCV`).
7. **Checked the missing-camera worry** by adding a "was missing" flag column. Its weight was about zero, so the simple fill was good enough.

## Results (test set)

| Model | Test R2 | Test RMSE |
|---|---|---|
| Linear Regression | 0.836 | 0.231 |
| Ridge / Lasso | 0.838 | 0.229 |
| Polynomial (deg 2) + Ridge | 0.857 | 0.215 |
| Random Forest (tuned) | 0.857 | 0.215 |

Best model: **Polynomial + Ridge** (ties with the tuned forest, but simpler and easier to explain). Train-test R2 gap is about 0.01, so it generalizes well. Mean absolute error is about 0.17.

## Key lessons

- Look at the data before cleaning it.
- Split first, and learn every fill and scale value from the training set only (avoids data leakage).
- Overfitting shows up as a gap between train and test scores, not as a high score.
- Ridge and depth limits reduce overfitting; tune with cross-validation on train, never on test.
- Lasso zeroed out 27 of 48 features, mostly brands, because `normalized_new_price` already carries most of the brand premium.

## Limitations and next steps

- The dataset has no phone condition, exact model, or battery health, which limits how accurate any model can be.
- Only polynomial features were tried. Next: feature engineering (phone age, log transforms of skewed columns, grouping rare brands, camera and battery ratios).
- Compare models with cross-validation instead of a single split.

## Tools

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Google Colab.
