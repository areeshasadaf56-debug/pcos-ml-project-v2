# Report: Infertility Dataset

## Dataset Overview
- Total samples after cleaning: 540
- Class balance (PCOS Y/N): {0: 363, 1: 177}

## Best-Performing Model
- **XGBoost** achieved the highest accuracy on full features: **0.7407**.
- Full comparison table available in `comparison_table.csv`.

## Effect of Feature Reduction
- Average accuracy (Full features): 0.6495
- Average accuracy (PCA): 0.6151
- Average accuracy (SelectKBest-RFE): 0.6283
- Feature reduction hurt performance slightly on average (best reduced-feature average accuracy 0.6283 vs full-feature average 0.6495).

## Limitations and Caveats
- Dataset size is modest (541 rows before cleaning), so results may vary across different train/test splits.
- Class balance should be checked above; if imbalanced, accuracy alone can be misleading — precision/recall/F1 in `comparison_table.csv` give a fuller picture.
- PCA components are linear combinations of original features and lose direct clinical interpretability compared to SelectKBest's original-feature selection.
- Default hyperparameters were used for all models; GridSearchCV blocks are available (commented out) in `models.py` for further tuning.
