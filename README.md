# Akeed Restaurant Recommendation

This competition is hosted on Zindi, a machine learning platform for data science challenges.  
Here is the link to the competition: [Akeed Restaurant Recommendation Challenge 🌾 - $3 000 USD](https://zindi.africa/competitions/akeed-restaurant-recommendation-challenge)

Ranked in the TOP 45%
---

**Competition**: Zindi – Akeed Restaurant Recommendation Challenge
**Date**: 2020
**Language**: R

## Task

Predict which of 100 restaurants each customer will order from, given customer locations, vendor information, and order history. ~35k train customers, ~10k test customers. Submission format: one row per (customer × location × vendor) triplet with a binary prediction.

## Approach

A **geography-first segmented recommendation engine** instead of standard collaborative filtering.

Why: with only 100 vendors, a strong delivery-radius constraint, and sparse interactions (most customers ordered from 1-2 vendors), the dominant signal is **spatial proximity within a demographic segment**, not user-item similarity.

## Engineering Decisions

### 1. Vendor Geographic Zones (K-Means)

Cluster vendors into 4 geographic zones based on coordinates. This creates the spatial structure that constrains recommendations — a customer won't order from a vendor 30km away.

```r
km_vend <- kmeans(
    x = vend_valid[, .(long_clean, lat_clean)],
    centers = 4,
    algorithm = "MacQueen",
    iter.max = 5000,
    nstart = 25
)
```

### 2. Customer Segmentation (18 segments)

Segment customers by: **gender** (3 levels) × **location count** (3 buckets) × **account tenure** (old/new). This captures behavioral groupings — a new male user with 1 address behaves differently from a long-tenured female with multiple delivery locations.

### 3. Vendor Tag Grouping

Original vendor tags are too granular (50+ unique tags). Grouped into 8 cuisine categories (Arabic, Indian, International, Desserts, Drinks, Sandwiches, Breakfast, Others) to make vendor profiles useful for matching.

### 4. Segment × Zone Recommendation

For each (segment, zone) pair in the test set:
- Find train customers in the same segment and zone who ordered
- Count vendor frequency (which vendors they ordered from)
- Score vendors for each test customer: `frequency / manhattan_distance`
- Recommend top-scoring vendors

### 5. Order Feature Engineering

Aggregate order history per (customer × location × vendor) triplet: order count, basket size, payment mode, ratings, delivery times, promo usage. These features characterize the customer-vendor relationship.

## Pipeline

| Step | File | Purpose |
|------|------|---------|
| 0 | `00_config.R` | Seed, paths, libraries |
| 1 | `01_data_loading.R` | Load CSVs, fix casing |
| 2 | `02_order_features.R` | Aggregate order history per triplet |
| 3 | `03_vendor_processing.R` | Tag grouping, coordinate cleanup |
| 4 | `04_customer_features.R` | Customer-location profiles |
| 5 | `05_clustering.R` | Vendor zones + customer segments |
| 6 | `06_recommendation.R` | Segment × zone recommendation |

## Run

```r
source("MAIN.R")
```

## Key Libraries

`data.table`, `dplyr`, `lubridate`, `sqldf`, `caret`
# Mental Health Text Classification

**Competition**: Zindi – AI4D iCompass Social Media Sentiment Analysis for Mental Health
**Date**: Late 2021
**Language**: R

## Task

Classify short free-text statements from Kenyan university students into four mental health categories: Depression, Alcohol, Suicide, Drugs. Statements respond to "What is on your mind?" and contain spelling errors, slang, and code-switching.

## Engineering Decisions

### 1. Spelling Correction Before Tokenization

Student text is noisy—misspellings fragment the vocabulary ("depressd", "deppressed", "depressed" → three separate tokens in TF-IDF). The pipeline applies UDPipe tokenization followed by hunspell correction, unifying variants before feature extraction. Kenyan slang terms (bhang, miraa) are whitelisted to avoid false corrections.

```r
tokens$is_correct <- hunspell_check(tokens$token)
slang_terms <- c("bhang", "miraa", "muguka", "shisha", "weed", "meth")
tokens$is_correct[tokens$token %in% slang_terms] <- TRUE
```

### 2. One-vs-Rest Decomposition

Rather than training a single multi-class model, each class gets its own binary classifier. This lets different models specialize—depression language patterns differ structurally from drug references. The ensemble averages probabilities across models independently per class.

### 3. TF-IDF + LSA (4 latent topics)

Sublinear TF-IDF with L2 normalization handles the short, variable-length statements. LSA with exactly 4 latent dimensions (matching the 4 classes) adds semantic structure to the sparse matrix, giving gradient-boosted models a denser signal.

### 4. Multiple Representation Strategies

The pipeline tests multiple embedding approaches in parallel:
- **TF-IDF + XGBoost**: sparse bag-of-words with boosting
- **GLMNet (Ridge)**: regularized logistic regression, strong on sparse text
- **Keras embedding**: learns task-specific 16-dim embeddings
- **StarSpace (ruimtehol)**: learns joint text-label embedding space

### 5. Simple Average Ensemble

With a small dataset (~600 train samples), a stacking meta-learner risks overfitting. Simple averaging of probability outputs from diverse model families (linear, tree, embedding-based) reduces variance while avoiding that risk.

## Pipeline

| Step | File | Purpose |
|------|------|---------|
| 0 | `00_config.R` | Seed, paths, libraries |
| 1 | `01_data_loading.R` | Load CSV, create binary targets |
| 2 | `02_text_preprocessing.R` | UDPipe + hunspell correction + stemming |
| 3 | `03_feature_engineering.R` | TF-IDF (ngram 1-2) + LSA |
| 4 | `04_model_glmnet.R` | Ridge logistic regression per class |
| 5 | `05_model_xgboost.R` | XGBoost per class with CV |
| 6 | `06_model_keras.R` | Keras text embedding network |
| 7 | `07_model_ruimtehol.R` | StarSpace joint embeddings |
| 8 | `08_ensemble_submission.R` | Average and write submission |

## Run

```r
source("MAIN.R")
```

## Key Libraries

`data.table`, `text2vec`, `udpipe`, `hunspell`, `xgboost`, `glmnet`, `keras`, `ruimtehol`
# Malawi Flood Extent Prediction

Predicting the fraction of 1km² grid squares flooded in southern Malawi during Cyclone Idai (March 2019). Training data is the 2015 flood event. Evaluation: RMSE on flood fraction [0, 1]. Zindi competition sponsored by UNICEF and Arm.

## Key Engineering Decisions

**Train on one flood event, predict another.** The fundamental challenge is not generalization across rows — it's generalization across events. The model trained on 2015 data must predict 2019 (Cyclone Idai), a different storm with a different path, different rainfall distribution, and larger extent. Event-specific features from 2015 (individual week rainfall values) carry limited signal for 2019. The features that generalize are physical landscape invariants: terrain shape, soil properties, proximity to water. The pipeline prioritizes these.

**Terrain derivative features from elevation raster.** Elevation alone doesn't predict flooding — slope, flow direction, and terrain position do. A low-elevation flat plain pools water differently than a low-elevation hillside that drains rapidly. From the raw elevation grid, six terrain derivatives are computed:

```r
terrain_stack <- raster::terrain(
    rast_elev,
    opt = c("slope", "aspect", "tpi", "tri", "roughness", "flowdir"),
    unit = "degrees", neighbors = 8
)
```

- **TPI** (Terrain Position Index): is this cell a depression or a ridge?
- **TRI** (Terrain Ruggedness Index): how rough is the local surface?
- **flowdir**: which of the 8 cardinal directions does water drain toward?
- **hillshade**: a proxy for local exposure and shadow patterns

**Zero-inflated target decomposition.** ~70% of grid squares have zero flood extent. A regression model minimizing RMSE on zero-inflated data will be pulled toward near-zero predictions everywhere. The pipeline creates two auxiliary classification variables from the training target: `bin` (did it flood at all?) and `zero` (no / partial / severe), used for EDA, stratified splitting, and understanding where the regression error originates.

**Prediction clipping.** The target is a bounded fraction [0, 1]. Gradient boosting models can predict outside this range. All predictions are clipped before evaluation and submission:

```r
clip_predictions <- function(x) pmax(0, pmin(1, x))
```

**Soil infiltration variables as primary external data.** Soil drainage class, parent material, erosion type, and texture (from FAO/ISRIC soil maps) directly control the rate at which water infiltrates vs. pools. These are joined to each grid square and treated as first-class features alongside terrain.

## Project Structure

```
00_config.R              # Global settings, libraries, utility functions
01_data_loading.R        # Load train/test, separate 2015/2019 rainfall columns
02_spatial_features.R    # Terrain derivation from elevation raster
03_external_data.R       # Soil data cleaning, zero-inflation auxiliary targets
04_train_test_split.R    # Train / validation / test split
05_model_lightgbm.R      # LightGBM with early stopping
06_model_xgboost.R       # XGBoost with CV-tuned nrounds
07_model_catboost.R      # CatBoost with native categorical support
08_model_h2o.R           # H2O AutoML
09_generate_submission.R # Inverse-RMSE weighted ensemble, submission file
MAIN.R                   # Pipeline orchestration
```

## Running the Pipeline

```r
source("MAIN.R")
```

Or run individual steps:
```r
source("00_config.R")
source("01_data_loading.R")
# ... etc.
```

## Data Files Required

| File | Description |
|---|---|
| `Train.csv` | Grid squares with 2015 and 2019 rainfall features + 2015 flood target |
| `SampleSubmission.csv` | Submission format template |

External soil data (FAO/ISRIC) must be separately downloaded and joined — see `03_external_data.R`.

## Configuration

Key settings in `00_config.R`:

| Variable | Default | Purpose |
|---|---|---|
| `GLOBAL_SEED` | 7 | Reproducibility |
| `CV_FOLDS` | 5 | Cross-validation folds |
| `NUM_THREADS` | 4 | Parallelism for boosting models |
| `DATA_DIR` | `"."` | Directory containing Train.csv |
# 5G Base Station Energy Consumption Prediction

Predicting energy consumption for 5G base stations from network load, energy-saving mode states, hardware configuration (antenna count, RU type, frequency/bandwidth), and time-of-day patterns. Competition dataset (2023), tabular regression.

## Key Engineering Decisions

**Target construction via frequency-weighted energy disaggregation.** The raw data records total energy at the base station level. For multi-cell stations (two cells operating at different frequencies), no per-cell energy measurement exists. To enable cell-level modeling, energy is attributed to each cell using frequency-specific proportions derived from hardware characteristics — `AVG_365`, `AVG_426`, `AVG_155`, `AVG_189`. Predicting cell energy is only possible after this attribution step. Without it, the label itself is undefined at the modeling granularity.

```r
energy <- energy %>%
  mutate(
    perct = case_when(
      nb_cell == 2 & freq_cat == '365'    ~ AVG_365,
      nb_cell == 2 & freq_cat == '426.98' ~ AVG_426,
      nb_cell == 2 & freq_cat == '155.6'  ~ AVG_155,
      nb_cell == 2 & freq_cat == '189'    ~ AVG_189,
      TRUE ~ 1
    ),
    target = Energy * perct
  )
```

**Domain-derived indicator features from hardware rules.** Features like `es1_load_low`, `es3_load_high`, `tree_energy_pack_F` are explicit interaction rules between energy-saving mode states, load levels, antenna counts, and frequencies. These encode known hardware behavior (e.g. ES1 activates under very low load with large antenna arrays) as binary flags rather than relying on the model to rediscover the interaction.

**Per-variable encoding strategy.** Each categorical variable receives an encoding matched to its structure: Helmert contrasts for `day_of_week` (ordered, sequential), polynomial contrasts for `RUType` (ordered by hardware generation), backward difference for `type_anten`, WoE for binary indicator flags, M-estimator smoothed encoding for `load_hour`. A separate dataset (`05_build_lightgbm_dataset.R`) is built with these pre-encoded representations specifically for LightGBM.

**Two-level modeling: base station and cell.** Models are trained independently at both granularities. Base-level models use aggregated cell features; cell-level models use individual cell observations. Final submission blends predictions from both levels.

**Three-way train / freeze / test split.** A dedicated freeze set is held out for ensemble weight optimization and model comparison, separate from both training and final test.

## Project Structure

```
00_config.R                  # Global settings and utility functions
01_data_loading.R            # Load raw data, apply hardware fixes, aggregate base info
02_feature_engineering.R     # Encoding functions and domain feature constructors
03_build_base_dataset.R      # Base-level dataset with aggregated cell features
04_build_cell_dataset.R      # Cell-level dataset with frequency-weighted energy split
05_build_lightgbm_dataset.R  # Pre-encoded dataset variant for LightGBM
06_train_test_split.R        # Train / freeze / test splits (base and cell)
07_model_lightgbm.R          # LightGBM with grid search
08_model_xgboost.R           # XGBoost with Bayesian tuning via tidymodels
09_model_catboost.R          # CatBoost with K-fold CV averaging
10_model_caret.R             # RF, GLMNet, GAM via caret
11_model_h2o.R               # H2O AutoML
12_model_stacking.R          # Tidymodels stacking ensemble
13_model_sl3.R               # SuperLearner (sl3) with NNLS metalearner
14_clustering_analysis.R     # Cluster features from load/ES mode patterns
15_model_cell_level.R        # LightGBM and XGBoost at cell granularity
16_generate_submission.R     # Ensemble and format final predictions
17_eda_and_visualization.R   # EDA reports
18_evaluate_freeze.R         # Model comparison on freeze set
MAIN.R                       # Pipeline orchestration
```

## Running the Pipeline

```r
source("MAIN.R")
run_pipeline()          # All steps

run_data_prep()         # Steps 1-6 only
run_training()          # Steps 7-15 only
run_evaluation()        # Steps 16-18 only

run_pipeline(steps = c(1, 3, 7))  # Specific steps
```

## Data Files Required

| File | Contents |
|---|---|
| `imgs_202307101549519358.csv` | Submission template |
| `imgs_2023071012133740345.csv` | Base station energy measurements |
| `imgs_2023071012130978799.csv` | Cell-level load and ES mode states |
| `imgs_2023071012123392536.csv` | Base station hardware configuration |

## Configuration

Key settings in `00_config.R`:

| Variable | Purpose |
|---|---|
| `GLOBAL_SEED` | Reproducibility seed |
| `FILE_SUBMISSION` | Path to submission template |
| `FILE_ENERGY` | Path to energy data |
| `FILE_CELL` | Path to cell data |
| `FILE_BASE` | Path to base station data |

Edit `00_config.R` to modify:
- `GLOBAL_SEED`: Random seed for reproducibility (default: 1618)
- `CV_FOLDS`: Number of cross-validation folds (default: 5)
- `TRAIN_SAMPLE_SIZE`: Training set size (default: 75000)
- `FREEZE_SAMPLE_SIZE`: Holdout validation size (default: 10000)
- `AVG_*`: Energy split proportions for multi-cell bases

## Output Files

### Intermediate Data Files
- `data_base_cleaned.RData`: Cleaned base station data
- `df_base.RData`: Base-level feature dataset
- `df_cell.RData`: Cell-level feature dataset
- `df_ligh.RData`: LightGBM-optimized dataset
- `splits_base.RData`: Train/test/freeze splits (base level)
- `splits_cell.RData`: Train/test/freeze splits (cell level)

### Model Files
- `model_lgb_*.txt`: LightGBM models
- `model_xgb_*.model`: XGBoost models
- `model_catboost_*.cbm`: CatBoost models
- `model_*_caret.rds`: Caret models
- `model_stacks.rds`: Stacked ensemble
- `model_sl3.rds`: Super Learner model

### Prediction Files
- `freez_*_*.csv`: Freeze set predictions for each model
- `submission_*.csv`: Final submission files

### Analysis Files
- `model_comparison_results.csv`: Model performance comparison
- `eda_results.rds`: EDA analysis results
- `cluster_summary.csv`: Clustering analysis results

## Dependencies

### Core Packages
- data.table, dplyr, tidyr, purrr

### Machine Learning
- caret, tidymodels, recipes, stacks
- xgboost, lightgbm, catboost
- ranger, glmnet, mgcv
- h2o, sl3
- treesnip, bonsai

### Parallel Processing
- parallel, doParallel, future

### Visualization & Analysis
- ggplot2, rpart, rpart.plot
- effectsize, ltm

## Original File Mapping

| Original File       | Reorganized To                           |
|---------------------|------------------------------------------|
| First.R             | 01, 02, 03, 17                           |
| Second.R            | 04                                       |
| Build_DF.R          | 03, 04, 05                               |
| Encoding.R          | 02, 05                                   |
| Freez.R             | 06, 18                                   |
| Caret.R             | 10                                       |
| Catboost.R          | 09                                       |
| draft.R, draf_2.R   | 07, 08                                   |
| ligh_random*.R      | 07                                       |
| ligh_recipes.R      | 07, 15                                   |
| h2o.R               | 11                                       |
| Model_Base.R        | 07, 08, 10, 11, 12, 13                   |
| Model_Cell.R        | 15                                       |
| Stacking.R          | 12, 18                                   |
| Clustering.R        | 14                                       |
| Submit.R            | 16                                       |
| Graphs.R            | 17                                       |

## Best Practices Applied

1. **Modular Design**: Each script has a single responsibility
2. **Configuration Separation**: All constants in 00_config.R
3. **Consistent Naming**: Files numbered in execution order
4. **Documentation**: Detailed comments and headers
5. **Error Handling**: Graceful handling of missing files
6. **Reproducibility**: Seeds set for all random operations
7. **Reusable Functions**: Helper functions in 02_feature_engineering.R
8. **Pipeline Orchestration**: MAIN.R for easy execution

## Notes

- The freeze set is used for local validation before submission
- Cell-level predictions need aggregation to base level for submission
- Some models (H2O, SL3) may require additional installation steps
- Run times vary significantly by model (LightGBM ~5min, H2O AutoML ~30min)
