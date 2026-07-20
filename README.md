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
