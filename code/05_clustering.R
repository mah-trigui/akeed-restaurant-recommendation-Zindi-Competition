# =============================================================================
# 05_clustering.R - Vendor Zones and Customer Segmentation
# Akeed Restaurant Recommendation Challenge
# =============================================================================
# Key design decision:
# Instead of collaborative filtering (sparse user-item matrix on 100 vendors),
# the recommendation is built on two spatial axes:
#   1. Vendor ZONES: K-means on vendor coordinates → geographic clusters
#   2. Customer SEGMENTS: gender × location count × account tenure
# Then within each (segment, zone) pair, recommend based on historical patterns.
# =============================================================================

source("00_config.R")
print_section("Step 5: Clustering")

set.seed(GLOBAL_SEED)

vend <- readRDS(file.path(OUTPUT_DIR, "vendors_processed.rds"))
tr <- readRDS(file.path(OUTPUT_DIR, "train_profiles.rds"))
ts <- readRDS(file.path(OUTPUT_DIR, "test_profiles.rds"))

# -----------------------------------------------------------------------------
# 1. VENDOR GEOGRAPHIC ZONES (K-MEANS)
# -----------------------------------------------------------------------------

print_step("Clustering vendors into geographic zones...")

# Exclude outlier vendors (invalid coordinates)
valid_coords <- !is.na(vend$lat_clean) & !is.na(vend$long_clean)
vend_valid <- vend[valid_coords, ]

km_vend <- kmeans(
    x = vend_valid[, .(long_clean, lat_clean)],
    centers = 4,
    algorithm = "MacQueen",
    iter.max = 5000,
    nstart = 25
)

vend$zone <- NA_integer_
vend$zone[valid_coords] <- km_vend$cluster

# Assign outlier vendors to nearest zone
if (any(!valid_coords)) {
    outlier_vend <- vend[!valid_coords, .(longitude, latitude)]
    for (i in seq_len(nrow(outlier_vend))) {
        dists <- apply(km_vend$centers, 1, function(ctr) {
            sum((c(outlier_vend$longitude[i], outlier_vend$latitude[i]) - ctr)^2)
        })
        vend$zone[which(!valid_coords)[i]] <- which.min(dists)
    }
}

vend$zone <- as.factor(vend$zone)
cat(sprintf("  Vendor zones: %s\n", paste(table(vend$zone), collapse = " | ")))

# -----------------------------------------------------------------------------
# 2. ASSIGN CUSTOMERS TO VENDOR ZONES
# -----------------------------------------------------------------------------

print_step("Assigning customers to vendor zones by proximity...")

assign_zone <- function(profiles, centers) {
    zones <- apply(profiles[, .(long, lat)], 1, function(row) {
        dists <- apply(centers, 1, function(ctr) sum((row - ctr)^2))
        which.min(dists)
    })
    zones
}

tr$zone <- assign_zone(tr, km_vend$centers)
ts$zone <- assign_zone(ts, km_vend$centers)

# -----------------------------------------------------------------------------
# 3. CUSTOMER SEGMENTATION
# -----------------------------------------------------------------------------

print_step("Segmenting customers (gender × locations × tenure)...")

segment_customers <- function(profiles) {
    # Age bucket: old (>= 300 days) vs new
    profiles$age_bucket <- ifelse(profiles$client_age >= 300, "O", "N")

    # Location count bucket: 1, 2-3, 4+
    profiles$loc_bucket <- ifelse(profiles$client_loc_nb == 1, 0,
        ifelse(profiles$client_loc_nb <= 3, 1, 2)
    )

    # Gender encoding: emp=0, Female=1, Male=2
    profiles$gender_enc <- ifelse(profiles$gender == "emp", 0,
        ifelse(profiles$gender == "Female", 1, 2)
    )

    # Segment ID: unique combination
    profiles$segment <- paste(profiles$gender_enc, profiles$loc_bucket,
        profiles$age_bucket,
        sep = "_"
    )
    profiles
}

tr <- segment_customers(tr)
ts <- segment_customers(ts)

cat(sprintf("  Unique segments: %d\n", length(unique(c(tr$segment, ts$segment)))))
cat(sprintf("  Train segment distribution:\n"))
print(sort(table(tr$segment), decreasing = TRUE)[1:10])

# -----------------------------------------------------------------------------
# 4. SAVE
# -----------------------------------------------------------------------------

saveRDS(tr, file.path(OUTPUT_DIR, "train_segmented.rds"))
saveRDS(ts, file.path(OUTPUT_DIR, "test_segmented.rds"))
saveRDS(vend, file.path(OUTPUT_DIR, "vendors_zoned.rds"))
saveRDS(km_vend, file.path(OUTPUT_DIR, "km_vendor_model.rds"))

print_step("Clustering complete.")
