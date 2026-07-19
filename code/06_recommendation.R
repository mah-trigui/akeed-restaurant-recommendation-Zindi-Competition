# =============================================================================
# 06_recommendation.R - Segment × Zone Recommendation Engine
# Akeed Restaurant Recommendation Challenge
# =============================================================================
# Strategy: For each (customer_segment, vendor_zone) pair in the test set:
#   1. Find train customers in the same segment and zone who ordered
#   2. Count which vendors they ordered from (frequency)
#   3. Assign those vendors to test customers, weighted by proximity
#
# Why this instead of collaborative filtering:
#   - Only 100 vendors → item space is small
#   - Strong geographic constraint (delivery radius)
#   - Sparse interaction matrix (~35k users, most ordered from 1-2 vendors)
#   - Geography + segment matching captures the dominant signal
# =============================================================================

source("00_config.R")
print_section("Step 6: Recommendation")

set.seed(GLOBAL_SEED)

tr <- readRDS(file.path(OUTPUT_DIR, "train_segmented.rds"))
ts <- readRDS(file.path(OUTPUT_DIR, "test_segmented.rds"))
vend <- readRDS(file.path(OUTPUT_DIR, "vendors_zoned.rds"))
ord_detail <- readRDS(file.path(OUTPUT_DIR, "ord_detail.rds"))

# -----------------------------------------------------------------------------
# 1. BUILD TEST SUBMISSION FRAME (CROSS JOIN)
# -----------------------------------------------------------------------------

print_step("Building customer × vendor cross join for test...")

test_loc_unique <- unique(ts[, .(customer_id, location_number)])
vendor_ids <- unique(vend$id)

test_frame <- as.data.table(
    sqldf("SELECT a.customer_id, a.location_number, b.id AS vendor_id
           FROM test_loc_unique a CROSS JOIN vendor_ids b")
)
# If sqldf doesn't work with vector, use CJ approach:
if (nrow(test_frame) == 0) {
    test_frame <- CJ(
        customer_id = test_loc_unique$customer_id,
        location_number = test_loc_unique$location_number,
        vendor_id = vendor_ids
    )
}

test_frame$ID <- paste(
    test_frame$customer_id, "X",
    test_frame$location_number, "X",
    test_frame$vendor_id
)
test_frame$target <- 0L

cat(sprintf("  Test frame: %d rows\n", nrow(test_frame)))

# -----------------------------------------------------------------------------
# 2. SEGMENT × ZONE RECOMMENDATION
# -----------------------------------------------------------------------------

print_step("Running segment × zone recommendation...")

# Join order info to train profiles
tr_with_orders <- merge(
    tr[, .(customer_id, location_number, zone, segment, lat, long)],
    ord_detail[, .(customer_id, location_number, vendor_id)],
    by = c("customer_id", "location_number"),
    all.x = TRUE
)
tr_ordered <- tr_with_orders[!is.na(vendor_id)]

# Vendor coordinates lookup
vend_coords <- vend[, .(id, longitude, latitude)]
setnames(vend_coords, c("vendor_id", "vend_long", "vend_lat"))

recommendations <- data.table()
segments <- unique(ts$segment)
zones <- unique(ts$zone)

for (seg in segments) {
    for (z in zones) {
        # Test customers in this segment × zone
        ts_subset <- ts[
            segment == seg & zone == z,
            .(customer_id, location_number, lat, long)
        ]
        if (nrow(ts_subset) == 0) next

        # Train orders from same segment × zone
        tr_subset <- tr_ordered[segment == seg & zone == z]
        if (nrow(tr_subset) == 0) next

        # Vendor frequency in this segment × zone
        vend_freq <- tr_subset[, .(freq = .N), by = vendor_id]
        vend_freq <- merge(vend_freq, vend_coords, by = "vendor_id", all.x = TRUE)

        # For each test customer: score vendors by frequency × inverse distance
        for (i in seq_len(nrow(ts_subset))) {
            cust_lat <- ts_subset$lat[i]
            cust_long <- ts_subset$long[i]

            dist <- abs(vend_freq$vend_long - cust_long) +
                abs(vend_freq$vend_lat - cust_lat)
            dist[dist == 0] <- 0.001 # avoid division by zero

            # Recommend vendors with high frequency and low distance
            vend_freq$score <- vend_freq$freq / dist

            # Take top vendors
            top_vend <- vend_freq[order(-score), vendor_id][1:min(nrow(vend_freq), 10)]

            recs <- data.table(
                customer_id = ts_subset$customer_id[i],
                location_number = ts_subset$location_number[i],
                vendor_id = top_vend
            )
            recommendations <- rbind(recommendations, recs)
        }
    }
}

cat(sprintf("  Recommendations generated: %d\n", nrow(recommendations)))

# -----------------------------------------------------------------------------
# 3. MARK RECOMMENDATIONS IN SUBMISSION FRAME
# -----------------------------------------------------------------------------

print_step("Marking recommendations in submission...")

recommendations$recommended <- 1L
test_frame <- merge(test_frame, recommendations,
    by = c("customer_id", "location_number", "vendor_id"),
    all.x = TRUE
)
test_frame$target <- ifelse(!is.na(test_frame$recommended), 1L, 0L)
test_frame$recommended <- NULL

cat(sprintf(
    "  Positive predictions: %d / %d (%.2f%%)\n",
    sum(test_frame$target), nrow(test_frame),
    100 * sum(test_frame$target) / nrow(test_frame)
))

# -----------------------------------------------------------------------------
# 4. WRITE SUBMISSION
# -----------------------------------------------------------------------------

submission <- test_frame[, .(ID, target)]
write.csv(submission, file.path(OUTPUT_DIR, "submission.csv"),
    row.names = FALSE, quote = FALSE
)

print_step("Recommendation complete.")
