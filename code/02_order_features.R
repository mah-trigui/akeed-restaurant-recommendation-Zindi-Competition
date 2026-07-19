# =============================================================================
# 02_order_features.R - Engineer Features from Order History
# Akeed Restaurant Recommendation Challenge
# =============================================================================
# Summarize order history per (customer × location × vendor) triplet:
# count, average basket, payment mode, ratings, delivery times, promo usage.
# =============================================================================

source("00_config.R")
print_section("Step 2: Order Feature Engineering")

ords <- readRDS(file.path(OUTPUT_DIR, "orders.rds"))
vend <- readRDS(file.path(OUTPUT_DIR, "vendors.rds"))

# -----------------------------------------------------------------------------
# 1. CLEAN ORDER TIMESTAMPS
# -----------------------------------------------------------------------------

print_step("Parsing order timestamps...")

ords$is_favorite[!(ords$is_favorite %in% c("Yes", "No"))] <- "Neut"
ords$is_favorite <- as.factor(ords$is_favorite)
ords$is_rated <- as.factor(ords$is_rated)
ords$vendor_rating[is.na(ords$vendor_rating)] <- 0
ords$preparationtime[is.na(ords$preparationtime)] <- 0

ords$order_accepted_time <- ymd_hms(ords$order_accepted_time)
ords$ready_for_pickup_time <- ymd_hms(ords$ready_for_pickup_time)
ords$delivered_time <- ymd_hms(ords$delivered_time)
ords$created_at <- ymd_hms(ords$created_at)

# Delivery time in minutes
ords$time <- as.numeric(difftime(ords$delivered_time, ords$order_accepted_time, units = "mins"))
ords$time_prep <- as.numeric(difftime(ords$ready_for_pickup_time, ords$order_accepted_time, units = "mins"))
ords$time[ords$time < 0] <- NA
ords$time_prep[ords$time_prep < 0] <- NA
ords$time[ords$time > 1440] <- 1440
ords$time_prep[ords$time_prep > 1440] <- 1440

ords$order <- 1L

# -----------------------------------------------------------------------------
# 2. AGGREGATE PER (CUSTOMER × LOCATION × VENDOR)
# -----------------------------------------------------------------------------

print_step("Aggregating order details per triplet...")

triplet_col <- "CID X LOC_NUM X VENDOR"

# Most recent favorite status
aux_fav <- ords[, .SD[which.max(created_at)], by = triplet_col][,
    c(triplet_col, "is_favorite"),
    with = FALSE
]

# Mean delivery distance (recent orders)
aux_dist <- ords[created_at >= "2019-10-27",
    .(deliverydistance = mean(deliverydistance, na.rm = TRUE)),
    by = triplet_col
]

# Mean delivery times (recent orders)
aux_times <- ords[created_at >= "2019-10-17",
    .(
        time = mean(time, na.rm = TRUE),
        time_prep = mean(time_prep, na.rm = TRUE)
    ),
    by = triplet_col
]

# Core order statistics
ord_detail <- ords[, .(
    nb_ord        = .N,
    avg_nb_item   = mean(item_count, na.rm = TRUE),
    avg_total     = mean(grand_total, na.rm = TRUE),
    mode_payment  = median(payment_mode),
    vendor_rating = mean(vendor_rating),
    driver_rating = mean(driver_rating)
), by = triplet_col]
ord_detail$avg_nb_item[is.nan(ord_detail$avg_nb_item)] <- 1

# Promo usage count
aux_promo <- ords[nchar(promo_code) > 0, .(nb_promo = .N), by = triplet_col]

# Join all
ord_detail <- ord_detail[aux_fav, on = triplet_col]
ord_detail <- ord_detail[aux_dist, on = triplet_col]
ord_detail <- ord_detail[aux_times, on = triplet_col]
ord_detail <- ord_detail[aux_promo, on = triplet_col]
ord_detail[is.na(nb_promo), nb_promo := 0]
ord_detail[is.na(deliverydistance), deliverydistance := -9999]
ord_detail[is.na(time), time := -9999]
ord_detail[is.na(time_prep), time_prep := -9999]

# Add vendor_id and location_number from orders
ord_uniq <- unique(ords[, .(`CID X LOC_NUM X VENDOR`, vendor_id, location_number, customer_id)])
ord_detail <- merge(ord_detail, ord_uniq, by = triplet_col, all.x = TRUE)

cat(sprintf("  Order detail rows: %d\n", nrow(ord_detail)))

# -----------------------------------------------------------------------------
# 3. SAVE
# -----------------------------------------------------------------------------

saveRDS(ord_detail, file.path(OUTPUT_DIR, "ord_detail.rds"))

print_step("Order features complete.")
