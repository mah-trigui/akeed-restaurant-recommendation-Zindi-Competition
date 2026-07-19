# =============================================================================
# 04_customer_features.R - Customer Profiles with Location
# Akeed Restaurant Recommendation Challenge
# =============================================================================
# Build customer-location profiles: demographics, account age, location count.
# Unify train and test customers for consistent feature processing.
# =============================================================================

source("00_config.R")
print_section("Step 4: Customer Features")

train_cust <- readRDS(file.path(OUTPUT_DIR, "train_cust.rds"))
test_cust <- readRDS(file.path(OUTPUT_DIR, "test_cust.rds"))
train_loc <- readRDS(file.path(OUTPUT_DIR, "train_loc.rds"))
test_loc <- readRDS(file.path(OUTPUT_DIR, "test_loc.rds"))

# -----------------------------------------------------------------------------
# 1. BUILD CUSTOMER-LOCATION PROFILES
# -----------------------------------------------------------------------------

build_customer_profile <- function(cust, loc) {
    # Join customer info with locations
    prof <- unique(merge(loc, cust,
        by.x = "customer_id", by.y = "akeed_customer_id",
        all.x = TRUE
    ))

    # Clean location type
    prof$location_type[is.na(prof$location_type) | prof$location_type == "Other"] <- "emp"
    prof$location_type <- as.factor(prof$location_type)

    # Clean coordinates
    prof$latitude[is.na(prof$latitude)] <- 0
    prof$longitude[is.na(prof$longitude)] <- 0

    # Clean gender
    prof$gender <- as.character(prof$gender)
    prof$gender[is.na(prof$gender) | prof$gender == "?????" | prof$gender == ""] <- "emp"
    prof$gender <- as.factor(prof$gender)

    # Clean language
    prof$language[is.na(prof$language) | prof$language == ""] <- "emp"
    prof$language <- as.factor(prof$language)

    # Account age (days since creation to reference date)
    prof$created_at <- as.Date(ymd_hms(prof$created_at))
    prof$client_age <- as.numeric(REFERENCE_DATE - prof$created_at)
    prof$client_age[is.na(prof$client_age)] <- -9999

    # Number of registered locations per customer
    loc_count <- loc[, .(client_loc_nb = .N), by = customer_id]
    prof <- merge(prof, loc_count, by = "customer_id", all.x = TRUE)

    # Simplified lat/long (clip outliers)
    prof$lat <- prof$latitude
    prof$lat[abs(prof$lat) > 2 & prof$lat > 0] <- 2
    prof$lat[abs(prof$lat) > 2 & prof$lat < 0] <- -2

    prof$long <- prof$longitude
    prof$long[abs(prof$long) > 79] <- -78
    prof$long[abs(prof$long) > 1 & prof$long > 0] <- 1

    prof
}

print_step("Building train customer profiles...")
tr <- build_customer_profile(train_cust, train_loc)
tr$source <- "train"

print_step("Building test customer profiles...")
ts <- build_customer_profile(test_cust, test_loc)
ts$source <- "test"

cat(sprintf("  Train profiles: %d | Test profiles: %d\n", nrow(tr), nrow(ts)))

# -----------------------------------------------------------------------------
# 2. SAVE
# -----------------------------------------------------------------------------

saveRDS(tr, file.path(OUTPUT_DIR, "train_profiles.rds"))
saveRDS(ts, file.path(OUTPUT_DIR, "test_profiles.rds"))

print_step("Customer features complete.")
