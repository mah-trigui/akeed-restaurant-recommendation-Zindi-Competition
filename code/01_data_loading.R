# =============================================================================
# 01_data_loading.R - Load and Clean Raw Data
# Akeed Restaurant Recommendation Challenge
# =============================================================================

source("00_config.R")
print_section("Step 1: Data Loading")

# -----------------------------------------------------------------------------
# 1. LOAD RAW FILES
# -----------------------------------------------------------------------------

train_cust <- fread(file.path(DATA_DIR, "train_customers.csv"))
test_cust <- fread(file.path(DATA_DIR, "test_customers.csv"))
train_loc <- fread(file.path(DATA_DIR, "train_locations.csv"))
test_loc <- fread(file.path(DATA_DIR, "test_locations.csv"))
vend <- fread(file.path(DATA_DIR, "vendors.csv"))
ords <- fread(file.path(DATA_DIR, "orders.csv"))

cat(sprintf(
    "  Train customers: %d | Test customers: %d\n",
    nrow(train_cust), nrow(test_cust)
))
cat(sprintf(
    "  Train locations: %d | Test locations: %d\n",
    nrow(train_loc), nrow(test_loc)
))
cat(sprintf("  Vendors: %d | Orders: %d\n", nrow(vend), nrow(ords)))

# -----------------------------------------------------------------------------
# 2. BASIC CORRECTIONS
# -----------------------------------------------------------------------------

print_step("Correcting inconsistencies...")

# Fix gender casing
train_cust$gender[train_cust$gender == "male"] <- "Male"
test_cust$gender[test_cust$gender == "male"] <- "Male"

# Keep only verified customers (deduplicate conflicting records)
train_cust <- unique(subset(train_cust, verified == 1))
test_cust <- unique(subset(test_cust, verified == 1))

# -----------------------------------------------------------------------------
# 3. SAVE
# -----------------------------------------------------------------------------

saveRDS(train_cust, file.path(OUTPUT_DIR, "train_cust.rds"))
saveRDS(test_cust, file.path(OUTPUT_DIR, "test_cust.rds"))
saveRDS(train_loc, file.path(OUTPUT_DIR, "train_loc.rds"))
saveRDS(test_loc, file.path(OUTPUT_DIR, "test_loc.rds"))
saveRDS(vend, file.path(OUTPUT_DIR, "vendors.rds"))
saveRDS(ords, file.path(OUTPUT_DIR, "orders.rds"))

print_step("Data loading complete.")
# =============================================================================
# 01_data_loading.R - Load and Prepare Data
# Mental Health Text Classification Pipeline
# =============================================================================

source("00_config.R")
print_section("Step 1: Data Loading")

# -----------------------------------------------------------------------------
# 1. LOAD RAW DATA
# -----------------------------------------------------------------------------

train_orig <- fread(TRAIN_FILE)
test_orig <- fread(TEST_FILE)

cat(sprintf("  Train: %d rows | Test: %d rows\n", nrow(train_orig), nrow(test_orig)))

# -----------------------------------------------------------------------------
# 2. CREATE BINARY TARGET COLUMNS (ONE-VS-REST)
# -----------------------------------------------------------------------------

print_step("Creating binary target columns...")

train_orig$depress <- as.factor(ifelse(train_orig$label == "Depression", "1", "0"))
train_orig$alch <- as.factor(ifelse(train_orig$label == "Alcohol", "1", "0"))
train_orig$suic <- as.factor(ifelse(train_orig$label == "Suicide", "1", "0"))
train_orig$drug <- as.factor(ifelse(train_orig$label == "Drugs", "1", "0"))

cat("  Class distribution:\n")
print(table(train_orig$label))

# -----------------------------------------------------------------------------
# 3. SAVE
# -----------------------------------------------------------------------------

saveRDS(train_orig, file.path(OUTPUT_DIR, "train_raw.rds"))
saveRDS(test_orig, file.path(OUTPUT_DIR, "test_raw.rds"))

print_step("Data loading complete.")
# =============================================================================
# 01_data_loading.R - Load and Prepare Raw Data
# Malawi Flood Extent Prediction Pipeline
# =============================================================================

source("00_config.R")
print_section("Step 1: Data Loading")

# -----------------------------------------------------------------------------
# 1. LOAD TRAIN AND SUBMISSION FILES
# -----------------------------------------------------------------------------

print_step("Loading raw data...")

df_raw <- fread(TRAIN_FILE)
submission <- fread(SUBMISSION_FILE)

df_raw$LC_Type1_mode <- as.factor(df_raw$LC_Type1_mode)

# Reorder: target first, then ID columns, then features
df_raw <- df_raw[, c(40, 1:4, 39, 5:38)]

cat(sprintf("  Train rows: %d | Columns: %d\n", nrow(df_raw), ncol(df_raw)))

# -----------------------------------------------------------------------------
# 2. SEPARATE 2015 AND 2019 RAINFALL COLUMNS
# -----------------------------------------------------------------------------

print_step("Separating 2015 and 2019 precipitation columns...")

precip_2015 <- df_raw[, !grepl("precip 2019", names(df_raw)), with = FALSE]
precip_2015 <- precip_2015[, 7:23]

precip_2019 <- df_raw[, grepl("precip 2019", names(df_raw)), with = FALSE]

for (i in seq_len(ncol(precip_2019))) {
    names(precip_2019)[i] <- paste0("week_", i)
    names(precip_2015)[i] <- paste0("week_", i)
}

precip_2015$target <- df_raw$target_2015
precip_2019$target <- NA

# Build train and test base frames
id_cols <- df_raw[, c(1:3, 5, 6)]

train <- cbind(id_cols, precip_2015)
test <- cbind(id_cols, precip_2019)

cat(sprintf("  Train (2015 rainfall): %d rows\n", nrow(train)))
cat(sprintf("  Test  (2019 rainfall): %d rows\n", nrow(test)))

# -----------------------------------------------------------------------------
# 3. COMBINE TRAIN + TEST FOR CONSISTENT FACTOR LEVELS
# -----------------------------------------------------------------------------

print_step("Combining train + test for unified factor encoding...")

all <- rbind(train[, -"target", with = FALSE], test[, -"target", with = FALSE])
all$target <- c(train$target, test$target)

# Fix land cover: merge level 7 into 4 (too sparse)
all$LC_Type1_mode[all$LC_Type1_mode == 7] <- 4
all$LC_Type1_mode <- droplevels(as.factor(all$LC_Type1_mode))

# -----------------------------------------------------------------------------
# 4. SAVE
# -----------------------------------------------------------------------------

saveRDS(train, file.path(OUTPUT_DIR, "train_raw.rds"))
saveRDS(test, file.path(OUTPUT_DIR, "test_raw.rds"))
saveRDS(all, file.path(OUTPUT_DIR, "all_raw.rds"))
saveRDS(submission, file.path(OUTPUT_DIR, "submission.rds"))

print_step("Data loading complete.")
# ============================================================================
# 01_DATA_LOADING.R - Data Import and Initial Cleaning
# ============================================================================
# Description: Loads raw data files and performs initial data quality fixes
#              including handling duplicates and known data errors.
# ============================================================================

source("00_config.R")

# ============================================================================
# 1. LOAD RAW DATA
# ============================================================================

cat("Loading raw data files...\n")

submission <- read.csv(FILE_SUBMISSION)
energy <- read.csv(FILE_ENERGY)
cell <- read.csv(FILE_CELL)
base <- read.csv(FILE_BASE)

cat(sprintf("  Submission: %d rows\n", nrow(submission)))
cat(sprintf("  Energy:     %d rows\n", nrow(energy)))
cat(sprintf("  Cell:       %d rows\n", nrow(cell)))
cat(sprintf("  Base:       %d rows\n", nrow(base)))

# ============================================================================
# 2. DATA QUALITY FIXES - BASE STATION DATA
# ============================================================================

cat("\nApplying data quality fixes to base station data...\n")

# -- Fix Antenna Inconsistencies --
# Replace the min antenna with the max value since it's a base characteristic
base$Antennas <- ave(
    base$Antennas,
    base$BS,
    FUN = function(x) if (length(unique(x)) > 1) max(x) else x
)

# -- Create TXpower Character Column --
base$TXpower_ch <- as.character(base$TXpower)

# -- Fix Bandwidth Issues --
# No 5MHz bandwidth exists - change to 10
base$Bandwidth[base$Bandwidth == 5] <- 10

# -- Fix Specific Station Errors --
base$Antennas[base$BS == "B_925"] <- 8
base$Bandwidth[base$Frequency == 364] <- 20
base$Frequency[base$Frequency == 364] <- 365
base$Antennas[base$RUType == "Type11"] <- 8

# -- Remove Invalid Records --
# Remove rows with invalid Frequency/Bandwidth combination
base <- base[!(base$Frequency == 426.98 & base$Bandwidth == 10), ]

# -- Create Factor Categories --
base$freq_cat <- as.factor(base$Frequency)
base$band_cat <- as.factor(base$Bandwidth)
base$anten_cat <- as.factor(base$Antennas)

# -- Remove CellName Column and Duplicates --
base <- base[, names(base) != "CellName"]
base <- unique(base)

# -- Create Cell Identifier --
base <- base %>%
    group_by(BS) %>%
    mutate(cellname = paste0("Cell", row_number() - 1)) %>%
    ungroup()

cat(sprintf("  Base stations after cleaning: %d unique records\n", nrow(base)))

# ============================================================================
# 3. CREATE BASE STATION AGGREGATED INFO
# ============================================================================

cat("Creating base station aggregated features...\n")

base_gp_info <- base %>%
    group_by(BS) %>%
    summarize(
        nb_cell   = n(),
        freq_nb   = n_distinct(Frequency),
        band_nb   = n_distinct(Bandwidth),
        power_nb  = n_distinct(TXpower),
        freq_avg  = mean(Frequency),
        freq_max  = max(Frequency),
        freq_min  = min(Frequency),
        band_avg  = mean(Bandwidth),
        band_max  = max(Bandwidth),
        band_min  = min(Bandwidth),
        power_sum = sum(TXpower),
        .groups   = "drop"
    )

cat(sprintf("  Aggregated info for %d base stations\n", nrow(base_gp_info)))

# ============================================================================
# 4. VERIFY DATA INTEGRITY
# ============================================================================

cat("\nData integrity checks:\n")

# Check unique counts
cat(sprintf("  Unique BS in base:   %d\n", length(unique(base$BS))))
cat(sprintf("  Unique BS in energy: %d\n", length(unique(energy$BS))))
cat(sprintf("  Unique BS in cell:   %d\n", length(unique(cell$BS))))

# Check RUType consistency (each base should have ONE RUType)
unique_rutypes <- sapply(split(base$RUType, base$BS), function(x) length(unique(x)))
if (all(unique_rutypes == 1)) {
    cat("  RUType consistency: OK (each base has one RUType)\n")
} else {
    cat("  WARNING: Some bases have multiple RUTypes!\n")
}

# Check Mode consistency
unique_modes <- sapply(split(base$Mode, base$BS), function(x) length(unique(x)))
if (all(unique_modes == 1)) {
    cat("  Mode consistency: OK (each base has one Mode)\n")
} else {
    cat("  WARNING: Some bases have multiple Modes!\n")
}

# ============================================================================
# 5. SAVE CLEANED DATA
# ============================================================================

cat("\nSaving cleaned base data...\n")

save(base, base_gp_info, file = "data_base_cleaned.RData")
write.csv2(base, "data_base_cleaned.csv", quote = FALSE, row.names = FALSE)

cat("Data loading complete.\n")
