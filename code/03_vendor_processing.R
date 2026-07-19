# =============================================================================
# 03_vendor_processing.R - Vendor Features and Tag Grouping
# Akeed Restaurant Recommendation Challenge
# =============================================================================
# Key decision: group vendor tags into broader cuisine categories.
# Original tags are too granular (50+ tags, many overlap). Grouping into
# 8 categories (Arabic, Indian, International, Desserts, Drinks, Sandwiches,
# Breakfast, Others) makes the vendor profile usable for matching.
# =============================================================================

source("00_config.R")
print_section("Step 3: Vendor Processing")

vend <- readRDS(file.path(OUTPUT_DIR, "vendors.rds"))

# -----------------------------------------------------------------------------
# 1. BASIC CLEANING
# -----------------------------------------------------------------------------

print_step("Cleaning vendor attributes...")

vend$vendor_category_id <- as.factor(vend$vendor_category_id)
vend$is_akeed_delivering <- as.factor(vend$is_akeed_delivering)

vend$created_at <- as.Date(ymd_hms(vend$created_at))
vend$vendor_age <- as.numeric(as.Date("2020-06-30") - vend$created_at)
vend$created_at <- NULL

# -----------------------------------------------------------------------------
# 2. TAG GROUPING
# -----------------------------------------------------------------------------

print_step("Grouping vendor tags into cuisine categories...")

tag <- vend[, .(id, vendor_tag_name)]
tag <- tag[, .(tag = unlist(strsplit(as.character(vendor_tag_name), ","))), by = id]
tag$tag <- trimws(tag$tag)
tag$n <- 1L

# Group into broader categories
tag$tag[tag$tag %in% c(
    "Cakes", "Donuts", "Sweets", "Waffles", "Pastry",
    "Mandazi", "Bagels", "Pancakes", "Ice creams"
)] <- "Desserts"
tag$tag[tag$tag %in% c(
    "Burgers", "Pizza", "Combos", "Hot Dogs", "Pizzas",
    "Fries", "American"
)] <- "Sandwiches"
tag$tag[tag$tag %in% c(
    "Spanish Latte", "Karak", "Frozen yoghurt", "Mojitos ",
    "Hot Chocolate", "Coffee", "Smoothies", "Cafe",
    "Mojitos", "Milkshakes", "Fresh Juices"
)] <- "Drinks"
tag$tag[tag$tag %in% c("Thali", "Biryani", "Indian")] <- "Indian"
tag$tag[tag$tag %in% c(
    "Shuwa", "Manakeesh", "Kushari", "Fatayers", "Mishkak",
    "Omani", "Kebabs", "Steaks", "Rice", "Shawarma",
    "Grills", "Arabic"
)] <- "Arabic"
tag$tag[tag$tag %in% c(
    "Sushi", "Dimsum", "Chinese", "Thai", "Japanese",
    "Mexican", "Lebanese", "Italian", "Asian"
)] <- "Inter"
tag$tag[tag$tag %in% c(
    "Vegetarian", "Seafood", "Pastas", "Organic",
    "Family Meal", "Rolls", "Healthy Food", "Soups",
    "Kids meal", "Pasta", "Salads"
)] <- "Others"
tag$tag[tag$tag %in% c("Churros", "Crepes", "Breakfast")] <- "Breakfast"

tag <- unique(tag)
tag_wide <- dcast(tag, id ~ tag, value.var = "n", fill = 0)

cat(sprintf("  Cuisine categories: %d\n", ncol(tag_wide) - 1))

# Join back
vend <- merge(vend, tag_wide, by = "id", all.x = TRUE)
vend$vendor_tag_name <- NULL

# -----------------------------------------------------------------------------
# 3. COORDINATE CLEANUP
# -----------------------------------------------------------------------------

print_step("Cleaning vendor coordinates...")

# Flag outlier coordinates
vend$lat_clean <- vend$latitude
vend$long_clean <- vend$longitude
vend$lat_clean[abs(vend$lat_clean) >= 1 & vend$lat_clean > 0] <- NA
vend$lat_clean[abs(vend$lat_clean) >= 1 & vend$lat_clean < 0] <- NA
vend$long_clean[abs(vend$long_clean) > 40] <- NA
vend$long_clean[vend$long_clean < 0] <- NA

cat(sprintf(
    "  Vendors with valid coordinates: %d / %d\n",
    sum(!is.na(vend$lat_clean)), nrow(vend)
))

# -----------------------------------------------------------------------------
# 4. SAVE
# -----------------------------------------------------------------------------

saveRDS(vend, file.path(OUTPUT_DIR, "vendors_processed.rds"))

print_step("Vendor processing complete.")
