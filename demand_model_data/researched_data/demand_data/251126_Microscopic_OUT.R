library(tidyverse)
library(lubridate)
library(data.table)
library(data.table)
library(viridis)
library(data.table)
library(sf)
library(dplyr)
library(ggplot2)

#---------------------------------------------------------------------------------------------------------------------------------------
### FILTER & LOAD DATA

#paths
#Leo
folder_path <- "/Users/leoaebli/Library/CloudStorage/OneDrive-ETHZurich/Microscopic/Data Processing/Loops_2_OD/"
#Timm
#folder_path <- "bitte da din Polybox path inetue polybox.ethz.ch/Shared/Microscopic Modelling/Input Data/"


# UTD .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .

utd_path <- file.path(folder_path, "UTD/utd19_u.csv")
detectors_path <- file.path(folder_path, "UTD/detectors.csv")
links_path <- file.path(folder_path, "UTD/links.csv")

#function filter
read_zurich <- function(path, city_col) {
  header <- fread(path, nrows = 0)
  tmp <- fread(cmd = paste("grep -i 'zurich'", shQuote(path)))
  if (nrow(tmp) > 0) setnames(tmp, names(header))
  if (city_col %in% names(tmp)) {
    tmp <- tmp[tolower(get(city_col)) == "zurich"]
  }
  return(tmp)
}

#load (IN so they appear at top)
#IN_utd_zurich <- read_zurich(utd_path, "city")
IN_utd_zurich <- fread(file.path(folder_path, "UTD/utd19_u_zurich.csv"))
IN_detectors <- read_zurich(detectors_path, "citycode")
IN_links     <- read_zurich(links_path, "citycode")


# DAV .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .

# IN_DAV_Verkehrszählung <- fread(file.path(folder_path, "DAV/sid_dav_verkehrszaehlung_miv_od2031_2015.csv"))
rm(utd_path,detectors_path,links_path,read_zurich)

#write zürich only csv
#out_path <- file.path(folder_path, "utd19_u_zurich.csv")
#fwrite(IN_utd_zurich, out_path)
#rm(out_path)

#---------------------------------------------------------------------------------------------------------------------------------------
### 1. Filter Project Perimeter


# UTD .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .

## Export as geopackage
# df_sf_wgs84 <- st_as_sf(IN_detectors, coords = c("long", "lat"), crs = 4326)
# df_sf_lv95 <- st_transform(df_sf_wgs84, 2056)
# df_sf_lv95$RELEVANZ <- 0L
# gpkg_path <- file.path(folder_path, "UTD/utd19_u_zh.gpkg")
# st_write(df_sf_lv95, gpkg_path, layer = "zurich_points_utd", delete_layer = TRUE)
# rm(df_sf_wgs84,df_sf_lv95,gpkg_path)

## Import selected from geopackage
S01_gpkg_selected_path <- file.path(folder_path, "UTD/utd19_u_zh_selected.gpkg")

S01_relevanz_tbl <- st_read(
  S01_gpkg_selected_path,
  query = "SELECT detid, RELEVANZ, INTO_NETWORK FROM zurich_points_utd"
)

# Convert to data.table
S01_relevanz_df <- as.data.table(st_drop_geometry(S01_relevanz_tbl))

# Merge and keep only RELEVANZ 1 or 2
S01_UTD_selected_points <- merge(
  IN_detectors,
  S01_relevanz_df,
  by = "detid",
  all.x = TRUE
)[RELEVANZ %in% c(1, 2)]

rm(S01_gpkg_selected_path,relevanz_tbl,S01_relevanz_tbl,S01_relevanz_df)


# # DAV .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
# 
# ## Export as geopackage
# S01_DAV_Points <- IN_DAV_Verkehrszählung %>%
#   filter(!is.na(EKoord) & !is.na(NKoord)) %>%
#   distinct(MSID, .keep_all = TRUE) %>%
#   select(MSID, EKoord, NKoord) %>%
#   st_as_sf(coords = c("EKoord", "NKoord"),
#            crs = 2056,
#            remove = TRUE) %>%
#   mutate(RELEVANZ = 0L)
# 
# gpkg_path <- file.path(folder_path, "DAV/DAV_u_zh.gpkg")
# st_write(S01_DAV_Points, gpkg_path, layer = "zurich_points_dav", delete_layer = TRUE)

## Import selected from geopackage
# S01_gpkg_selected_path <- file.path(folder_path, "DAV/DAV_u_zh_selected.gpkg")
# 
# S01_relevanz_tbl <- st_read(
#   S01_gpkg_selected_path,
#   query = "SELECT MSID, RELEVANZ FROM zurich_points_dav"
# )
# 
# S01_relevanz_df <- st_drop_geometry(S01_relevanz_tbl)
# setDT(S01_relevanz_df)
# S01_DAV_selected_points <- merge(IN_DAV_Verkehrszählung, S01_relevanz_df, by = "MSID", all.x = TRUE)
# 
# S01_DAV_selected_points <- S01_relevanz_df
# 
# S01_DAV_selected_points <- S01_DAV_selected_points %>%
#   filter(RELEVANZ == 1)
# 
# rm(S01_gpkg_selected_path,S01_relevanz_tbl,S01_relevanz_df,S01_DAV_Points,gpkg_path)

#---------------------------------------------------------------------------------------------------------------------------------------
### 2. Filter Loop Data

S02_UTD_selected_loops <- IN_utd_zurich %>%
  filter(detid %in% S01_UTD_selected_points$detid)

# S02_DAV_selected_loops <- IN_DAV_Verkehrszählung %>%
#   filter(MSID %in% S01_DAV_selected_points$MSID)

#---------------------------------------------------------------------------------------------------------------------------------------
### 3. Prepare for OD conversion

S03_selected_date <- "2015-10-27"

#---------------------------------------------------------------------------------------------------------------------------------------
### Shared IPF / constraint setup (used for both Morning and Evening)

# Priority detids
priority_detids <- c(
  "K607D12","K607D16","K540D11","K540D14","K642D11","K642D15",
  "K638D11","K638D12","K638D13","K638D14",
  "K621D15"  # replaces K607X01 and K607X02 per your correction
)
priority_detids <- sort(unique(toupper(priority_detids)))

# Forbidden pairs
forbidden_pairs <- rbindlist(list(
  data.table(from = c("K607D16","K607D12"), to = "K621D15"),
  data.table(from = "K540D14", to = "K549D11"),
  CJ(from = c("K340D17","K340D16","K340D18","K340D19"), to = c("K340D14","K340D15")),
  data.table(from = "K365D14", to = "K341D12"),
  CJ(from = c("K638D13","K638D14"), to = c("K638D11","K638D12")),
  data.table(from = "K642D15", to = "K642D11"),
  CJ(from = "K601D17", to = c("K540D11","K610D12","K610D13","K621D15","K621D16")),
  CJ(from = c("K611D15","K611D16"), to = c("K610D12","K610D13","K621D15","K621D16")),
  data.table(from = "K540D14", to = c("K610D12","K610D13","K621D15","K621D16"))
))
forbidden_pairs[, `:=`(from = toupper(from), to = toupper(to))]
forbidden_pairs <- unique(forbidden_pairs)

# Algorithm parameters
S05_max_iter     <- 100
S05_tol_abs      <- 0.5       # absolute tolerance in vehicles
S05_weight_high  <- 2         # weight for priority detids
S05_weight_other <- 1         # default weight

# function to generate allowed mask for a given interval block
make_allowed_mask <- function(orig_vec, dest_vec) {
  # orig_vec, dest_vec: character vectors of detid names (uppercase or with _IN/_OUT)
  n_o <- length(orig_vec); n_d <- length(dest_vec)
  mask <- matrix(TRUE, nrow = n_o, ncol = n_d,
                 dimnames = list(orig_vec, dest_vec))
  # apply forbidden pairs (match on detid without _IN/_OUT suffix)
  orig_detids <- sub("_IN$", "", orig_vec)
  dest_detids <- sub("_OUT$", "", dest_vec)
  
  if (nrow(forbidden_pairs) > 0) {
    for (r in seq_len(nrow(forbidden_pairs))) {
      fr <- forbidden_pairs$from[r]
      to <- forbidden_pairs$to[r]
      o_idx <- which(orig_detids == fr)
      d_idx <- which(dest_detids == to)
      if (length(o_idx) > 0 && length(d_idx) > 0) {
        mask[o_idx, d_idx] <- FALSE
      }
    }
  }
  # disallow self->self for same detid
  common <- intersect(orig_detids, dest_detids)
  if (length(common) > 0) {
    for (c in common) {
      o_idx <- which(orig_detids == c)
      d_idx <- which(dest_detids == c)
      if (length(o_idx) > 0 && length(d_idx) > 0) {
        mask[o_idx, d_idx] <- FALSE
      }
    }
  }
  return(mask)
}

message("Priority detids (alphabetical): ", paste(priority_detids, collapse = ", "))
message("Forbidden pair count: ", nrow(forbidden_pairs))

# IPF helper: scale ORIGINS to DESTINATIONS
run_ipf_block <- function(orig_flows, dest_flows, allowed_mask,
                          weight_priority = S05_weight_high,
                          weight_other = S05_weight_other,
                          max_iter = S05_max_iter,
                          tol_abs = S05_tol_abs) {
  
  o_names <- names(orig_flows); d_names <- names(dest_flows)
  n_o <- length(o_names); n_d <- length(d_names)
  
  if (n_o == 0 || n_d == 0) {
    return(list(
      od  = matrix(0, nrow = n_o, ncol = n_d,
                   dimnames = list(o_names, d_names)),
      log = data.table(iter = 0, max_row_err = NA_real_, max_col_err = NA_real_)
    ))
  }
  
  # scale origins to destinations
  total_o <- sum(orig_flows)
  total_d <- sum(dest_flows)
  
  dest_target <- dest_flows
  if (total_o == 0) {
    orig_target <- orig_flows * 0
  } else {
    orig_target <- orig_flows * (total_d / total_o)
  }
  
  # initial Q using orig_target and weights
  weights <- matrix(0, nrow = n_o, ncol = n_d,
                    dimnames = list(o_names, d_names))
  for (j in seq_len(n_d)) {
    dname <- d_names[j]
    w <- ifelse(sub("_OUT$", "", dname) %in% priority_detids, weight_priority, weight_other)
    weights[, j] <- w
  }
  weights[!allowed_mask] <- 0
  
  row_weight_sums <- rowSums(weights, na.rm = TRUE)
  Q <- matrix(0, nrow = n_o, ncol = n_d,
              dimnames = list(o_names, d_names))
  for (i in seq_len(n_o)) {
    if (row_weight_sums[i] > 0) {
      Q[i, ] <- (weights[i, ] / row_weight_sums[i]) * orig_target[i]
    }
  }
  
  iter_log <- data.table(iter = 0, max_row_err = NA_real_, max_col_err = NA_real_)
  for (it in seq_len(max_iter)) {
    # column scaling → match dest_target (unchanged observations)
    col_sums <- colSums(Q, na.rm = TRUE)
    for (j in seq_len(n_d)) {
      if (col_sums[j] > 0 && dest_target[j] >= 0) {
        scale_j <- dest_target[j] / col_sums[j]
        Q[allowed_mask[, j], j] <- Q[allowed_mask[, j], j] * scale_j
      } else {
        Q[, j] <- 0
      }
    }
    
    # row scaling → match orig_target (scaled inflow)
    row_sums <- rowSums(Q, na.rm = TRUE)
    for (i in seq_len(n_o)) {
      if (row_sums[i] > 0 && orig_target[i] >= 0) {
        scale_i <- orig_target[i] / row_sums[i]
        Q[i, allowed_mask[i, ]] <- Q[i, allowed_mask[i, ]] * scale_i
      } else {
        Q[i, ] <- 0
      }
    }
    
    current_row_sums <- rowSums(Q, na.rm = TRUE)
    current_col_sums <- colSums(Q, na.rm = TRUE)
    max_row_err <- max(abs(current_row_sums - orig_target), na.rm = TRUE)
    max_col_err <- max(abs(current_col_sums - dest_target), na.rm = TRUE)
    
    iter_log <- rbind(
      iter_log,
      data.table(iter = it,
                 max_row_err = max_row_err,
                 max_col_err = max_col_err)
    )
    
    if (max_row_err <= tol_abs && max_col_err <= tol_abs) break
  }
  
  list(od = Q, log = iter_log)
}

#---------------------------------------------------------------------------------------------------------------------------------------
### Wrapper to run the whole pipeline for a given time window

run_od_for_period <- function(label, start_sec, end_sec, suffix) {
  message("---- Running OD pipeline for ", label, 
          " (", start_sec, "–", end_sec, " seconds) ----")
  
  # 3. Prepare for OD conversion (time filter)
  S03_UTD_selected_loops <- S02_UTD_selected_loops %>%
    filter(day == S03_selected_date) %>%
    filter(interval >= start_sec & interval < end_sec)
  
  setDT(S03_UTD_selected_loops)
  
  # 1. compute 30-min intervals
  S03_UTD_selected_loops[, flow := flow * (3 / 60)]
  S03_UTD_selected_loops[, interval30 := floor(interval / 1800)]
  
  # 2. aggregate 3-min intervals to 30-min per detid
  S03_UTD_30min <- S03_UTD_selected_loops[, .(
    flow_30min = sum(flow),
    occ_30min  = mean(occ),
    day        = first(day),
    intervalT  = paste0("T", interval30 - 11)
  ), by = .(detid, interval30)]
  
  # 3. combine sets of detids
  S03_UTD_30min[
    detid %in% c("K642D17", "K364D11"),
    detid := fifelse(detid == "K642D17", 
                     "K642D15",      # replace 17 → 15
                     "K365D14")      # replace 11 → 14
  ]
  
  # 4. sum/average again for the combined detids
  S03_UTD_30min <- S03_UTD_30min[, .(
    flow_30min = sum(flow_30min),
    occ_30min  = mean(occ_30min),
    day        = first(day),
    intervalT  = first(intervalT)
  ), by = .(detid, interval30)]
  
  # 5. filter only IN/OUTPUT detectors
  setDT(S03_UTD_30min)
  setDT(S01_UTD_selected_points)
  
  S03_UTD_30min <- S01_UTD_selected_points[S03_UTD_30min, 
                                           on = "detid",
                                           .(detid, interval30, intervalT, flow_30min, occ_30min,
                                             RELEVANZ, INTO_NETWORK)]
  
  S03_UTD_30min_border <- S03_UTD_30min %>%
    filter(RELEVANZ == 2)
  
  message(label, ": number of border detids = ", length(unique(S03_UTD_30min_border$detid)))
  
  #-------------------------------------------------------------------
  ### 4. Create OD matrix
  
  S04_detids    <- sort(unique(S03_UTD_30min_border$detid))
  S04_intervals <- sort(unique(S03_UTD_30min_border$intervalT))
  
  # Build Cartesian product
  S04_index_dt <- CJ(detid = S04_detids, intervalT = S04_intervals, unique = TRUE)
  
  # suffix for IN/OUT
  suffix_dt <- unique(S03_UTD_30min_border[, .(detid, INTO_NETWORK)])
  suffix_dt[, suffix := ifelse(INTO_NETWORK == 1, "IN", "OUT")]
  suffix_lookup <- setNames(suffix_dt$suffix, suffix_dt$detid)
  
  # order
  setorder(S04_index_dt, intervalT, detid)
  
  # label: T1_K540D14_IN
  S04_index_dt[, detid_interval := paste0(intervalT, "_", detid, "_", suffix_lookup[detid])]
  
  S04_n <- nrow(S04_index_dt)
  message(label, ": S04 index rows: ", S04_n)
  
  S04_labels <- S04_index_dt$detid_interval
  
  S04_mat <- matrix(
    NA_real_, 
    nrow = S04_n, 
    ncol = S04_n,
    dimnames = list(S04_labels, S04_labels)
  )
  
  intervals <- unique(S04_index_dt$intervalT)
  for (it in intervals) {
    idx <- which(S04_index_dt$intervalT == it)
    S04_mat[idx, idx] <- 0
  }
  
  message(label, ": Matrix dimensions: ", paste(dim(S04_mat), collapse = " x "))
  message(label, ": Number of NA cells: ", sum(is.na(S04_mat)))
  message(label, ": Number of zero cells (same-interval blocks): ", sum(S04_mat == 0, na.rm = TRUE))
  
  #-------------------------------------------------------------------
  ### 5. Populate OD matrix (IPF)
  
  # Reuse index and labels for S05
  S05_index_dt <- copy(S04_index_dt)
  S05_n        <- nrow(S05_index_dt)
  
  # Reuse the matrix scaffold
  S05_mat <- matrix(as.numeric(S04_mat), nrow = nrow(S04_mat), ncol = ncol(S04_mat),
                    dimnames = list(rownames(S04_mat), colnames(S04_mat)))
  message(label, ": Reused S04 index/matrix for S05. Rows: ", S05_n,
          "; matrix dims: ", paste(dim(S05_mat), collapse = " x "))
  
  # Prepare DT_edges (RELEVANZ == 2)
  DT_edges <- copy(S03_UTD_30min)[RELEVANZ == 2]
  DT_edges[, detid := toupper(detid)]
  DT_edges[, INTO_NETWORK := as.integer(INTO_NETWORK)]
  
  S05_intervals <- unique(S05_index_dt$intervalT)
  S05_detids    <- sort(unique(S05_index_dt$detid))
  
  # Build S05_interval_meta
  S05_interval_meta <- vector("list", length(S05_intervals))
  names(S05_interval_meta) <- S05_intervals
  for (it in S05_intervals) {
    DT_it   <- DT_edges[intervalT == it]
    origins <- DT_it[INTO_NETWORK == 1, .(detid, flow_30min)]
    dests   <- DT_it[INTO_NETWORK == 0, .(detid, flow_30min)]
    S05_interval_meta[[it]] <- list(origins = origins, dests = dests)
  }
  message(label, ": Prepared S05_interval_meta for ", length(S05_intervals), 
          " intervals and ", length(S05_detids), " detids.")
  
  # IPF per interval
  S05_iter_logs           <- list()
  S05_margin_summary_list <- list()
  
  for (it in S05_intervals) {
    meta       <- S05_interval_meta[[it]]
    origins_dt <- meta$origins
    dests_dt   <- meta$dests
    
    orig_names <- paste0(origins_dt$detid, "_IN")
    dest_names <- paste0(dests_dt$detid, "_OUT")
    orig_flows <- setNames(origins_dt$flow_30min, orig_names)
    dest_flows <- setNames(dests_dt$flow_30min, dest_names)
    
    allowed_mask <- make_allowed_mask(orig_names, dest_names)
    
    ipf_res   <- run_ipf_block(orig_flows, dest_flows, allowed_mask)
    Q_block   <- ipf_res$od
    log_block <- ipf_res$log
    log_block[, intervalT := it]
    S05_iter_logs[[it]] <- log_block
    
    # write Q_block into S05_mat
    row_labels <- paste0(it, "_", rownames(Q_block))
    col_labels <- paste0(it, "_", colnames(Q_block))
    
    missing_rows <- setdiff(row_labels, rownames(S05_mat))
    missing_cols <- setdiff(col_labels, colnames(S05_mat))
    if (length(missing_rows) > 0 || length(missing_cols) > 0) {
      warning("Some labels missing for interval ", it, " (", label, ")",
              " rows missing: ", paste(missing_rows, collapse = ","),
              " cols missing: ", paste(missing_cols, collapse = ","))
    }
    
    rows_present <- intersect(row_labels, rownames(S05_mat))
    cols_present <- intersect(col_labels, colnames(S05_mat))
    if (length(rows_present) && length(cols_present)) {
      rnames <- sub(paste0(it, "_"), "", rows_present)
      cnames <- sub(paste0(it, "_"), "", cols_present)
      S05_mat[rows_present, cols_present] <- Q_block[rnames, cnames, drop = FALSE]
    }
    
    modeled_row_sums <- rowSums(Q_block, na.rm = TRUE)
    modeled_col_sums <- colSums(Q_block, na.rm = TRUE)
    sum_orig         <- sum(orig_flows)
    sum_dest         <- sum(dest_flows)
    sum_modeled      <- sum(modeled_row_sums)
    
    S05_margin_summary_list[[it]] <- data.table(
      intervalT        = it,
      total_orig_obs   = sum_orig,
      total_dest_obs   = sum_dest,
      total_orig_scaled = sum_dest,   # origins scaled to match destinations
      total_modeled    = sum_modeled,
      n_orig           = length(orig_flows),
      n_dest           = length(dest_flows)
    )
  }
  
  S05_OD_iter_log   <- rbindlist(S05_iter_logs, use.names = TRUE, fill = TRUE)
  S05_margin_summary <- rbindlist(S05_margin_summary_list, use.names = TRUE, fill = TRUE)
  
  message(label, ": Completed S05 iterative allocation for ", length(S05_intervals), " intervals.")
  
  #-------------------------------------------------------------------
  ### Final objects & write to CSV
  
  S05_OD_final <- round(S05_mat)
  
  # Expose labeled objects in global environment
  assign(paste0("S05_OD_final_", label),        S05_OD_final,        envir = .GlobalEnv)
  assign(paste0("S05_OD_iter_log_", label),     S05_OD_iter_log,     envir = .GlobalEnv)
  assign(paste0("S05_margin_summary_", label),  S05_margin_summary,  envir = .GlobalEnv)
  
  # Prepare data.frame for writing (rows as index + matrix)
  S05_df_for_write <- as.data.frame(S05_OD_final, stringsAsFactors = FALSE)
  S05_df_for_write <- cbind(index = rownames(S05_df_for_write), S05_df_for_write)
  
  S05_out_path <- file.path(folder_path, paste0("S05_OD_", suffix, "_final.csv"))
  data.table::fwrite(S05_df_for_write, file = S05_out_path, na = "NA")
  message("S05_OD_", suffix, "_final saved to: ", S05_out_path)
  
  log_out_path <- file.path(folder_path, paste0("S05_OD_", suffix, "_iter_log.csv"))
  data.table::fwrite(S05_OD_iter_log, file = log_out_path)
  message("S05 iteration log (", suffix, ") saved to: ", log_out_path)
  
  margin_out_path <- file.path(folder_path, paste0("S05_", suffix, "_margin_summary.csv"))
  data.table::fwrite(S05_margin_summary, file = margin_out_path)
  message("S05 margin summary (", suffix, ") saved to: ", margin_out_path)
  
  invisible(list(
    OD_final       = S05_OD_final,
    OD_iter_log    = S05_OD_iter_log,
    margin_summary = S05_margin_summary
  ))
}

#---------------------------------------------------------------------------------------------------------------------------------------
### Run for Morning and Evening (OUTPUT FILENAMES UNCHANGED)

# Morning: 06:00–10:00 → 21 600–36 000
run_od_for_period(
  label  = "Morning",
  start_sec = 21600,
  end_sec   = 36000,
  suffix    = "morning"
)

# Evening: 15:00–19:00 → 54 000–68 400
run_od_for_period(
  label  = "Evening",
  start_sec = 54000,
  end_sec   = 68400,
  suffix    = "evening"
)
