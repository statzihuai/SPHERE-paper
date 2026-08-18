#!/usr/bin/env Rscript
# GNPC WSS pathway-module vs cognition associations, real and SPHERE arms.
#
# Collects the per-outcome GLM tables written by the WSS pipeline, applies
# BH-FDR within each outcome x arm, and writes one tidy table for render.py
# to draw panel g from.
#
# Directory layout expected under --wss-dir:
#     <outcome>/actual_output/GNPC.*.<outcome>.linreg_glm.csv[.gz]   real arm
#     <outcome>/synthetic_output.<outcome>.linreg_glm.csv[.gz]       SPHERE arm
#
# Outcomes ending in "_bin" take the binary GLM table, the others the
# continuous one.
#
# Usage:
#   Rscript step6_gnpc_wss.R \
#     --wss-dir $DATA_DIR/WSS_clin \
#     --outcomes MoCA,diagnosis_bin \
#     --out-dir output/gnpc

suppressMessages(library(data.table))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 6) {
  cat("Usage: Rscript step6_gnpc_wss.R --wss-dir DIR --outcomes o1,o2 --out-dir DIR\n")
  quit(status = 1)
}
wss_dir  <- args[match("--wss-dir", args) + 1]
outcomes <- strsplit(args[match("--outcomes", args) + 1], ",")[[1]]
out_dir  <- args[match("--out-dir", args) + 1]
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Pick one GLM table per outcome and arm. diagnosis_bin ships both a binary
# and a continuous fit; the published panel uses the binary one.
pick_file <- function(oc, arm) {
  if (arm == "actual") {
    dir <- file.path(wss_dir, oc, "actual_output")
    pat <- sprintf("^GNPC\\..*%s\\.linreg_glm\\.csv(\\.gz)?$", oc)
  } else {
    dir <- file.path(wss_dir, oc)
    pat <- sprintf("^synthetic_output\\.%s\\.linreg_glm\\.csv(\\.gz)?$", oc)
  }
  files <- list.files(dir, pattern = pat, full.names = TRUE)
  if (length(files) == 0) return(NA_character_)
  if (length(files) > 1) {
    want <- if (grepl("_bin$", oc)) "binary" else "continuous"
    hit <- grep(want, files, value = TRUE)
    files <- if (length(hit) > 0) hit[1] else files[1]
  }
  files[1]
}

all_dt <- list()
for (oc in outcomes) {
  for (arm in c("actual", "sphere")) {
    f <- pick_file(oc, arm)
    if (is.na(f)) {
      cat(sprintf("[gnpc] missing %s / %s, skipping\n", oc, arm))
      next
    }
    d <- fread(f)
    d[, dx := paste0(oc, " (", arm, ")")]
    d[, padj := p.adjust(pval, method = "BH")]
    d[, LogP := -log10(pval) * sign(beta)]
    all_dt <- c(all_dt, list(d))
  }
}

if (length(all_dt) == 0) {
  cat("[gnpc] no GLM result files found\n")
  quit(status = 1)
}

dt <- rbindlist(all_dt, fill = TRUE)
fwrite(dt, file.path(out_dir, "gnpc_wss_heatmap.csv"))
cat(sprintf("[gnpc] -> %s/gnpc_wss_heatmap.csv  (%d modules x %d columns)\n",
            out_dir, length(unique(dt$ix)), length(unique(dt$dx))))
