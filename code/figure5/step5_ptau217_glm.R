#!/usr/bin/env Rscript
#' ---------------------------
#'
#' Script name: 02-linreg_glm.R
#'
#' Version: 0.0.4
#'
#' Purpose of script: calculate correlation of plasma ptau with SomaLogic plasma
#' proteomic levels
#'
#' Author: Edward Wilson, Robert B Butler III, Michael P Brown
#'
#' Date Created: 2023-08-25
#'
#' Copyright (c) 2023
#' Email: rrbutler@stanford.edu
#'
#' ---------------------------
#'
#' Notes:
#'   v0.0.4 - adding threads, bugfix recodes, shielding outcome variable
#'   Linear regression glm for baseline visit only with corrections 
#'   for fixed effects: filter covariates based on importance
#'   
#'   PlateRunDate & ScannerID couldn't be fit to the model, suggesting some 
#'   other issue, so dropping those
#'   
#'   This takes a while, so batch unless you like waiting
#'   Usage:
#'     sbatch -J ptau --mem=12G -c 16 -t 06:00:00 -p interactive \
#'       -o %A_02-linreg_lme4.log \
#'       --wrap "ml R/4.2.2; Rscript 02-linreg_glm.R"
#'
#'   REMEMBER, WHEN COMMENTING IN CODE, DO NOT REFERENCE INDIVIDUAL INFORMATION:
#'   SAMPLES/PIDNS, DATES OR ANY INDIVIDUAL LEVEL INFO
#'
#' ---------------------------

# load up the packages we will need:  (uncomment as required)
suppressPackageStartupMessages({
  library(tictoc)
  tic()
  library(data.table)
  library(optparse)
  library(tools)
  library(stringr)
  # library(lmerTest) # loads lme4
})

# Nameset helper function
nameset_default <- function(input){
  file_path_sans_ext(basename(input))
}

# read in args
option_list <- list(
  make_option(
    c("-i", "--input"),
    type = "character",
    default = ".",
    help = "File path to DE input file (*.csv.gz)[default: %default]"
  ),
  make_option(
    c("-o", "--output"),
    type = "character",
    default = ".",
    help = "File path to output directory [default: %default]"
  ),
  make_option(
    c("-p", "--prots"),
    type = "character",
    default = "../prot_names.txt",
    help = "File path to prots file [default: %default]"
  ),
  make_option(
    c("-v", "--covs"),
    type = "character",
    default = "",
    help = "Covariates in comma-separated format [default: %default]"
  ),
  make_option(
    c("-d", "--recodes"),
    type = "character",
    default = "",
    help = "Recode covariates in comma-separated format and pipe between covs [default: %default]"
  ),
  make_option(
    c("-n", "--nameset"),
    type = "character",
    help = "Name of project [default: Nameset of input]"
  ),
  make_option(
    c("-t", "--threads"),
    default = 0,
    help = "number of threads [default: setDTthreads]"
  ),
  make_option(
    c("-c", "--outcome"),
    type = "character",
    help = "String specifying the column in data containing the outcome variable [required]"     
  )
)

args_list <- parse_args(OptionParser(option_list = option_list))

# Nameset default case
if(is.null(args_list$nameset)){
  args_list$nameset <- nameset_default(args_list$input)
}

# start logfile
con <- file(
  file.path(args_list$output, paste(args_list$nameset, "linreg.log", sep = "."))
)
sink(con, append = TRUE)
sink(con, append = TRUE, type = "message")

# read in fields
fields <- c(
  "input",
  "output",
  "prots",
  "covs",
  "recodes",
  "nameset",
  "threads",
  "outcome"
)

# check for values
for (field in fields) {
  if (!field %in% names(args_list)) {
    cat(str_c('Argument "', field, '" must be provided\n'), fill = TRUE)
    stop(parse_args(OptionParser(option_list = option_list), args = "--help"),
      call. = FALSE
    )
  }

  assign(field, args_list[[field]])
  print(paste0(field, ": ", get(field)))
}

# handle threading
setDTthreads(threads = threads)

# variables
covs <- strsplit(covs, ",")[[1]]
prots <- readLines(prots)
recodes <- strsplit(unlist(strsplit(recodes, "\\|")), ",")

# Functions ----------------------------------------------------------------
#' Generalized linear model based on a flexible formula
#'
#' @param dat Data.table with the original data
#' @param outcome String specifying the column in dat containing the outcome variable 
#' @param exposure String specifying the column in dat containing the exposure variable
#' @param covs Vector of column names containing covariates to be included in the formula
#'
#' @returns A named list: 'Estimate' (beta), 'Std. Error' (se), 't value' (test statistic), 'Pr(>|t|)' (p-value), 'nobs' (number of observations)
get_linreg = function(dat, outcome, exposure, covs=c()){
  model = glm(paste(outcome, "~", paste(c(exposure, covs), collapse=" + ")),
              na.action=na.exclude, data=dat)
  a = summary(model)
  return(as.list(c(a$coefficients[2, ], nobs=nobs(model))))
}

#' Recodes dummy categorical variables and can subset by vector of categories
#'
#' @param dat Data.table with the original data
#' @param dcol String representing the column to be split
#' @param keep Vector of strings that are the categories within dcol to extract. default is all c()
#'
#' @returns modified data.table with new dummy variable columns named "dcol.category" after original column
recode_dummies = function(dat, dcol, keep=c()){
  dt = dat
  # check input
  if (length(keep) == 0){
    inds <- unique(dt[[dcol]])
  } else if (length(keep) > 0 & all(keep %in% unique(dt[[dcol]]))){
    inds <- unique(dt[[dcol]])[unique(dt[[dcol]]) %in% keep]
  } else {
    stop("Keep must be a vector of names in the original column")
  }
  # split out dummies
  dt[, c(paste(dcol, inds, sep=".")) := lapply(inds, function(x) ifelse(get(dcol) == x, 1, 0))]
  # reorder behind original variable
  v = grep(paste0("^", dcol, "$"), names(dt))[1]
  inds = paste(dcol, inds, sep=".")
  setcolorder(dt, c(names(dt)[1:v], inds, names(dt)[(v + 1):(ncol(dt) - length(inds))]))
  return(dt)
}


# Main -------------------------
dat = fread(input)
names(dat) = make.names(names(dat))
names(prots) <- make.names(prots)
# similar to other name shielding, but reversed for ease of use
orig_outcome <- outcome
outcome <- make.names(outcome)

# add in single category columns with feature importance
# PlateId was important, but P0028449, P0028565, P0028437, P0028430, P0028452, P0028448 were top30 in either glmcross-validated or plr estimates
# dat = recode_dummies(dat, "PlateId", keep=c("P0028449", "P0028565", "P0028437", "P0028430", "P0028452", "P0028448"))
if(length(recodes) > 0) {
  new_covs <- c()
  for(recode_item in recodes) {
    if(recode_item[1] %in% covs) {
      dat <- recode_dummies(dat, 
        dcol = recode_item[1],
        keep = recode_item[2:length(recode_item)]
      )
      new_covs <- c(new_covs, paste(recode_item[1], recode_item[2:length(recode_item)], sep = "."))
      covs <- covs[covs != recode_item[1]]
    } else {
      print(sprintf("%s not found in input, dropping & skipping recode...", recode_item[1]))
      covs <- covs[covs != recode_item[1]]
    }
  }
  covs <- c(covs, new_covs)
}

# list covariates
print(paste("Running covs:", paste(covs, collapse=", ")))

# make out table
dt1 = CJ(dx = outcome, ix = names(prots))

# fast run function
dt1[, c("beta", "se", "t stat", "pval", "nobs") := get_linreg(dat, dx, ix, covs), by=.I]

# remove Xs needed for formula
dt1$ix <- plyr::mapvalues(x = dt1$ix, from = names(prots), to = prots)
dt1$dx <- plyr::mapvalues(x = dt1$dx, from = outcome, to = orig_outcome)
# dt1[startsWith(ix, "X."), ix := gsub("^X.", ".", ix)]
fwrite(dt1, 
  file = file.path(output, paste(nameset, "linreg_glm", "csv.gz", sep="."))
)

# cleanup
toc()
# system("squeue -j $SLURM_JOBID --Format=TimeUsed")


# # model testing ---------------------
# # spot check 10 proteins
# set(123)
# prots = sample(names(dat)[grep(first_prot, names(dat))[1]:last(grep(last_prot, names(dat)))], 10)

# model = glm(paste("PlasmaPTau181", "~", paste(c(prots[2], cols), collapse=" + ")),
             # na.action=na.exclude, data=dat)
# summary(model)
# plot(fitted(model), resid(model, type = "pearson")) 
# abline(0,0, col="red")
# shapiro.test(resid(model, type="pearson"))


