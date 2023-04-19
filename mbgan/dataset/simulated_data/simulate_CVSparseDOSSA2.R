library(readr)
library(dplyr)
library(tidyr)
library(rlist)
library(SparseDOSSA2)
library(tibble)

#### Load data ----
wgs <- read_csv('dataset/wgs_species_ra.csv')

# Put data into a matrix-form suitable for fit_SparseDOSSA2
# feature-by-sample matrix of abundances
wgs_matrix <- wgs %>%
  pivot_longer(-sample) %>%
  pivot_wider(names_from = sample, values_from = value) %>%
  select(-name) %>%
  as.matrix()

plan(sequential, sequential, multisession)

fitted <- fitCV_SparseDOSSA2(data = wgs_matrix, lambdas = c(0.1, 1), K = 3, control = list(verbose=TRUE))

list.save(fitted, 'list.rds')
list.save(fitted, 'list.rdata')
list.save(fitted, 'list.yaml')