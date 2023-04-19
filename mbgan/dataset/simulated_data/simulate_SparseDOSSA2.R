library(readr)
library(dplyr)
library(tidyr)
library(rlist)
library(SparseDOSSA2)
library(tibble)

#### Load data ----
wgs <- read_csv('dataset/wgs_species_ra.csv')
s16 <- read_csv('dataset/s16_genus_ra.csv')

# Put data into a matrix-form suitable for fit_SparseDOSSA2
# feature-by-sample matrix of abundances
wgs_matrix <- wgs %>%
  pivot_longer(-sample) %>%
  pivot_wider(names_from = sample, values_from = value) %>%
  select(-name) %>%
  as.matrix()


#### Simple SparseDOSSA2 ----
simplefit <- fit_SparseDOSSA2(wgs_matrix)

# Not all features carried over; extract them from the original frame
simulated_wgs_holder <- SparseDOSSA2(simplefit, n_sample=1585, new_features=F)

simulated_wgs <- simulated_wgs_holder$simulated_data

simulated_wgs_otu <- wgs %>%
  pivot_longer(-sample, names_to = 'otu', values_to = 'ra') %>%
  select(otu) %>%
  distinct() %>% 
  filter(simulated_wgs_holder$template$l_filtering$ind_feature) %>%
  pull(otu)

simulated_wgs <- simulated_wgs %>%
  as_tibble() %>%
  mutate(otu = simulated_wgs_otu) %>%
  relocate(otu, .before = Sample1)


#### Convert simulated WGS to simulated S16 ----
# Remove Virus and Eukar (they do not occur in S16)
# remove species, group by genus and sum RA
# then standardize RA
simulated_s16 <- simulated_wgs %>% 
  separate(otu, into = c('kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species'), sep = "\\|") %>%
  select(-species) %>%
  filter(kingdom %in% c('k__Archaea', 'k__Bacteria')) %>%
  unite('otu', kingdom:genus, sep = "|") %>%
  pivot_longer(-otu, names_to = 'sample', values_to = 'RA') %>%
  group_by(otu, sample) %>%
  summarise(RA = sum(RA)) %>%
  ungroup() %>%
  group_by(sample) %>%
  mutate(RA = RA/sum(RA)) %>%
  ungroup() %>%
  pivot_wider(names_from = sample, values_from = RA)


#### Add white noise to simulated S16 ----
# Use log-normal noise to add to the data.
# Maintains composition data requirement without
# requiring tranformation via LR

# Set seed
set.seed(42)

# Log-normal parameters
mu <- 0
sigma <- 0.1

# Noise
noise <- matrix(rlnorm(length(simulated_s16), meanlog = mu, sd = sigma), 
                nrow = nrow(simulated_s16),
                ncol = ncol(simulated_s16))

noise_otu <- simulated_s16 %>% pull(otu)

noisy_s16 <- ((simulated_s16 %>% select(-otu)) * noise)

normalized_noisy_s16 <- sweep(noisy_s16, 2, colSums(noisy_s16), FUN = "/") %>%
  as_tibble() %>%
  mutate(otu = noise_otu) %>%
  relocate(otu, .before = Sample1)

rm(noise_otu, noise, mu, sigma, noisy_s16)


#### Train / Validate data frames ----
# Create a train/validate split and save the data

##### Pivot rows & columns ----
# First pivot rows to columns
simulated_s16 <- normalized_noisy_s16 %>%
  pivot_longer(-otu, names_to = 'sample', values_to = 'ra') %>%
  pivot_wider(names_from = 'otu', values_from = 'ra')

simulated_wgs <- simulated_wgs %>%
  pivot_longer(-otu, names_to = 'sample', values_to = 'ra') %>%
  pivot_wider(names_from = 'otu', values_from = 'ra')

rm(normalized_noisy_s16)

train_rows <- sample(1:1585, size = 785)


##### Select rows ----
train_simulated_wgs <- simulated_wgs[train_rows,]
train_simulated_16s <- simulated_s16[train_rows,]

valid_simulated_wgs <- simulated_wgs[-train_rows,]
valid_simulated_16s <- simulated_s16[-train_rows,]


##### Write .csv files ----
write_csv(train_simulated_16s, 'dataset/simulated_data/train_sim_16s.csv')
write_csv(train_simulated_wgs, 'dataset/simulated_data/train_sim_wgs.csv')
write_csv(valid_simulated_16s, 'dataset/simulated_data/valid_sim_16s.csv')
write_csv(valid_simulated_wgs, 'dataset/simulated_data/valid_sim_wgs.csv')



