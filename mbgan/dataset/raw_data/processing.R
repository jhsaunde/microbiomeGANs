
#### 16S & WGS Processing ----
# Transforms 16S & WGS data 
# links across the two data sets
# and transforms that each row is a 
# subject and each column an OTU
# at a given taxonomic resolution
library(tidyverse)
library(here)
wgs <- read_tsv("dataset/raw_data/diabimmune_karelia_metaphlan_table.txt")
s16 <- read_tsv("dataset/raw_data/diabimmune_karelia_16s_otu_table.txt")
meta <- read_tsv("dataset/raw_data/diabimmune_karelia_metadata.txt")



#### 16S ----
# Some OTU were incompletely parsed
# such that rather than have a set of rows specify the entire taxonomic
# tree for a given 'sample', it would end instead.
# Example: 
#   k p c
#   k p c o
#   k p c o f
#   k p c o f g s s <- this row should specify genus, instead jumps to strain
#   recalculated total RA for each level  

# Strain
strain <- s16 %>%
  separate(sample, into = c("kingdom", "phylum", "class", "order", "family", "genus", "species", "strain"), sep = "\\|", remove = F) %>%
  filter(!is.na(strain))


# Species
species <- s16 %>%
  separate(sample, into = c("kingdom", "phylum", "class", "order", "family", "genus", "species", "strain"), sep = "\\|", remove = F) %>%
  filter(!is.na(strain)) %>%
  unite(col = 'flag', kingdom:genus, sep = "|", remove = T, na.rm = T) %>%
  group_by(flag, species) %>% 
  summarise(across(-c(sample:strain), ~ sum(.x))) %>%
  ungroup() %>%
  unite(col = 'sample', flag:species, sep = "|", remove = T, na.rm = T)

# Genus 
genus <- s16 %>%
  separate(sample, into = c("kingdom", "phylum", "class", "order", "family", "genus", "species", "strain"), sep = "\\|", remove = F) %>%
  filter(!is.na(strain)) %>%
  unite(col = 'flag', kingdom:family, sep = "|", remove = T, na.rm = T) %>%
  group_by(flag, genus) %>% 
  summarise(across(-c(sample:strain), ~ sum(.x))) %>%
  ungroup() %>%
  unite(col = 'sample', flag:genus, sep = "|", remove = T, na.rm = T)

# Family
family <- s16 %>%
  separate(sample, into = c("kingdom", "phylum", "class", "order", "family", "genus", "species", "strain"), sep = "\\|", remove = F) %>%
  filter(!is.na(strain)) %>%
  unite(col = 'flag', kingdom:order, sep = "|", remove = T, na.rm = T) %>%
  group_by(flag, family) %>% 
  summarise(across(-c(sample:strain), ~ sum(.x))) %>%
  ungroup() %>%
  unite(col = 'sample', flag:family, sep = "|", remove = T, na.rm = T)

# Order
order <- s16 %>%
  separate(sample, into = c("kingdom", "phylum", "class", "order", "family", "genus", "species", "strain"), sep = "\\|", remove = F) %>%
  filter(!is.na(strain)) %>%
  unite(col = 'flag', kingdom:class, sep = "|", remove = T, na.rm = T) %>%
  group_by(flag, order) %>% 
  summarise(across(-c(sample:strain), ~ sum(.x))) %>%
  ungroup() %>%
  unite(col = 'sample', flag:order, sep = "|", remove = T, na.rm = T)

# Class
class <- s16 %>%
  separate(sample, into = c("kingdom", "phylum", "class", "order", "family", "genus", "species", "strain"), sep = "\\|", remove = F) %>%
  filter(!is.na(strain)) %>%
  unite(col = 'flag', kingdom:phylum, sep = "|", remove = T, na.rm = T) %>%
  group_by(flag, class) %>% 
  summarise(across(-c(sample:strain), ~ sum(.x))) %>%
  ungroup() %>%
  unite(col = 'sample', flag:class, sep = "|", remove = T, na.rm = T)

# Phylum
phylum <- s16 %>%
  separate(sample, into = c("kingdom", "phylum", "class", "order", "family", "genus", "species", "strain"), sep = "\\|", remove = F) %>%
  filter(!is.na(strain)) %>%
  unite(col = 'flag', kingdom:kingdom, sep = "|", remove = T, na.rm = T) %>% 
  group_by(flag, phylum) %>% 
  summarise(across(-c(sample:strain), ~ sum(.x))) %>%
  ungroup() %>%
  unite(col = 'sample', flag:phylum, sep = "|", remove = T, na.rm = T)

# Kingdom
kingdom <- s16 %>%
  separate(sample, into = c("kingdom", "phylum", "class", "order", "family", "genus", "species", "strain"), sep = "\\|", remove = F) %>%
  filter(!is.na(strain)) %>%
  group_by(kingdom) %>% 
  summarise(across(-c(sample:strain), ~ sum(.x))) %>%
  ungroup() %>%
  rename(sample = kingdom)

# Bind all together
s16 <- strain %>%
  select(-c(kingdom:strain)) %>%
  bind_rows(species,
            genus, 
            family, 
            order,
            class,
            phylum, 
            kingdom)

##### 16S RA  -----
# (1) 16S data is counts, convert to relative abundance, scale 0 - 1
# (2) rename person identifier to match wgs
s16 <- s16 %>%
  pivot_longer(-sample, names_to = 'SampleID', values_to = 'RA') %>%
  group_by(SampleID) %>%
  mutate(total = sum(RA)) %>%
  ungroup() %>%
  mutate(RA = RA/total) %>%
  select(-total) %>%
  rename(OTU = sample) %>%
  pivot_wider(names_from = SampleID, values_from = RA) %>%
  arrange(OTU)


#### Create Files ----
# Create 16S files for genus and species
# but have rows as samples and columns as OTU 

s16_species_ra <- s16 %>%
  separate(OTU, into = c("kingdom", "phylum", "class", "order", "family", "genus", "species", "strain"), sep = "\\|", remove = F) %>%
  filter(is.na(strain), !is.na(species)) %>%
  select(-c(kingdom:strain)) %>%
  pivot_longer(-OTU, names_to = 'sample', values_to = 'RA') %>%
  pivot_wider(names_from = OTU, values_from = RA)


s16_genus_ra <- s16 %>%
  separate(OTU, into = c("kingdom", "phylum", "class", "order", "family", "genus", "species", "strain"), sep = "\\|", remove = F) %>%
  filter(is.na(species), !is.na(genus)) %>%
  select(-c(kingdom:strain)) %>%
  pivot_longer(-OTU, names_to = 'sample', values_to = 'RA') %>%
  pivot_wider(names_from = OTU, values_from = RA)


# 16S: 2195 total OTU
# WGS: 1599 total OTU

#### WGS ----
# (1) WGS data is from 0 - 100 relative abundance, scale to 0 - 1
# (2) WGS data uses 'gid_wgs' to indicate persons while 16S uses SampleID
#     replace WGS identifier with S16 identifier
# (3) rename 'ID' column (containing taxo. info.) to 'sample' (S16 name)
wgs <- wgs %>%
  pivot_longer(-ID, names_to = 'IDs', values_to = 'RA') %>%
  mutate(RA = RA/100) %>% # scale
  left_join(meta %>% # join by ID
              select(SampleID, gid_16s, gid_wgs), by = c("IDs" = "gid_wgs")) %>%
  select(ID, SampleID, RA) %>%
  rename(OTU = ID) %>%
  pivot_wider(names_from = 'SampleID', values_from = RA)

wgs_species_ra <- wgs %>%
  separate(OTU, into = c("kingdom", "phylum", "class", "order", "family", "genus", "species", "strain"), sep = "\\|", remove = F) %>%
  filter(is.na(strain), !is.na(species)) %>%
  select(-c(kingdom:strain)) %>%
  pivot_longer(-OTU, names_to = 'sample', values_to = 'RA') %>%
  pivot_wider(names_from = OTU, values_from = RA)
  
#### Save Full Datasets ----
write_csv(x = wgs_species_ra, "/Users/jamessaunders/dev/microbiomeGANs/mbgan/dataset/wgs_species_ra.csv")
write_csv(x = s16_genus_ra, "/Users/jamessaunders/dev/microbiomeGANs/mbgan/dataset/s16_genus_ra.csv")
write_csv(x = s16_species_ra, "/Users/jamessaunders/dev/microbiomeGANs/mbgan/dataset/s16_species_ra.csv")


#### Test / Train ----
# No matches available
nomatches_genus <- s16_genus_ra %>%
  filter(!sample %in% wgs_species_ra$sample)

nomatches_species <- s16_species_ra %>%
  filter(!sample %in% wgs_species_ra$sample)

write_csv(x = nomatches_genus, "/Users/jamessaunders/dev/microbiomeGANs/mbgan/dataset/s16_genus_nomatches.csv")

write_csv(x = nomatches_species, "/Users/jamessaunders/dev/microbiomeGANs/mbgan/dataset/s16_species_nomatches.csv")


# Matches available, train / test 80:20
# First setup variables
s16_genus_ra <- s16_genus_ra %>%
  filter(sample %in% wgs_species_ra$sample)

wgs_species_ra <- wgs_species_ra %>%
  filter(sample %in% s16_genus_ra$sample)

n_samples_total <- length(wgs_species_ra$sample)

n_samples_training <- floor(0.8*n_samples_total)

set.seed(42)

rows_for_training <- sample(1:n_samples_total, n_samples_training)

##### Train / Test Split -----

train_wgs_species <- wgs_species_ra[rows_for_training,]
test_wgs_species <- wgs_species_ra[-rows_for_training,]

train_16s_genus <- s16_genus_ra[rows_for_training,]
test_16s_genus <- s16_genus_ra[-rows_for_training,]

train_16s_species <- s16_species_ra[rows_for_training,]
test_16s_species <- s16_species_ra[-rows_for_training,]

write_csv(x = train_wgs_species, "/Users/jamessaunders/dev/microbiomeGANs/mbgan/dataset/train_wgs_species.csv")
write_csv(x = test_wgs_species, "/Users/jamessaunders/dev/microbiomeGANs/mbgan/dataset/validate_wgs_species.csv")
write_csv(x = train_16s_genus, "/Users/jamessaunders/dev/microbiomeGANs/mbgan/dataset/train_16s_genus.csv")
write_csv(x = test_16s_genus, "/Users/jamessaunders/dev/microbiomeGANs/mbgan/dataset/validate_16s_genus.csv")
write_csv(x = train_16s_species, "/Users/jamessaunders/dev/microbiomeGANs/mbgan/dataset/train_16s_species.csv")
write_csv(x = test_16s_species, "/Users/jamessaunders/dev/microbiomeGANs/mbgan/dataset/validate_16s_species.csv")

