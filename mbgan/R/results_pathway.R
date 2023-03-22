#### Results Analysis ----
library(tidyverse)

file_names <- list.files('logs/', pattern = '.csv', full.names = T)

df <- map2_dfr(
  .x = file_names,
  .y = file_names,
  .f = ~ read_csv(.x, col_types = cols(.default = col_character())) %>%
    rename_with( ~ c('iter', 'loss')) %>%
    mutate(nodes = if_else(str_detect(.y, "_n64_"), "64", "128")) %>%
    mutate(transform = if_else(
      str_detect(.y, "_LT"), "Log-Transform", "None"
    )) %>%
    mutate(type = if_else(str_detect(.y, "AE_"), "AE", "FFN")) %>%
    mutate(architect = str_extract(.y, "1e1d|3e3d|3e5d|3e")) %>%
    mutate(batch_size = str_extract(.y, "b32|b128")) %>%
    mutate(loss_type = str_extract(.y, "L1|mse|kl|swd"))
) %>%
  mutate(loss = as.numeric(loss)) %>%
  mutate(iter = as.numeric(iter))

df %>%
  filter(batch_size == 'b32',
         transform == 'Log-Transform') %>% 
  mutate(loss_type = str_to_upper(loss_type)) %>%
  mutate(batch_size_architect = paste0(architect, " ", type)) %>%
  ggplot(aes(x = iter, y = loss, color = batch_size_architect)) + 
  facet_wrap(~ loss_type + nodes, scales = 'free', nrow = 2) +
  geom_point(size = 3) +
  geom_line(linewidth = 1) +
  theme_classic(base_size = 16) +
  labs(x = "Iteration", y = "Loss",
       title = "Log-Transform, Batch Size 32",
       color = "Architecture & Nodes") +
  scale_color_brewer(palette = "Set1")



df %>%
  filter(batch_size == 'b32',
         nodes == '128',
         transform == 'Log-Transform') %>% 
  mutate(loss_type = str_to_upper(loss_type)) %>%
  mutate(batch_size_architect = paste0(architect, " ", batch_size)) %>%
  ggplot(aes(x = iter, y = loss, color = batch_size_architect)) + 
  facet_wrap(~ loss_type + nodes, scales = 'free', nrow = 2) +
  geom_point(size = 3) +
  geom_line(linewidth = 1) +
  theme_classic(base_size = 16) +
  labs(x = "Iteration", y = "Loss",
       title = "Log-Transform, Batch Size 32",
       color = "Architecture & Nodes") +
  scale_color_brewer(palette = "Set1")
