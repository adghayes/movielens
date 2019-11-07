#################################################
#################################################
## SUMMARIZE DATA FOR SHINY REGULARIZATION APP ##
#################################################
#################################################

library(tidyverse)
library(caret)
load("../data/movielens.rda")

lambda_min <- 0
lambda_max <- 5
lambda_step <- .01
lambdas <- seq(lambda_min, lambda_max, by = lambda_step)


# Calculates Movie Effects w/o Regularization
mu <- mean(edx$rating)

raw_effects <- edx %>%
  group_by(movieId) %>%
  summarise(
    effect = mean(rating) - mu, 
    n = n(),
    title = first(title),
    genres = first(genres)
    ) %>%
  extract(title, c("title","year"), "(.*)\\s\\((\\d{4})\\)")

# Adjusts Effects w/ Regularization
f.effects <- function(raw_effects, lambda){
  raw_effects %>%
    mutate(effect = effect*n/(n + lambda))
}

# Find RMSEs for all lambdas
rmses <- sapply(lambdas, function(lambda){
  reg_effects <- f.effects(raw_effects, lambda)
  prediction <- validation %>% 
    left_join(reg_effects, by = "movieId") %>%
    mutate(prediction = mu + effect) %>%
    pull(prediction)
  RMSE(prediction, validation$rating)
})

rmses_tbl <- tibble(lambda = lambdas, rmse = rmses)

save(mu, raw_effects, f.effects, rmses_tbl, lambda_min, lambda_max, lambda_step, file = "./data.rda")
