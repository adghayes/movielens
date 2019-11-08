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

# Calculate movie effects from rating set
f.effects <- function(data){
  mu <- mean(data$rating)
  data %>%
    group_by(movieId) %>%
    summarise(
      effect = mean(rating) - mu, 
      n = n(),
      title = first(title),
      genres = first(genres)
    )
}

# Get mu and raw_effects for entire dataset
# rather than test train, for Shiny app visualization
all <- union(edx, validation)
mu <- mean(all$rating)
raw_effects <- f.effects(all) %>%
  extract(title, c("title","year"), "(.*)\\s\\((\\d{4})\\)")

# Regularize Movie Effects
f.regularize <- function(effects, lambda){
  effects %>%
    mutate(effect = effect*n/(n + lambda))
}

# Find RMSEs of predictions on test set
# with effects from train set, for all lambdas
train_effects <- f.effects(edx)
train_mu <- mean(edx$rating)
rmses <- sapply(lambdas, function(lambda){
  reg_effects <- f.regularize(train_effects, lambda)
  prediction <- validation %>% 
    left_join(reg_effects, by = "movieId") %>%
    mutate(prediction = train_mu + effect) %>%
    pull(prediction)
  RMSE(prediction, validation$rating)
})

rmses_tbl <- tibble(lambda = lambdas, rmse = rmses)

save(mu, raw_effects, f.regularize, rmses_tbl, 
     lambda_min, lambda_max, lambda_step, file = "./data.rda")