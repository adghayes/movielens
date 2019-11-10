#######################################
#######################################
####### GET DATA FROM SOURCE ##########
#######################################
#######################################
#* Download code adapted from edx Data Science: Capstone
dl<- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-20m.zip", dl)

# Import from file to df
library(readr)
ratings <- read_csv(unzip(dl, "ml-20m/ratings.csv")) %>%
  select(-timestamp)
movies <- read_csv(unzip(dl, "ml-20m/movies.csv"))

#################################################
#################################################
## SUMMARIZE DATA FOR SHINY REGULARIZATION APP ##
#################################################
#################################################
library(dplyr)
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
      n = n()
    )
}

# Get mu and raw_effects for entire dataset
# rather than test train, for Shiny app visualization
# all <- union(edx, validation)
mu <- mean(ratings$rating)
raw_effects <- f.effects(ratings) %>%
  left_join(movies, by = "movieId") %>%
  extract(title, c("title","year"), "(.*)\\s\\((\\d{4})\\)") %>%
  mutate(title = str_remove(title, "\\s+\\(.*\\)")) %>% 
  mutate(title = str_replace(title, "^(.*),\\s(The|A|An|Le|Les|La|Las|El|Lose|Das|Da|De)$", "\\2 \\1"))

# Regularize Movie Effects
f.regularize <- function(effects, lambda){
  effects %>%
    mutate(effect = effect*n/(n + lambda))
}

# Partition Data, 
# Test set will be 10% of MovieLens data
library(caret)
set.seed(1)
test_index <- createDataPartition(y = ratings$rating, times = 1, p = 0.1, list = FALSE)
train <- ratings[-test_index,]
test <- ratings[test_index,]

# Find RMSEs of predictions on test set
# with effects from train set, for all lambdas
# For unkonwn movies, guess mu
train_effects <- f.effects(train)
train_mu <- mean(train$rating)
rmses <- sapply(lambdas, function(lambda){
  reg_effects <- f.regularize(train_effects, lambda)
  prediction <- test %>% 
    left_join(reg_effects, by = "movieId") %>%
    mutate(prediction = if_else(is.na(effect),
                                train_mu,
                                train_mu + effect)) %>%
    pull(prediction)
  RMSE(prediction, test$rating)
})

# df representing relationship of lambda to RMSE
rmses_tbl <- tibble(lambda = lambdas, rmse = rmses)

# Save to file for easy loading for Shiny App
save(mu, raw_effects, f.regularize, rmses_tbl, 
     lambda_min, lambda_max, lambda_step, file = "./data.rda")
