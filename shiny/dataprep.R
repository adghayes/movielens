#################################################
#################################################
## SUMMARIZE DATA FOR SHINY REGULARIZATION APP ##
#################################################
#################################################

# Code for downloading and loading movielens
# files are borrowed from edx course material

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp")) %>%
  select(movieId, rating)

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

##################################
####### Prepare App Data #########
##################################
ps <- seq(.1, .9, .1)
lambda_step <- .01
lambdas <- seq(3,3, by = lambda_step)


# Calculates Movie Effects w/o Regularization
f.raw_effects <- function(ratings){
  mu <- mean(ratings$rating)
  ratings %>%
    group_by(movieId) %>%
    summarise(effect = mean(rating) - mu, n = n())
}

raw_effects <- f.raw_effects(ratings)

# Adjusts Effects w/ Regularization
f.effects <- function(raw_effects, lambda){
  raw_effects %>%
    mutate(effect = effect*n/(n + lambda))
}

# For set of lambdas and single p, find RMSEs
f.rmses <- function(ratings, lambdas, p){
  set.seed(1, sample.kind="Rounding")
  train_idx <- createDataPartition(y = ratings$rating, p = p, list = FALSE)
  test <- ratings[-train_idx,]
  train <- ratings[train_idx,]
  train_mu <- mean(train$rating)
  train_raw_effects <- f.raw_effects(train)
  rmses <- sapply(lambdas, function(lambda){
    train_reg_effects <- f.effects(train_raw_effects, lambda)
    prediction <- test %>% 
      left_join(train_reg_effects, by = "movieId") %>%
      mutate(
        prediction = if_else(!is.na(effect), 
                  train_mu + effect, train_mu)) %>%
      pull(prediction)
      RMSE(prediction, test$rating)
  })
  return(tibble(p = p, lambda = lambdas, rmse = rmses))
}

rmses_list <- lapply(ps, function(p){
  f.rmses(ratings, lambda, p)
  })

rmses <- do.call(rbind, lapply(rmses_list, data.frame))


save(movies, raw_effects, f.effects, rmse_dfs)

