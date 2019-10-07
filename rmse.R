#########################################
#########################################
## PREDICT RATINGS AND CALCULATE RMSES ##
#########################################
#########################################
library(tidyverse)
library(caret)

this.dir <- dirname(parent.frame(2)$ofile)
setwd(this.dir)
load("./data/movielens.rda")

###################################
#   SIMPLE AVERAGE APPROACH (c)   #
###################################
mu <- edx %>% pull(rating) %>% mean()

c_predict <- function(newdata){
  rep(mu, nrow(newdata))
}

c_prediction <- c_predict(validation)
c_RMSE <- RMSE(validation$rating, c_prediction)
c_RMSE

###########################################
#   MOVIE + USER EFFECTS APPROACH (muf)   #
###########################################
movie_effects <- edx %>% 
  group_by(movieId) %>%
  summarize(m_i = mean(rating - mu), n_i = n())

user_effects <- edx %>% 
  left_join(movie_effects, by = "movieId") %>%
  group_by(userId) %>%
  summarize(u_j = mean(rating - mu - m_i), n_j = n())

# Helper function to deal with out of bounds predictions
bounded <- function(rating){
  case_when(
   rating > 5 ~ 5,
   rating < .5 ~ .5,
   TRUE ~ rating
  )
}

muf_predict <- function(newdata){
  newdata %>% 
    left_join(movie_effects, by = "movieId") %>%
    left_join(user_effects, by = "userId") %>%
    mutate(rating = bounded(mu + m_i + u_j)) %>%
    pull(rating)
}

muf_prediction <- muf_predict(validation)
muf_RMSE <- RMSE(validation$rating, muf_prediction)
muf_RMSE

##########################################################
# MOVIE + USER EFFECTS APPROACH w/ REGULARIZATION (mufr) #
##########################################################
# Need to use cross validation to optimize the lambdas, a
# lambda_m for movie effects and lambda_u for user effects

# Model which can be trained on various train
# sets for cross-validation
mufr_train_predict <- function(data, newdata, lambda){
  movie_effects_r_temp <- data %>% 
    group_by(movieId) %>%
    summarize(
      m_i = sum(rating - mu)/(n() + lambda[1]),
      n_i = n()
    )
  
  user_effects_r_temp <- data %>% 
    left_join(movie_effects_r_temp, by = "movieId") %>%
    group_by(userId) %>%
    summarize(
      u_j = sum(rating - mu - m_i)/(n() + lambda[2]),
      n_j = n()
    )
  
  newdata %>% 
    left_join(movie_effects_r_temp, by = "movieId") %>%
    left_join(user_effects_r_temp, by = "userId") %>%
    mutate(rating = bounded(mu + m_i + u_j)) %>%
    pull(rating)
}

# Creating Folds
k <- 10
k
set.seed(23, sample.kind="Rounding")
edx_folds <- createFolds(y = edx$rating, k = k)

# Removing unknown users/movies from test partitions, 
# as in the creation of the full dataset
clean_edx_fold <- function(fold){
  test <- bind_cols(edx[fold,], index = fold)
  train <- edx[-fold,]
  test <- test %>%
    semi_join(train, by = "movieId") %>%
    semi_join(train, by = "userId")
  test$index
}

edx_fold_sizes_pre <- sapply(edx_folds, length) # pre-clean sizes
edx_folds <- lapply(edx_folds, clean_edx_fold)
edx_fold_sizes_post <- sapply(edx_folds, length) # post-clean sizes

edx_fold_summary <- data.frame(name = names(edx_folds),
                               intial_size = edx_fold_sizes_pre,
                               clean_size = edx_fold_sizes_post)
edx_fold_summary

# Function to calculate mean of RMSEs across folds given a lambda, for optim
cross_validate_rmse <- function(lambda){
  mean(sapply(edx_folds, function(fold){
    RMSE(mufr_train_predict(edx[-fold,],edx[fold,],lambda), edx[fold,]$rating)
  }))
}

# Calibrate lambda with Nelder-Mead optimization
lambda_0 <- c(lambda_m = 4.70262, lambda_u = 4.96466)
lambda_optim <- optim(par = lambda_0, fn = cross_validate_rmse, 
                      control = list(maxit = 25, trace = TRUE))
lambda <- lambda_optim$par
lambda_optim

# Understand lambda's neighborhood (to plot in report)
lambda_m <- lambda[1]
lambda_u <- lambda[2]
lambda_ms <- lambda_m + seq(-2,2,1)
lambda_us <- lambda_u + seq(-2,2,1)
tuningData <- expand_grid(lambda_m = lambda_ms, lambda_u = lambda_us) %>% 
  mutate(value = apply(X = ., MARGIN = 1, FUN = cross_validate_rmse))


# With lambda tuned, create actual model with regularized effects
movie_effects_r <- edx %>% 
  group_by(movieId) %>%
  summarize(
    m_i = sum(rating - mu)/(n() + lambda[1]),
    n_i = n()
  )

user_effects_r <- edx %>% 
  left_join(movie_effects_r, by = "movieId") %>%
  group_by(userId) %>%
  summarize(
    u_j = sum(rating - mu - m_i)/(n() + lambda[2]),
    n_j = n()
  )

mufr_predict <- function(newdata){
  newdata %>% 
    left_join(movie_effects_r, by = "movieId") %>%
    left_join(user_effects_r, by = "userId") %>%
    mutate(rating = bounded(mu + m_i + u_j)) %>% pull(rating)
}

mufr_prediction <- mufr_predict(validation)
mufr_RMSE <- RMSE(validation$rating, mufr_prediction)
mufr_RMSE

#######################################
#  Appendix: RECOMMENDER LAB (rl)     #
#######################################
library(recommenderlab)
n_movies <- 1000
n_users <- 1000

# Normalized, regularized copy of dataset for easy use
edx_c <- edx %>% 
  left_join(movie_effects_r, by = "movieId") %>%
  left_join(user_effects_r, by = "userId") %>%
  mutate(rating = rating - mu - m_i - u_j)

all_movies <- edx_c %>%
  group_by(movieId) %>% summarise(n = n()) %>%
  arrange(desc(n)) %>% pull(movieId)

top_movies <- all_movies[1:n_movies]

all_users <- edx_c %>%
  filter(movieId %in% top_movies) %>%
  group_by(userId) %>% summarise(n = n()) %>%
  arrange(desc(n)) %>% pull(userId)

top_users = all_movies[1:n_users]

# Matrix of ratings for 1000 most common movies and
# 1000 most prolific users, to train recommenderlab models
edx_c_mini <- edx_c %>%
  filter(userId %in% top_users & movieId %in% top_movies) %>%
  select(userId, movieId, rating) %>% as("realRatingMatrix")

# Matrix of ratings for 1000 most common movies for all 
# users in edx, to predict with recommenderlab models
edx_c_top_movies <- edx_c %>%
  filter(movieId %in% top_movies) %>%
  select(userId, movieId, rating) %>% as("realRatingMatrix")

# Training different models on the same data
ubcf_rec <- Recommender(edx_c_mini, method = "UBCF", parameter = 
                          list(method = "Pearson", nn = 10))
ibcf_rec <- Recommender(edx_c_mini, method = "IBCF", parameter = 
                          list(k = 400, method = "Pearson"))
svd_rec <- Recommender(edx_c_mini, method = "SVD", parameter = 
                         list(k = 20))

# Get all ratings possible for movie-user 
# pairs that are not in knownRatings
rl_predict_all <- function(rec, knownRatings){
  recommenderlab::predict(object = rec, newdata = knownRatings, type = "ratings") %>% 
  as("matrix") %>% 
  as.data.frame() %>%
  rownames_to_column(var = "userId") %>%
  remove_rownames() %>%
  gather(movieId, rating, -userId, na.rm = TRUE) %>%
  mutate(
    movieId = as.integer(movieId),
    userId = as.integer(userId)
  ) %>%
    left_join(movie_effects_r, by = "movieId") %>%
    left_join(user_effects_r, by = "userId") %>%
    mutate(rating = rating + mu + m_i + u_j) %>%
    select(movieId, userId, rating)
}

# Get only ratings of movie-user pairs in newdata
rl_predict <- function(rec, knownRatings, newdata){
  newdata %>% select(movieId, userId) %>%
    left_join(rl_predict_all(rec, knownRatings), by = c("movieId", "userId")) %>%
    mutate(rating = bounded(rating)) %>%
    pull(rating)
}

ubcf_prediction <- rl_predict(ubcf_rec, edx_c_top_movies, validation)
ibcf_prediction <- rl_predict(ibcf_rec, edx_c_top_movies, validation)
svd_prediction <- rl_predict(svd_rec, edx_c_top_movies, validation)

predictions <- data.frame(mufr = mufr_prediction,
                          svd = svd_prediction,
                          ibcf = ibcf_prediction,
                          ubcf = ubcf_prediction)

# Combine new predictions with our standard model 
predictions <- predictions %>% mutate_all( ~ if_else(is.na(.), mufr, .))
ubcf_RMSE <- RMSE(validation$rating, predictions$ubcf)
ibcf_RMSE <- RMSE(validation$rating, predictions$ibcf)
svd_RMSE <- RMSE(validation$rating, predictions$svd)
ubcf_RMSE
ibcf_RMSE
svd_RMSE

#####################################################################################
#####################################################################################
#####################################################################################
# Compiling results to save to file for later user in report...

results <- data.frame(Name = c("Simple Average", "Movie + User Effects", 
                               "Regularized Effects (MUFR)", "MUFR + UBCF", 
                               "MUFR + IBCF", "MUFR + SVD"),
                      RMSE = c(c_RMSE, muf_RMSE, mufr_RMSE,
                               ubcf_RMSE, ibcf_RMSE, svd_RMSE))

movies <- edx %>% distinct(movieId, title, genres)

# Need to rerun the models to get recommendations for all movies 
# in top 1000 for 250 users

# Known Ratings for top 250 users
edx_c_250 <- edx_c %>% 
  filter(movieId %in% top_movies & userId %in% top_users[1:250]) %>%
  select(userId, movieId, rating) %>% as("realRatingMatrix")

models <-  list(UBCF = ubcf_rec, IBCF = ibcf_rec, SVD = svd_rec)

# Get Recommender Lab predicted ratings for all unrated 
# movies, don't include user effect
rl_ratings_250 <- lapply(models, function(x){
  rl_predict_all(x, edx_c_250)
}) 

# Get predicted ratings for all unrated movies via MUFR, but 
# no need to include user effects
mufr_ratings_250 <- bind_rows(rl_ratings_250, .id = "model") %>%
  distinct(userId, movieId) %>% 
  left_join(movie_effects_r, by = "movieId") %>%
  mutate(rating = mu + m_i, model = "MUFR") %>%
  select(model, userId, movieId, rating)
  
# Only take top ten per user/model to reduce size for knitr
all_ratings_250 <- bind_rows(rl_ratings_250, .id = "model") %>%
  bind_rows(mufr_ratings_250) %>%
  group_by(model, userId) %>%
  group_modify(~ {
    .x %>% 
      arrange(desc(.x$rating)) %>%
      head(10L)
    }) %>%
  ungroup()


######################
save(mu, c_prediction, 
     movie_effects, user_effects,
     k, edx_fold_summary, lambda_optim, lambda, tuningData,
     movie_effects_r, user_effects_r,
     results, movies, all_ratings_250,
     file = "./data/output.rda"
     )
