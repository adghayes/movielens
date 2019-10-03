#########################################
#########################################
## PREDICT RATINGS AND CALCULATE RMSES ##
#########################################
#########################################
library(tidyverse)
library(caret)
load("./data/movielens.rda")
results <- data.frame(Approach = character(), RMSE = double())

###################################
#   SIMPLE AVERAGE APPROACH (c)   #
###################################
mu <- edx %>% pull(rating) %>% mean()

c_predict <- function(newdata){
  rep(mu, nrow(newdata))
}

c_approach <- "Simple Average"
c_prediction <- c_predict(validation)
c_RMSE <- RMSE(validation$rating, c_prediction)
results <- add_row(results, Approach = c_approach, RMSE = c_RMSE)

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

muf_predict <- function(newdata){
  newdata %>% 
    left_join(movie_effects, by = "movieId") %>%
    left_join(user_effects, by = "userId") %>%
    mutate(rating = mu + m_i + u_j) %>%
    pull(rating)
}

muf_approach <- "Movie + User Effects"
muf_prediction <- muf_predict(validation)
muf_RMSE <- RMSE(validation$rating, muf_prediction)
results <- add_row(results, Approach = muf_approach, RMSE = muf_RMSE)

##########################################################
# MOVIE + USER EFFECTS APPROACH w/ REGULARIZATION (mufr) #
##########################################################

lambda <- c(lambda_m = 4.702620605468748493649, lambda_u = 4.964661132812500099476)
calibrate_lambda <- FALSE

if(calibrate_lambda){
  mufr_train_predict <- function(train, test, par){
    movie_effects_r_temp <- train %>% 
      group_by(movieId) %>%
      summarize(
        m_i = sum(rating - mu)/(n() + par[1]),
        n_i = n()
      )
    
    user_effects_r_temp <- train %>% 
      left_join(movie_effects_r_temp, by = "movieId") %>%
      group_by(userId) %>%
      summarize(
        u_j = sum(rating - mu - m_i)/(n() + par[2]),
        n_j = n()
      )
    
    test %>% 
      left_join(movie_effects_r_temp, by = "movieId") %>%
      left_join(user_effects_r_temp, by = "userId") %>%
      mutate(rating = mu + m_i + u_j) %>% pull(rating)
  }
  
  # Determining lambda(s) requires cross-validation #
  k <- 10
  set.seed(23, sample.kind="Rounding")
  edx_folds <- createFolds(y = edx$rating, k = k)
  
  # Removing unknown users/movies from test partitions
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
  
  # Optimizing Lambda on the average RMSEs of cross validated sets
  cross_val_rmse <- function(par){
    mean(sapply(edx_folds, function(fold){
      RMSE(mufr_train_predict(edx[-fold,],edx[fold,],par), edx[fold,]$rating)
    }))
  }
  
  # Calibrate lambda
  lambda_optim <- optim(par = lambdas, fn = cross_val_rmse, control = list(maxit = 25, trace = TRUE))
  lambda <- lambda_optim$par
  
  # Plot Lambda
  lambda_m <- lambda[1]
  lambda_u <- lambda[2]
  lambda_ms <- lambda_m + seq(-2,2,1)
  lambda_us <- lambda_u + seq(-2,2,1)
  tuningData <- expand_grid(lambda_m = lambda_ms, lambda_u = lambda_us) %>% 
    rowwise() %>%
    mutate(RMSE = cross_val_rmse(c(lambda_m,lambda_u))) %>%
    ungroup()
  
  library(ggthemes)
  tuningData %>% 
    ggplot(aes(x = lambda_m, y = lambda_u, color = RMSE)) +
    geom_point() + theme_few() + scale_color_gradient(low = "#f03a3a", high = "#030ffc")
  
  save(lambda_optim, lambda, tuningData, file = "./data/mufr")  
}



# Now with a decent lambda we can build and evaluate the tuned model
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
    mutate(rating = mu + m_i + u_j) %>% pull(rating)
}

mufr_prediction <- mufr_predict(validation)
mufr_RMSE <- RMSE(validation$rating, mufr_prediction)

########################
# Recommender Lab      #
########################
# https://cran.r-project.org/web/packages/recommenderlab/vignettes/recommenderlab.pdf
library(recommenderlab)
n_movies <- 1000
n_users <- 1000

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

########################
# SVD                  #
########################

edx_c_mini <- edx_c %>%
  filter(userId %in% svd_users & movieId %in% svd_movies) %>%
  select(userId, movieId, rating) %>% as("realRatingMatrix")

edx_c_subset <- edx_c %>%
  filter(movieId %in% svd_movies) %>%
  select(userId, movieId, rating) %>% as("realRatingMatrix")


svd_rec <- Recommender(edx_c_mini, method = "SVD", parameter = 
                         list(k = 100))

ibcf_rec <- Recommender(edx_c_mini, method = "IBCF", parameter = 
                          list(k = 100, method = "Pearson"))

rl_predict <- function(rec, newdata){
  rrm <- recommenderlab::predict(object = rec, newdata = edx_c_subset, type = "ratings")
  
  rrdf <- rrm %>% 
    as("matrix") %>% 
    as.data.frame() %>%
    rownames_to_column(var = "userId") %>%
    remove_rownames() %>%
    gather(movieId, rating, -userId, na.rm = TRUE) %>%
    mutate(
      movieId = as.integer(movieId),
      userId = as.integer(userId)
    )
  
  newdata %>% select(movieId, userId) %>%
    left_join(svd_output_df, by = c("movieId", "userId")) %>%
    left_join(movie_effects_r, by = "movieId") %>%
    left_join(user_effects_r, by = "userId") %>%
    mutate(rating = rating + mu + m_i + u_j) %>%
    pull(rating)
}

svd_prediction <- rl_predict(svd_rec, validation)
ibcf_prediction <- rl_predict(ibcf_rec, validation)

