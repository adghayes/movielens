################################
# Model Data + Calculate RMSEs #
################################
library(tidyverse)
library(caret)
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

muf_prediction <- muf_predict(validation)
muf_RMSE <- RMSE(validation$rating, muf_prediction)

##########################################################
# MOVIE + USER EFFECTS APPROACH w/ REGULARIZATION (mufr) #
##########################################################
calibrate_lambda <- FALSE
lambda <- c(3.87, 4.97)

if(calibrate_lambda){
  mufr_predict_pars <- function(train, test, lambda_m, lambda_u){
    movie_effects_r_temp <- train %>% 
      group_by(movieId) %>%
      summarize(
        m_i = sum(rating - mu)/(n() + lambda_m),
        n_i = n()
      )
    
    user_effects_r_temp <- train %>% 
      left_join(movie_effects_r_temp, by = "movieId") %>%
      group_by(userId) %>%
      summarize(
        u_j = sum(rating - mu - m_i)/(n() + lambda_u),
        n_j = n()
      )
    
    test %>% 
      left_join(movie_effects_r_temp, by = "movieId") %>%
      left_join(user_effects_r_temp, by = "userId") %>%
      mutate(rating = mu + m_i + u_j) %>% pull(rating)
  }
  
  # Determining lambda(s) requires cross-validation #
  k <- 5
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
  cross_val_rmse <- function(lambda){
    mean(sapply(edx_folds, function(fold){
      RMSE(mufr_predict_pars(edx[-fold,],edx[fold,],lambda[1],lambda[2]), edx[fold,]$rating)
    }))
  }
  
  lambda_optim <- optim(par = lambda, fn = cross_val_rmse, control = list(maxit = 2))
  lambda <- lambda_optim$par
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

###################
# PCA             #
###################
pca_n_movies <- 50
pca_lambda_movies <- 1000
pca_n_users <- 10000
pca_n_centers <- 27
lambda_c <- 25

pca_movies <- edx %>% 
  group_by(movieId) %>%
  summarize(n = n(), sd = sd(rating), sdr = sd*n/(n + pca_lambda_movies)) %>%
  top_n(pca_n_movies, wt = sdr) %>%
  pull(movieId)

pca_users <- edx %>% 
  filter(movieId %in% pca_movies) %>%
  group_by(userId) %>%
  tally() %>%
  arrange(n) %>%
  rowid_to_column() %>%
  # top_n(pca_n_users, wt = rowid) %>%
  pull(userId)

q <- edx %>% 
  left_join(movie_effects_r, by = "movieId") %>%
  left_join(user_effects_r, by = "userId") %>%
  mutate(rating = rating - mu - m_i - u_j) %>%
  select(userId, movieId, rating) %>%
  filter(userId %in% pca_users & movieId %in% pca_movies) %>%
  spread(movieId, rating) %>%
  as.matrix()

rownames(q)<- q[,1]
q <- q[,-1]
q[is.na(q)] <- 0
movies <- edx %>% 
  distinct(movieId, title, genres)
colnames(q) <- with(movies, title[match(colnames(q), movieId)])

# q <- sweep(q, 2, colMeans(q, na.rm=TRUE))
# q <- sweep(q, 1, rowMeans(q, na.rm=TRUE))

pca <- prcomp(q)
pc <- 1:nrow(pca$rotation)
var_explained <- cumsum(pca$sdev^2 / sum(pca$sdev^2))
qplot(pc, var_explained)



# pca$x has PC's for columms and users for rows
# pca$rotation has PC's for columns and movies for rows
# Estimating PC's for ALL users
users_pc <- edx %>% 
  left_join(movie_effects_r, by = "movieId") %>%
  left_join(user_effects_r, by = "userId") %>%
  mutate(rating = rating - mu - m_i - u_j) %>%
  filter(title %in% colnames(q)) %>%
  group_by(userId) %>%
  filter(n() > 1) %>%
  group_modify(~ {
    data.frame(t(.x$rating) %*% pca$rotation[.x$title,])
  }) 

# ALTERNATE IF COMING STRAIGHT FROM PCA
users_pc <- data.frame(pca$x) %>% 
  rownames_to_column(var = "userId") %>% 
  remove_rownames() %>%
  mutate(userId = as.integer(userId))

# Must mock up data for users who haven't seen the movies included in the analysis
users_wo_pc <- edx %>% 
  distinct(userId) %>% 
  anti_join(users_pc, by = "userId")
empty_pc_cols <- matrix(
  data = rep(0, nrow(users_wo_pc)*length(pca_movies)), nrow = nrow(users_wo_pc))
colnames(empty_pc_cols) <- paste("PC",1:length(pca_movies), sep = "")
users_wo_pc <- bind_cols(users_wo_pc, data.frame(empty_pc_cols))
users_pc <- users_pc %>% union(users_wo_pc)



# Find clusters of users
clustering <- kmeans(users_pc[,-1], centers = pca_n_centers, iter.max = 25)
user_clusters <- users_pc %>% select(userId) %>% add_column(cluster = clustering$cluster)
cluster_means <- edx %>% 
  left_join(movie_effects_r, by = "movieId") %>%
  left_join(user_effects_r, by = "userId") %>%
  left_join(user_clusters, by = "userId") %>%
  mutate(rating = rating - mu - m_i - u_j) %>%
  select(movieId, rating, cluster) %>%
  group_by(movieId, cluster) %>%
  summarize(c_k = sum(rating)/(n() + lambda_c), n_k = n())

cluster_means %>% ggplot(aes(x = n_k, y = c_k)) + geom_point()

mclust_predict <- function(newdata){
  newdata %>% 
    left_join(movie_effects_r, by = "movieId") %>%
    left_join(user_effects_r, by = "userId") %>%
    left_join(user_clusters, by = "userId") %>%
    left_join(cluster_means, by = c("movieId", "cluster")) %>%
    mutate(c_k = if_else(is.na(c_k), 0, c_k)) %>%
    mutate(rating = mu + m_i + u_j + c_k) %>% pull(rating)
}

mclust_prediction <- mclust_predict(validation)
mclust_RMSE <- RMSE(validation$rating, mclust_prediction)

5, .85429
47, .85154
111, .852

