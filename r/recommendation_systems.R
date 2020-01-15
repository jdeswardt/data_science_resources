################################################################################################################################################################

#GJ DE SWARDT
#RECOMMENDATION SYSTEMS

################################################################################################################################################################
#1.) INTRODUCTION AND SETUP

##Introduce recommendation systems in the form of collaborative filtering:
###User-Based collaborative filering
###Item-Based collaborative filtering
###Collaborative filtering using Matrix Factorization

##This work lends heavily on the following resources:
###Chapter 22 of Joel Grus' ["Data Science from Scratch: First Principles with Python"](http://shop.oreilly.com/product/0636920033400.do)
###Python code from is [here](https://github.com/joelgrus/data-science-from-scratch)
###Part of [Lesson 4](http://course.fast.ai/lessons/lesson4.html) of the fast.ai course "Practical Deep Learning for Coders"
###Python code is [here](https://github.com/fastai/courses/tree/master/deeplearning1)

##Data
###The data used for this tutorial is a small subset obtained from the Movielens data set
###The idea is therefore to build recommendation systems from using Movies and Movies watchers
###The watchers will be referred to as users and the movies will be referred to as items

##Load required packages
library(tidyverse)

##Load dataset and view data
load("/Users/jdeswardt/Documents/GitHub/data_science_repository/data/recommendation_systems.RData")
View(ratings_red)
View(viewed_movies)

################################################################################################################################################################
#2.) USER-BASED COLLABORATIVE FILTERING
#2.1) UNDERSTANDING SIMILARITY
##The most basic form of a recommendations system, is one where the most popular item is recommended to all users:
sort(apply(viewed_movies, 2, sum), decreasing=TRUE)

##With this approach eeveryone gets the same recommendation, after filtering out movies which that user has seen already.
##In this case each users vote counts the same. 
##User-based collaborative filtering extends the approach by changing how much each person's vote counts. 
##The system upweights the votes of people that are most similar to me. In this context similar means has seen many of the same movies as me. 
##There are various kinds of similarity measures, one of the most popular is cosine similarity, which we will make use of.
                                                                                 
##Function calculating cosine similarity (Dot product)
cosine_sim <- function(a, b){crossprod(a, b) / sqrt(crossprod(a) * crossprod(b))}

##The cosine similarity measure lies between zero and one, the more similar the higher the value.
##Maximally similar
x1 <- c(1, 1, 1, 0, 0)
x2 <- c(1, 1, 1, 0, 0)
cosine_sim(x1, x2)

##Maximally dissimilar
x1 <- c(1, 1, 1, 0, 0)
x2 <- c(0, 0, 0, 1, 1)
cosine_sim(x1, x2)

##Also
x1 <- c(1, 1, 0, 0, 0)
x2 <- c(0, 0, 0, 1, 1)
cosine_sim(x1, x2)
                                                                                 
##Calculate the cosine similarity between user 1 and user 2 from the data
as.numeric(viewed_movies[1,])
as.numeric(viewed_movies[2,])
cosine_sim(viewed_movies[1,], viewed_movies[2,])
                                                                                 
##Create a similarity matrix using a loop to calculate similarity scores between all users
size <- nrow(viewed_movies)
size2 <- size - 1

user_similarities <- matrix(0, nrow=size, ncol=size)
for (i in 1:size2) {
  for (j in (i + 1):size) {
    user_similarities[i, j] <- cosine_sim(viewed_movies[i,], viewed_movies[j,])
 }
}
user_similarities <- user_similarities + t(user_similarities)
diag(user_similarities) <- 0
row.names(user_similarities) <- row.names(viewed_movies)
colnames(user_similarities) <- row.names(viewed_movies)

##Check who are the most similar to "User 149"
user_similarities["149",]
                                                                                 
##The most similar is "User 303"
##The most dissimilar is "User 236"
##Lets check if this makes sense according to viewed movies
viewed_movies[c("149","303","236"),]

#2.2) UNDERSTANDING RECOMMENDATION                                                                       

##Recommend a item to a single user (User 149)
##First check which movies User 149 has already viewed
viewed_movies["149",]

##From the list of viewed movies for "User 149" it is clear that they haven't seen "Apocalypse Now".
##Lets see who otherwise has seen "Apocalypse Now" and their overall similarity score with "User 149"
##In order to recommend a movie we look at two parts, who else has seen that movie and what is their simlarity score
seen_movie <- viewed_movies[,"Apocalypse Now (1979)"]
sim_to_user <- user_similarities["149",]
cbind(seen_movie, sim_to_user)

##From the output above the idea is that "User 236" vote counts less than "User 408" because "User 408" is more similar to our target "User 149"
##Now to recommend a movie we need to add the number of users that have seen each movie, and weight each user by their similarity to "User 149"
##To calculate a recommendation score for "Apocalypse Now" for "User 149" we multiply together each row in table above and sum these products (Dot product)
crossprod(viewed_movies[, "Apocalypse Now (1979)"], user_similarities["149",])

##Calculate the recommendation scores for "User 149" for all movies
user_similarities["149",] %*% viewed_movies

##For a final recommendation, remove movies that the "User 149" has already seen and sort the recommendation scores in descending order
user_scores <- data.frame(title=colnames(viewed_movies), 
                          score=as.vector(user_similarities["149",] %*% viewed_movies), 
                          seen=viewed_movies["149",])
user_scores <- user_scores %>% 
               filter(seen == 0) %>% 
               arrange(desc(score)) 

#2.3) FUNCTION TO CREATE RECOMMENDATIONS FOR ANY USER

##Function to generate User-based Collaborative Filtering recommendations for any user
user_based_recommendations <- function(user, user_similarities, viewed_movies){
                                
  user <- ifelse(is.character(user), user, as.character(user))
                                
  user_scores <- data.frame(title = colnames(viewed_movies), 
                            score = as.vector(user_similarities[user,] %*% viewed_movies), 
                            seen = viewed_movies[user,])
                                
  user_scores %>% filter(seen == 0) %>% 
                  arrange(desc(score)) %>% 
                  select(-seen)
}

##Create recommendations using function for "User 149"
recommendation_user_149 <- user_based_recommendations(user=149, user_similarities=user_similarities, viewed_movies=viewed_movies)
View(recommendation_user_149)

##A variant on the above is a *k-nearest-neighbours* approach that bases recommendations *only on k most similar users*. This is faster when there are many users. 
##Try to implement this as an exercise.

################################################################################################################################################################
#3.) ITEM-BASED COLLABORATIVE FILTERING
#3.1) UNDERSTANDING SIMILARITY

##Item-based collaborative filtering works very similarly to its user-based counterpart, but is a tiny bit less intuitive (in my opinion). It is also based on 
##similarities, but similarities between *movies* rather than *users*.

##There are two main conceptual parts to item-based collaborative filtering:

##1. One movie is similar to another if many of the same users have seen both movies.
##2. When deciding what movie to recommend to a particular user, movies are evaluated on how similar they are to movies *that the user has already seen*.

##Let's start by computing the similarities between all pairs of movies. We can reuse the same code we used to compute user similarities, if we first transpose 
##the *viewed_movies* matrix.

# transpose the viewed_movies matrix
movies_user <- t(viewed_movies)

# get all similarities between MOVIES
movie_similarities = matrix(0, nrow=20, ncol=20)
for (i in 1:19) {
  for (j in (i + 1):20) {
    movie_similarities[i,j] <- cosine_sim(viewed_movies[,i], viewed_movies[,j])
  }
}
movie_similarities <- movie_similarities + t(movie_similarities)
diag(movie_similarities) <- 0
row.names(movie_similarities) <- colnames(viewed_movies)
colnames(movie_similarities) <- colnames(viewed_movies)

##We can use the result to see, for example, what movies are most similar to "Apocalypse Now":
sort(movie_similarities[,"Apocalypse Now (1979)"], decreasing=TRUE)

### Recommending movies for a single user

##Let's again look at a concrete example of recommending a movie to a particular user, say user 236.

##User 236 has seen the following movies:
which(viewed_movies["236", ] == 1)

##Another way of doing the same thing:
ratings_red %>% 
  filter(userId == 236) %>% 
  select(userId, title)

##We now implement the main idea behind item-based filtering. For each movie, we find the similarities between that movie and each of the four movies user 236 
##has seen, and sum up those similarities. The resulting sum is that movie's "recommendation score".

##We start by identifying the movies the user has seen:
user_seen <- ratings_red %>% 
  filter(userId == 236) %>% 
  select(title) %>% 
  unlist() %>% 
  as.character()

##We then compute the similarities between all movies and these "seen" movies. For example, similarities for the first seen movie, *Taxi Driver* are:
user_seen[1]
movie_similarities[,user_seen[1]]

##We can do the same for each of the four seen movies or, more simply, do all four at once:
movie_similarities[,user_seen]

##Each movie's recommendation score is obtained by summing across columns, each column representing a seen movie:
apply(movie_similarities[,user_seen],1,sum)


##The preceding explanation hopefully makes the details of the calculations clear, but it is quite unwieldy. We can do all the calculations more neatly as:
user_scores <- tibble(title = row.names(movie_similarities), 
                      score = apply(movie_similarities[,user_seen], 1, sum),
                      seen = viewed_movies["236",])

user_scores %>% 
  filter(seen == 0) %>% 
  arrange(desc(score))

##So we'd end up recommending "Minority Report" to this particular user.

##Let's repeat the process to generate a recommendation for one more user, user 149:

# do for user 149
user <- "149"
user_seen <- ratings_red %>% 
  filter(userId == user) %>% 
  select(title) %>% 
  unlist() %>% 
  as.character()

user_scores <- tibble(title = row.names(movie_similarities), 
                      score = apply(movie_similarities[,user_seen],1,sum),
                      seen = viewed_movies[user,])

user_scores %>% 
  filter(seen == 0) %>% 
  arrange(desc(score))

### A simple function to generate an item-based CF recommendation for any user

# a function to generate an item-based recommendation for any user
item_based_recommendations <- function(user, movie_similarities, viewed_movies){
  
  # turn into character if not already
  user <- ifelse(is.character(user), user, as.character(user))
  
  # get scores
  user_seen <- row.names(movie_similarities)[viewed_movies[user,] == TRUE]
  user_scores <- tibble(title = row.names(movie_similarities), 
                        score = apply(movie_similarities[,user_seen], 1, sum),
                        seen = viewed_movies[user,])
  
  # sort unseen movies by score and remove the 'seen' column
  user_scores %>% 
    filter(seen == 0) %>% 
    arrange(desc(score)) %>% 
    select(-seen)
  
}

##Let's check that its working with a user we've seen before, user 236:
item_based_recommendations(user = 236, movie_similarities = movie_similarities, viewed_movies = viewed_movies)

##And now do it for all users with `lapply'
#lapply(sorted_my_users, item_based_recommendations, movie_similarities, viewed_movies)

################################################################################################################################################################
#4.) COLLABORATIVE FILTERING USING MATRIX FACTORIZATION

##In this section we're going to look at a different way of doing collaborative filtering, one based on the idea of *matrix factorization*, a topic from linear 
##algebra.

##Matrix factorization, also called matrix decomposition, takes a matrix and represents it as a product of other (usually two) matrices. There are many ways to 
##do matrix factorization, and different problems tend to use different methods.

##In recommendation systems, matrix factorization is used to decompose the ratings matrix into the product of two matrices. This is done in such a way that the 
##known ratings are matched as closely as possible. 

##The key feature of matrix factorization for recommendation systems is that while the ratings matrix is incomplete (i.e. some entries are blank), the two 
##matrices the ratings matrix is decomposed into are *complete* (no blank entries). This gives a straightforward way of filling in blank spaces in the original 
##ratings matrix, as we'll see.

# get ratings in wide format
ratings_wide <- ratings_red %>% 
  select(userId, title, rating) %>% 
  complete(userId, title) %>% 
  spread(key=title, value=rating)

# convert data to matrix form 
sorted_my_users <- as.character(unlist(ratings_wide[,1]))
ratings_wide <- as.matrix(ratings_wide[,-1])
row.names(ratings_wide) <- sorted_my_users

# save as csv for Excel demo
write.csv(ratings_wide,"output/ratings_for_excel_example.csv")

##Now let's set up the same computations in R, which will be faster and easier to generalise beyond a particular size dataset. We start by defining a function 
##that will compute the sum of squared differences between the observed movie ratings and any other set of predicted ratings (for example, ones predicted by 
##matrix factorization). Note that we only count movies that have already been rated in the accuracy calculation.

recommender_accuracy <- function(x, observed_ratings){
    
  # extract user and movie factors from parameter vector (note x is defined such that 
  # the first 75 elements are latent factors for users and rest are for movies)
  user_factors <- matrix(x[1:75], 15, 5)
  movie_factors <- matrix(x[76:175], 5, 20)
  
  # get predictions from dot products of respective user and movie factor
  predicted_ratings <- user_factors %*% movie_factors
  
  # model accuracy is sum of squared errors over all rated movies
  errors <- (observed_ratings - predicted_ratings) ^ 2 
  
  sqrt(mean(errors[!is.na(observed_ratings)]))   # only use rated movies
}

##**Exercise**: This function isn't general, because it refers specifically to a ratings matrix with 15 users, 20 movies, and 5 latent factors. Make the 
##function general.

##We'll now optimize the values in the user and movie latent factors, choosing them so that the root mean square error (the square root of the average squared 
##difference between observed and predicted ratings) is a minimum. I've done this using R's inbuilt numerical optimizer `optim()`, with the default "Nelder-Mead" 
##method. There are better ways to do this - experiment! Always check whether the optimizer has converged (although you can't always trust this), see 
##`help(optim)` for details.

set.seed(10)
# optimization step
rec1 <- optim(par = runif(175), recommender_accuracy, 
              observed_ratings = ratings_wide, control = list(maxit = 100000))
rec1$convergence
rec1$value

##The best value of the objective function found by `optim()` after 100000 iterations is 0.258, but note that it hasn't converged yet, so we should really run 
##for longer or try another optimizer! Ignoring this for now, we can extract the optimal user and movie factors. With a bit of work, these can be interpreted 
##and often give useful information. Unfortunately we don't have time to look at this further (although it is similar to the interpretation of principal 
##components, if you are familiar with that).

# extract optimal user factors
user_factors <- matrix(rec1$par[1:75], 15, 5)
head(user_factors)

# extract optimal movie factors
movie_factors <- matrix(rec1$par[76:175], 5, 20)
head(movie_factors)

##Most importantly, we can get **predicted movie ratings** for any user, by taking the appropriate dot product of user and movie factors. Here we show the 
##predictions for user 1:

# check predictions for one user
predicted_ratings <- user_factors %*% movie_factors
rbind(round(predicted_ratings[1,], 1), as.numeric(ratings_wide[1,]))

### Adding L2 regularization
##One trick that can improve the performance of matrix factorization collaborative filtering is to add L2 regularization. L2 regularization adds a penalty term 
##to the function that we're trying to minimize, equal to the sum of the L2 norms over all user and movie factors. This penalizes large parameter values. 

##We first rewrite the *evaluate_fit* function to make use of L2 regularization:

## adds L2 regularization, often improves accuracy
evaluate_fit_l2 <- function(x, observed_ratings, lambda){
  
  # extract user and movie factors from parameter vector
  user_factors <- matrix(x[1:75], 15, 5)
  movie_factors <- matrix(x[76:175], 5, 20)
  
  # get predictions from dot products
  predicted_ratings <- user_factors %*% movie_factors
  
  errors <- (observed_ratings - predicted_ratings) ^ 2 
  
  # L2 norm penalizes large parameter values
  penalty <- sum(sqrt(apply(user_factors ^ 2, 1, sum))) + 
    sum(sqrt(apply(movie_factors ^ 2, 2, sum)))
  
  # model accuracy contains an error term and a weighted penalty 
  accuracy <- sqrt(mean(errors[!is.na(observed_ratings)])) + lambda * penalty
  
  return(accuracy)
}

##We now rerun the optimization with this new evaluation function:
set.seed(10)
# optimization step
rec2 <- optim(par = runif(175), evaluate_fit_l2, 
            lambda = 3e-3, observed_ratings = ratings_wide, control = list(maxit = 100000))
rec2$convergence
rec2$value


##The best value found is **worse** than before, but remember that we changed the objective function to include the L2 penalty term, so the numbers are not 
##comparable. We need to extract just the RMSE that we're interested in. To do that we first need to extract the optimal parameter values (user and 
##movie factors), and multiply these matrices together to get predicted ratings. From there, its easy to calculate the errors.

# extract optimal user and movie factors
user_factors <- matrix(rec2$par[1:75], 15, 5)
movie_factors <- matrix(rec2$par[76:175], 5, 20)

# get predicted ratings
predicted_ratings <- user_factors %*% movie_factors

# check accuracy
errors <- (ratings_wide - predicted_ratings) ^ 2 
sqrt(mean(errors[!is.na(ratings_wide)]))

##Compare this with what we achieved without L2 regularization: did it work? As before, we can extract user and movie factors, and get predictions for any user.

# check predictions for one user
rbind(round(predicted_ratings[1,],1), as.numeric(ratings_wide[1,]))

### Adding bias terms

##We've already seen bias terms in the Excel example. Bias terms are additive factors that model the fact that some users are more generous than others 
##(and so will give higher ratings, on average) and some movies are better than others (and so will get higher ratings, on average). 

##Let's adapt our evaluation function further to include bias terms for both users and movies:

## add an additive bias term for each user and movie

evaluate_fit_l2_bias <- function(x, observed_ratings, lambda){
  # extract user and movie factors and bias terms from parameter vector
  user_factors <- matrix(x[1:75], 15, 5)
  movie_factors <- matrix(x[76:175], 5, 20)
  # the bias vectors are repeated to make the later matrix calculations easier 
  user_bias <- matrix(x[176:190],nrow = 15, ncol = 20)
  movie_bias <- t(matrix(x[191:210], nrow = 20, ncol = 15))
  
  # get predictions from dot products + bias terms
  predicted_ratings <- user_factors %*% movie_factors + user_bias + movie_bias
  
  errors <- (observed_ratings - predicted_ratings) ^ 2 
  
  # L2 norm penalizes large parameter values (note not applied to bias terms)
  penalty <- sum(sqrt(apply(user_factors ^ 2, 1, sum))) + 
    sum(sqrt(apply(movie_factors ^ 2, 2, sum)))
  
  # model accuracy contains an error term and a weighted penalty 
  sqrt(mean(errors[!is.na(observed_ratings)])) + lambda * penalty
}

##Again, rerun the optimization:

set.seed(10)
# optimization step (note longer parameter vector to include bias)
rec3 <- optim(par = runif(220), evaluate_fit_l2_bias,
              observed_ratings = ratings_wide, lambda = 3e-3, control = list(maxit = 100000))
rec3$convergence
rec3$value

##This value isn't comparable to either of the previous values, for the same reason as before: the objective function has changed to include bias terms. 
##Extracting just the RMSE:

# extract optimal user and movie factors and bias terms
user_factors <- matrix(rec3$par[1:75], 15, 5)
movie_factors <- matrix(rec3$par[76:175], 5, 20)
user_bias <- matrix(rec3$par[176:190], nrow = 15, ncol = 20)
movie_bias <- t(matrix(rec3$par[191:210], nrow = 20, ncol = 15))

# get predicted ratings
predicted_ratings <- user_factors %*% movie_factors + user_bias + movie_bias

# check accuracy
errors <- (ratings_wide - predicted_ratings) ^ 2 
sqrt(mean(errors[!is.na(ratings_wide)]))

##This is indeed an improvement over what we've seen before (at least, for the parameter settings above!). 

##We can examine and interpret the user or movie latent factors, or bias terms, if we want to. Below we show the movie bias terms, which give a reasonable 
##reflection of movie quality (with some notable exceptions!)
data.frame(movies = colnames(viewed_movies), bias = movie_bias[1,]) %>% arrange(desc(bias))

##Finally, we again get predicted ratings for one user:

# check predictions for one user
rbind(round(predicted_ratings[1,], 1), as.numeric(ratings_wide[1,]))

################################################################################################################################################################
#5.) NOTES

##1. Adapt the pairwise similarity function so that it doesn't use loops.
##2. Implement a k-nearest-neighbours version of item-based collaborative filtering.
##3. Adapt the `recommender_accuracy()` function so that it can be used with an arbitrary number of users and movies.
##4. Experiment with the optimizers used in the matrix factorization collaborative filter.

################################################################################################################################################################