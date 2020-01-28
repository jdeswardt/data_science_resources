################################################################################################################################################################

#GJ DE SWARDT
#RECOMMENDATION SYSTEMS

################################################################################################################################################################
#1.) INTRODUCTION AND SETUP

##Introduce recommendation systems in the form of collaborative filtering:
##User-Based collaborative filering
##Item-Based collaborative filtering
##Collaborative filtering using Matrix Factorization

##This work lends heavily on the following resources:
##Chapter 22 of Joel Grus' ["Data Science from Scratch: First Principles with Python"](http://shop.oreilly.com/product/0636920033400.do)
##Python code from is [here](https://github.com/joelgrus/data-science-from-scratch)
##Part of [Lesson 4](http://course.fast.ai/lessons/lesson4.html) of the fast.ai course "Practical Deep Learning for Coders"
##Python code is [here](https://github.com/fastai/courses/tree/master/deeplearning1)

##Data:
##The data used for this tutorial is a small subset obtained from the Movielens data set
##The idea is therefore to build recommendation systems from using Movies and Movies viewers
##The viewers will be referred to as users and the movies will be referred to as items interchangeably

##Load required packages
library(tidyverse)

##Load dataset and view data
load("/Users/jdeswardt/Documents/GitHub/data_science_repository/data/recommendation_systems.RData")
View(ratings_red)
View(viewed_movies)

################################################################################################################################################################
#2.) USER-BASED COLLABORATIVE FILTERING
#2.1) USER-BASED SIMILARITY

##The most basic form of a recommendation system, is one where the most popular item is recommended to all users
sort(apply(viewed_movies, 2, sum), decreasing=TRUE)

##With this approach everyone gets the same recommendation, after filtering out movies which that user has seen already
##In this case each users vote counts the same
##User-based collaborative filtering extends this approach by changing how much each person's vote counts
##The system upweights the votes of people that are most similar to me. In this context similar means has seen many of the same movies as me
##There are various kinds of similarity measures, one of the most popular is cosine similarity, which we will make use of
                                                                                 
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

##Check user-similarities matrix
View(user_similarities)

##Check which users are the most similar to "User 149"
sort(user_similarities["149",], decreasing=TRUE)
                                                                                 
##The most similar is "User 303"
##The most dissimilar is "User 240"
##Lets check if this makes sense according to viewed movies
viewed_movies[c("149","303","240"),]

#2.2) USER-BASED RECOMMENDATION                                                                       

##Recommend a item to a single user (User 149)
##First check which movies "User 149" has already viewed
viewed_movies["149",]

##From the list of viewed movies for "User 149" it is clear that they haven't seen "Apocalypse Now".
##Lets see who otherwise has seen "Apocalypse Now" and their overall similarity score with "User 149"
##In order to recommend a movie we look at two parts, who else has seen that movie and what is their simlarity score
seen_movie <- viewed_movies[,"Apocalypse Now (1979)"]
sim_to_user <- user_similarities["149",]
cbind(seen_movie, sim_to_user)

##From the output the idea is that "User 236" vote counts less than "User 270" because "User 270" is more similar to our target "User 149"
##Now to recommend a movie we need to add the number of users that have seen each movie, and weight each user by their similarity to "User 149"
##To calculate a recommendation score for "Apocalypse Now" for "User 149" we multiply together each row in table above and sum these products (Dot product)
crossprod(viewed_movies[, "Apocalypse Now (1979)"], user_similarities["149",])

##Lets use the similarity matrix we created
##Calculate the recommendation scores for "User 149" for all movies
user_similarities["149",] %*% viewed_movies

##For a final recommendation, remove movies that the "User 149" has already seen and sort the recommendation scores in descending order
user_scores <- data.frame(title=colnames(viewed_movies), 
                          score=as.vector(user_similarities["149",] %*% viewed_movies), 
                          seen=viewed_movies["149",])

user_scores <- user_scores %>% 
               filter(seen == 0) %>% 
               arrange(desc(score))

View(user_scores)

#2.3) FUNCTION TO CREATE USER-BASED RECOMMENDATIONS FOR ANY USER

##Function to generate User-based Collaborative Filtering recommendations for any user
user_based_recommendations <- function(user, user_similarities, viewed_movies){
                                
  user <- ifelse(is.character(user), user, as.character(user))
                                
  user_scores <- data.frame(title=colnames(viewed_movies), 
                            score=as.vector(user_similarities[user,] %*% viewed_movies), 
                            seen=viewed_movies[user,])
                                
  user_scores %>% filter(seen == 0) %>% 
                  arrange(desc(score)) %>% 
                  select(-seen)
}

##Create recommendations using function for "User 149"
user_based_recommendation_149 <- user_based_recommendations(user=149, user_similarities=user_similarities, viewed_movies=viewed_movies)
View(user_based_recommendation_149)

##Use lapply to create recommendations for all users (NOT WORKING)
lapply(sorted_my_users, user_based_recommendations, user_similarities, viewed_movies)

################################################################################################################################################################
#3.) ITEM-BASED COLLABORATIVE FILTERING
#3.1) ITEM-BASED SIMILARITY

##Item-based collaborative filtering works similarly to User-based collaborative filtering
##Instead of using the similarity between users to upweight recommendation we now use the similarities between items

##Two important points to assist this thinking for item-based collaborative filtering:
##Two movies are similar to one another, if many of the same users have seen the movie
##When deciding what movie to recommend to a particular user, movies are evaluated on how similar they are to movies that the user has already seen
##Grocery items as example

##Create a item-similarity matrix using the loop above to calculate the dot product
movies_user <- t(viewed_movies)
size <- nrow(movies_user)
size2 <- size - 1
item_similarities = matrix(0, nrow=size, ncol=size)

for (i in 1:size2) {
  for (j in (i + 1):size) {
    item_similarities[i,j] <- cosine_sim(viewed_movies[,i], viewed_movies[,j])
  }
}

item_similarities <- item_similarities + t(item_similarities)
diag(item_similarities) <- 0
row.names(item_similarities) <- colnames(viewed_movies)
colnames(item_similarities) <- colnames(viewed_movies)
View(item_similarities)

##Check which movies are the most similar to "Apocalypse Now"
sort(item_similarities[,"Apocalypse Now (1979)"], decreasing=TRUE)

#3.2) ITEM-BASED RECOMMENDATION

##Create recommendations for a single user "User 149"
##User 149 has seen the following movies:
viewed_movies["149",]

##Another way of doing the same thing
user_seen <- row.names(item_similarities)[viewed_movies["149",] == TRUE]

##We now implement the main idea behind item-based filtering. 
##For each movie, we find the similarities between that movie and each of the four movies user 236 has seen, and sum up those similarities. 
##The resulting sum is that movie's "recommendation score".

##We start by identifying the movies the user has seen:
##We then compute the similarities between all movies and these "seen" movies. For example, similarities for the first seen movie, *Taxi Driver* are:
user_seen[1]
item_similarities[,user_seen[1]]

##We can do the same for each of the four seen movies or, more simply, do all four at once:
item_similarities[,user_seen]

##Each movie's recommendation score is obtained by summing across columns, each column representing a seen movie:
apply(item_similarities[,user_seen], 1, sum)

##The preceding explanation hopefully makes the details of the calculations clear, but it is quite unwieldy. We can do all the calculations more neatly as:
user_scores <- data.frame(title=colnames(viewed_movies), 
                          score=apply(item_similarities[,user_seen], 1, sum),
                          seen=viewed_movies["149",])

user_scores <- user_scores %>% 
               filter(seen==0) %>% 
               arrange(desc(score))

View(user_scores)

#3.3) FUNCTION TO CREATE ITEM-BASED RECOMMENDATIONS FOR ANY USER

# a function to generate an item-based recommendation for any user
item_based_recommendations <- function(user, item_similarities, viewed_movies){
  
  user <- ifelse(is.character(user), user, as.character(user))
  
  user_seen <- row.names(item_similarities)[viewed_movies[user,]==TRUE]
  
  user_scores <- data.frame(title=row.names(item_similarities), 
                            score=apply(item_similarities[,user_seen], 1, sum),
                            seen=viewed_movies[user,])
  
  user_scores %>% filter(seen==0) %>% 
                  arrange(desc(score)) %>% 
                  select(-seen)
}

##Create recommendations using function for "User 149"
item_based_recommendation_149 <- item_based_recommendations(user=149, item_similarities=item_similarities, viewed_movies=viewed_movies)
View(item_based_recommendation_149)

##And now do it for all users with `lapply'
#lapply(sorted_my_users, item_based_recommendations, item_similarities, viewed_movies)

################################################################################################################################################################
#4.) COLLABORATIVE FILTERING USING MATRIX FACTORIZATION

##In this section we're going to look at a different way of doing collaborative filtering using matrix factorization.
##Matrix factorization is a idea from linear algebra also referred to as matrix decomposition
##This method takes a matrix and represents it as a product of other (usually two) matrices

##In this example we will use the ratings given to movies as our user-item matrix
##We will decompose the ratings matrix into the product of two matrices. 
##This is done in such a way that the known ratings are matched as closely as possible

##The key feature of matrix factorization for recommendation systems is that while the ratings matrix is incomplete (i.e. some entries are blank), the two 
##matrices the ratings matrix is decomposed into are *complete* (no blank entries). 
##This gives a straightforward way of filling in blank spaces in the original ratings matrix, as we'll see.
##We use the ratings_red dataframe and use user-movie ratings instead of one-hot encoding for watched


##Get ratings into wide format and create rownames as user ids
ratings_wide <- ratings_red %>% 
                select(userId, title, rating) %>% 
                complete(userId, title) %>% 
                spread(key=title, value=rating)

sorted_my_users <- as.character(unlist(ratings_wide[,1]))
ratings_wide <- as.matrix(ratings_wide[,-1])
row.names(ratings_wide) <- sorted_my_users
View(ratings_wide)

##We start by defining a function that will compute the sum of squared differences between the observed movie ratings and any other set of predicted ratings 
##(for example, ones predicted by matrix factorization). 
##Note that we only count movies that have already been rated in the accuracy calculation.

##Initialize variables for funtion based on the desired number of latent factors d
d <- 5
users <- nrow(ratings_wide)
items <- ncol(ratings_wide)
U_dim <- users * d
V_dim_start <- U_dim + 1
V_dim <- U_dim + (d * items)

recommendation_accuracy <- function(x, observed_ratings){
    
  # extract user and movie factors from parameter vector (note x is defined such that 
  # the first 75 elements are latent factors for users and rest are for movies)
  U <- matrix(x[1:U_dim], users, d)
  V <- matrix(x[V_dim_start:V_dim], d, items)
  
  # get predictions from dot products of respective user and movie factor
  R <- U %*% V
  
  # model accuracy is sum of squared errors over all rated movies
  errors <- (observed_ratings - R)^2 
  
  sqrt(mean(errors[!is.na(observed_ratings)]))   # only use rated movies
}

##We'll now optimize the values in the user and movie latent factors, choosing them so that the root mean square error (the square root of the average squared 
##difference between observed and predicted ratings) is a minimum. I've done this using R's inbuilt numerical optimizer `optim()`, with the default "Nelder-Mead" 
##method. There are better ways to do this - experiment! Always check whether the optimizer has converged (although you can't always trust this), see 
##`help(optim)` for details.

##Optimize
set.seed(10)
recommendation <- optim(par=runif(V_dim), recommendation_accuracy, observed_ratings=ratings_wide, control=list(maxit=100000))
recommendation$convergence
recommendation$value

##The best value of the objective function found by `optim()` after 100000 iterations is 0.258, but note that it hasn't converged yet, so we should really run 
##for longer or try another optimizer! Ignoring this for now, we can extract the optimal user and movie factors. With a bit of work, these can be interpreted 
##and often give useful information. Unfortunately we don't have time to look at this further (although it is similar to the interpretation of principal 
##components, if you are familiar with that).

##User factors
user_factors <- matrix(rec1$par[1:75], 15, 5)
View(user_factors)

##Item factors
item_factors <- matrix(rec1$par[76:175], 5, 20)
View(item_factors)

##Most importantly, we can get **predicted movie ratings** for any user, by taking the appropriate dot product of user and movie factors. Here we show the 
##predictions for user 1:
# check predictions for one user
predicted_ratings <- user_factors %*% movie_factors
row.names(predicted_ratings) <- row.names(viewed_movies)
colnames(predicted_ratings) <- colnames(viewed_movies)
View(predicted_ratings)

##Compare predicted rating with actual ratings for "User 149"
rbind(round(predicted_ratings["149",], 1), as.numeric(ratings_wide["149",]))

##To recommend a movie to a user is intuitive from this step onwards
##We now recommend movies based on the highest ranking movies the user has not seen
##We can examine and interpret the user or movie latent factors, or bias terms, if we want to
##Create a function to generate matrix factorization recommendations for any user
mf_recommendations <- function(user, predicted_ratings, viewed_movies){
  
  user <- ifelse(is.character(user), user, as.character(user))
  
  user_scores <- data.frame(title=colnames(viewed_movies), 
                            score=round(predicted_ratings[user,], 1),
                            seen=viewed_movies[user,])
  
  user_scores %>% filter(seen==0) %>% 
                  arrange(desc(score)) %>% 
                  select(-seen)
}

##Create recommendations using function for "User 149"
mf_recommendation_149 <- mf_recommendations(user=149, predicted_ratings=predicted_ratings, viewed_movies=viewed_movies)
View(mf_recommendation_149)

################################################################################################################################################################
#5.) NOTES

##1. Adapt the pairwise similarity function so that it doesn't use loops.
##2. Implement a k-nearest-neighbours version of item-based collaborative filtering.
##3. Experiment with the optimizers used in the matrix factorization collaborative filter.

################################################################################################################################################################