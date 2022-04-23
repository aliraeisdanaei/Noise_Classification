### Noise Classification

library(jpeg)
library(fields)
library(spatstat)
library(imager)

library(e1071)
library(class)

photo_meta <- read.csv("/home/aliraeis/4./MATH_3333/Final_Project/photoMetaData.csv")
img_dir <- "/home/aliraeis/4./MATH_3333/Final_Project/columbiaImages/"
num_rows <- nrow(photo_meta)
num_channels <- 3
num_predictors <- 6

# lets build the data matrix
X <- array(0, dim=c(num_rows, num_predictors))

for (k in 1:num_rows){
  img <- load.image(paste0(img_dir, photo_meta$name[k]))
  img.blur1 <- isoblur(img, 4)
  img.blur2 <- isoblur(img, 6)
  img.blur3 <- isoblur(img, 8)
  
  # lets calculate the image derivatives
  # img.x <- array(0, dim=c(nrow(img), ncol(img), num_channels))
  # img.y <- array(0, dim=c(nrow(img), ncol(img), num_channels))
  # img.blur.x <- array(0, dim=c(nrow(img), ncol(img), num_channels))
  # img.blur.y <- array(0, dim=c(nrow(img), ncol(img), num_channels))
  
  img.gradient <- imgradient(img)
  img.blur1.gradient <- imgradient(img.blur1)
  img.blur2.gradient <- imgradient(img.blur2)
  img.blur3.gradient <- imgradient(img.blur3)
  
  # calculate energy and noise
  img.energy <- img.gradient$x ^ 2 + img.gradient$y ^ 2
  img.blur1.energy <- img.blur1.gradient$x ^ 2 + img.blur1.gradient$y ^ 2
  img.blur2.energy <- img.blur2.gradient$x ^ 2 + img.blur2.gradient$y ^ 2
  img.blur3.energy <- img.blur3.gradient$x ^ 2 + img.blur3.gradient$y ^ 2
  
  img.noise1 <- img.energy - img.blur1.energy
  img.noise2 <- img.energy - img.blur2.energy
  img.noise3 <- img.energy - img.blur3.energy
  
  
  #X[k, 1] <- mean(img.energy)
  #X[k, 2] <- mean(img.blur.energy)
  #X[k, 3] <- mean(img.noise)
  
  X[k, 1] <- mean(img.noise1)
  X[k, 2] <- mean(img.noise2)
  X[k, 3] <- mean(img.noise3)
  X[k, 4] <- mean(img)
}

X[, 5] <- as.factor(photo_meta$camera)
X[, 6] <- as.factor(photo_meta$location)

# X[, 4] <- array(0, dim=length(num_rows))
# X[, 5] <- array(0, dim=length(num_rows))

# lets now scale the design matrix
X_unscaled <- X
X <- scale(X)

# lets build the labels
# labels <- photo_meta$category
 labels <- as.numeric(photo_meta$category == 'outdoor-day')
labels <- as.numeric(grepl('outdoor*', photo_meta$category))

labels <- as.factor(labels)

num_trials <- 500
num_k_tries <- 9

best_k <- 0
best_misclass <- 1

for (k in 1:num_k_tries){
  misclassification_ave <- 0
  for (i in 1:num_trials){
    
    rand_indexes <- sample(num_rows, size=num_rows)
    
    train_x <- X[rand_indexes[1: 600], ]
    test_x <- X[rand_indexes[601: num_rows],]
    train_labels <- labels[rand_indexes[1: 600]]
    test_labels <- labels[rand_indexes[601: num_rows]]
    
    # now for the training 
    classifier <- knn(train=train_x, test=test_x, cl=train_labels, k=k)
    #classifier
    
    # for the testing
    confusion_matrix <- table(test_labels, classifier)
    #confusion_matrix
    
    misclassification <- mean(classifier != test_labels)
    #misclassification
    
    misclassification_ave <- misclassification_ave + misclassification
  }
  misclassification_ave <- misclassification_ave / num_trials
  if(misclassification_ave < best_misclass){
    best_misclass <- misclassification_ave
    best_k <- k
  }
}

print('best k')
print(best_k)
print('best misclassification error: ')
print(best_misclass)


## ROC curve (see lecture 12)
roc <- function(y, pred) {
  alpha <- quantile(pred, seq(0,1,by=0.01))
  N <- length(alpha)
  
  sens <- rep(NA,N)
  spec <- rep(NA,N)
  for (i in 1:N) {
    predClass <- as.numeric(pred >= alpha[i])
    sens[i] <- sum(predClass == 1 & y == 1) / sum(y == 1)
    spec[i] <- sum(predClass == 0 & y == 0) / sum(y == 0)
  }
  return(list(fpr=1- spec, tpr=sens))
}


rand_indexes <- sample(num_rows, size=num_rows)

train_x <- X[rand_indexes[1: 600], ]
test_x <- X[rand_indexes[601: num_rows],]
train_labels <- labels[rand_indexes[1: 600]]
test_labels <- labels[rand_indexes[601: num_rows]]

# now for the training 
classifier <- knn(train=train_x, test=test_x, cl=train_labels, k=best_k)

# the value is 1 iff the index was not trained
trained_indexes <- array(1, num_rows)
trained_indexes[rand_indexes[1:600]] = 0


r <- roc(test_labels, classifier)
plot(r$fpr, r$tpr, xlab="false positive rate", ylab="true positive rate", type="l")
abline(0,1,lty="dashed")