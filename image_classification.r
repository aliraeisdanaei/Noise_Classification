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
rand_indexes <- sample(num_rows, size=num_rows)
num_channels <- 3
num_predictors <- 4

# lets build the data matrix
X <- array(0, dim=c(num_rows, num_predictors))

for (k in 1:num_rows){
  img <- load.image(paste0(img_dir, photo_meta$name[k]))
  img.blur <- isoblur(img, 6)
  
  # lets calculate the image derivatives
  img.x <- array(0, dim=c(nrow(img), ncol(img), num_channels))
  img.y <- array(0, dim=c(nrow(img), ncol(img), num_channels))
  img.blur.x <- array(0, dim=c(nrow(img), ncol(img), num_channels))
  img.blur.y <- array(0, dim=c(nrow(img), ncol(img), num_channels))
  
  img.gradient <- imgradient(img)
  img.blur.gradient <- imgradient(img.blur)
  
  # calculate energy and noise
  img.energy <- img.gradient$x ^ 2 + img.gradient$y ^ 2
  img.blur.energy <- img.blur.gradient$x ^ 2 + img.blur.gradient$y ^ 2
  img.noise <- img.energy - img.blur.energy
  
  X[k, 1] <- mean(img.energy)
  X[k, 2] <- mean(img.blur.energy)
  X[k, 3] <- mean(img.noise)
}

# lets build the labels
labels <- photo_meta$category
labels <- as.numeric(photo_meta$category == 'outdoor-day')
labels <- as.factor(labels)

train_x <- X[rand_indexes[1: 600], ]
test_x <- X[rand_indexes[601: num_rows],]
train_labels <- labels[rand_indexes[1: 600]]
test_labels <- labels[rand_indexes[601: num_rows]]

# now for the training 
classifier <- knn(train=train_x, test=test_x, cl=train_labels, k=5)
classifier

# for the testing
confusion_matrix <- table(test_labels, classifier)
confusion_matrix

misclassification <- mean(classifier != test_labels)
misclassification
