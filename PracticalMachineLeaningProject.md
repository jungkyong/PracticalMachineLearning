# Practical machine learning - Predicting weight lift activity using tree-based methods
Jung-Kyong Kim  
February 28, 2016  

# Background

With the advent of wearable devices, it is now possible to collect a large amount of data about personal activity inexpensively and efficiently.  Regular use of these devices can help generate data that can be utilized to quantify physical activities with the ultimate goal of improving one's health.  The current project examined a weight lifting exercise dataset collected from accelerometers on the belt, forearm, arm, and dumbell of 6 individuals. Participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways, and the goal was to predict the manner in which they did the exercise.  Multiple tree-based methods were used to select the model with the best prediction accuracy.

More information about the data can be found at http://groupware.les.inf.puc-rio.br/har

# Getting and cleaning data

## Data preparation

### Downloading files

```r
train_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
test_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(train_url, destfile="train.csv")
download.file(test_url, destfile="test.csv")
```

### Loading packages and reading data


```r
packages <- c("purrr", "caret", "tree", "gbm", "randomForest")
lapply(packages, require, character.only=TRUE)

raw_train <- read.csv("train.csv", na.strings=c("NA", "#DIV/0!", ""))
raw_test <- read.csv("test.csv", na.strings=c("NA", "#DIV/0!", ""))
```

### Cleaning data

We remove all the *Near Zero Variables* as well as those containing NAs more than 50% of the time


```r
#identify Near Zero Variables

nz_names <- 
  nearZeroVar(raw_train, saveMetrics=TRUE) %>%
  (function(data){
    true <- data$nzv==TRUE
    rownames(data)[true]
    })

# Identify variables with more than 50% NAs

na_names <- 
  raw_train %>%
  map(is.na) %>%
  map(sum) %>%
  (function(data){
    names(data)[data > dim(raw_train)[1]*.5]})
  
# remove variables that meet either or both the NZV and NA criteria in addition to 5 irrelevant variables

rm_names <- unique(c(nz_names, na_names, "X", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "num_window"))

train <- raw_train[,!colnames(raw_train)%in% rm_names]
test <- raw_test[,!colnames(raw_test)%in% rm_names]
```
We have now removed 106 variables and are left with 54 variables to incorporate into prediction models.


### Partitioning data into training and testing sets

70 and 30% of the data were partitioned to training and testing sets, respectively.


```r
set.seed(1)
inTrain <- createDataPartition(y=train$classe, p=.7, list=FALSE)

training <-  train[inTrain,]
testing <- train[-inTrain,]
```

# Prediction with tree-based methods

We first try the simplest decision tree model to test prediction

## Classification tree

```r
tree.classe <- tree(classe~., training)
```

Advantages of using a decision tree is that it can be interpreted easily and displayed graphically.  The plot below shows that this model has yielded 14 terminal nodes using 10 variables
<img src="PracticalMachineLeaningProject_files/figure-html/unnamed-chunk-6-1.png" title="" alt="" style="display: block; margin: auto;" />

However, decision trees do not usually yield accuracy as good as more powerful, complex models. Here we predict the testing set using the trained model and our error rate shows 61% accuracy, which is not very good.


```r
tree.pred <- predict(tree.classe,newdata=testing, type="class")
confusionMatrix(tree.pred, testing$classe)$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1301  285   83   37   33
##          B   44  429  103   55  223
##          C   60   95  691  158   95
##          D  243  330  149  714  248
##          E   26    0    0    0  483
```

```r
confusionMatrix(tree.pred, testing$classe)$overall[1]
```

```
##  Accuracy 
## 0.6147833
```

## Pruning using CV

The bad prediction of the classifcation tree in part has to do with the fact the trained tree overfits the data.  Therefore, we try pruning the tree to improve accuracy.  We use `cv.tree()` to perform cross validation to determine the optimal level of tree complexity.    


```r
cv.classe <- cv.tree(tree.classe, FUN=prune.misclass)
cv.classe
```

```
## $size
## [1] 14 13 12 11  8  7  6  2  1
## 
## $dev
## [1] 4842 4842 5087 5262 5911 6217 7722 8672 9831
## 
## $k
## [1]      -Inf    0.0000  137.0000  158.0000  209.3333  302.0000  420.0000
## [8]  473.7500 1168.0000
## 
## $method
## [1] "misclass"
## 
## attr(,"class")
## [1] "prune"         "tree.sequence"
```

```r
plot(cv.classe$size, cv.classe$dev, type="b")
```

<img src="PracticalMachineLeaningProject_files/figure-html/unnamed-chunk-8-1.png" title="" alt="" style="display: block; margin: auto;" />
This plot shows that the trees with 13 and 14 terminal nodes results in the lowest CV error rate, with 4834 CV errors. This number of terminal nodes is same as that of our original tree, suggesting our cross validation did not help pruning the tree.  Therefore, we are unable to use a pruned tree to improve our prediction.

Since 61% is not a good prediction accuracy, we turn to three other tree-based methods of prediction: *boosting*, *bagging*, *random forest*

# Boosting

We use `gbm()` to train the data, using `n.tree` of 5000 and `interaction.depth` of 4.  

```r
set.seed(1)
boost.classe <- gbm(classe~., data=training, distribution="multinomial", n.trees=5000,interaction.depth=4, verbose=FALSE)
 
boost.pred <- predict(boost.classe, newdata = testing, n.trees=5000, type="response")
convert.boost.pred <- apply(boost.pred, 1, which.max)
 
table(convert.boost.pred, testing$classe)
```

```
##                   
## convert.boost.pred    A    B    C    D    E
##                  1 1598   76    1    6    4
##                  2   32 1000   58   10   17
##                  3   11   59  951   54   22
##                  4   25    3   13  886   18
##                  5    8    1    3    8 1021
```

```r
confusionMatrix(chartr("12345", "ABCDE", convert.boost.pred), testing$classe)$overall[1]
```

```
##  Accuracy 
## 0.9271028
```
We see that our accuracy is now 92.71%, which is a significant improvement from the decision tree model.

# Bagging

Bagging is same as random forest when all the variables are considered for sampling, so we use `randomForest()` with the full number of the training variables.  We first fit the model to the training data, then apply the model to the test set.


```r
set.seed(1)
bag.classe <- randomForest(classe~., data=training, mtry=length(names(training))-1, importance=TRUE)
bag.pred <- predict(bag.classe, newdata=testing)
table(bag.pred, testing$classe)
```

```
##         
## bag.pred    A    B    C    D    E
##        A 1658   15    5    2    0
##        B    6 1114    7    2    4
##        C    9    7 1009    7    3
##        D    0    2    5  950    2
##        E    1    1    0    3 1073
```

```r
confusionMatrix(bag.pred, testing$classe)$overall[1]
```

```
##  Accuracy 
## 0.9862362
```
The accuracy from the prediction is 98.62%, which is an improvement from the prediction with boosting.

# Random forest

Finally, we apply random forest, which limits the number of variables that are considered for sampling.


```r
set.seed(1)
rf.classe <- randomForest(classe~., data=training, mtry=sqrt(length(names(training))-1), importance=TRUE)
rf.pred <- predict(rf.classe, newdata=testing)
table(rf.pred, testing$classe)
```

```
##        
## rf.pred    A    B    C    D    E
##       A 1674    2    0    0    0
##       B    0 1135    7    0    0
##       C    0    2 1018    5    0
##       D    0    0    1  958    3
##       E    0    0    0    1 1079
```

```r
confusionMatrix(rf.pred, testing$classe)$overall[1]
```

```
##  Accuracy 
## 0.9964316
```

Our random forest model is able to predict the exercise activities with the highest accuracy rate of 99.6%.  It seems that bootstrapped tree methods (i.e., baggin and random forest) yield good prediction results for our current dataset.

# Final prediction

Using random forest that has yieled the best prediction results, the test dataset was predicted


```r
predict(rf.classe, newdata=test)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
