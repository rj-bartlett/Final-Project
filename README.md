### Abstract

Wine quality is typically determined through scent and taste. Modern day data collection and technology brings into question whether a machine learning algorithm can accurately predict wine quality based on physicochemical attributes. A study conducted in 2016 used a dataset on red wines to create four different fuzzy models in an attempt to change the wine industry’s method of categorizing wine quality. If machine learning models can automate this process using physicochemical attributes such as alcohol level, residual sugar, and pH, instead of relying on subjective human testers, it would increase the competition and provide wine manufacturers with a stronger baseline on how to create better wines. In this study, it is mentioned that neural networks and other related machine learning models are not capable of predicting wine quality accurately. In this project, I aim to test that out by comparing models learned in this course, specifically neural networks and random forest models to attempt to achieve as high accuracy levels as possible.

### Code

#### Neural Network

``` python
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tensorflow import keras

wine = pd.read_csv('/content/winequality-red (1).csv')
wine2 = wine.copy()
wine3 = wine.copy()
wine4 = wine.copy()

#Obtaining maximum for each column
maximum = []

for i in wine.columns:
  maximum.append(wine[i].max())
  print(wine[i].max())

#Changing wine to match version 1 specifications
#Good/Bad categories of quality
wine['good'] = wine['quality'] >=6
idxGood = wine['quality'] >= 6
wine['good'] = 0
wine.loc[idxGood, 'good'] = 1

#Drop quality column
wine.drop(columns = 'quality')

#Normalizing version 1
wine['fixed acidity'] = wine['fixed acidity'] / maximum[0]
wine['volatile acidity'] = wine['volatile acidity'] / maximum[1]
wine['citric acid'] = wine['citric acid'] / maximum[2]
wine['residual sugar'] = wine['residual sugar'] / maximum[3]
wine['chlorides'] = wine['chlorides'] / maximum[4]
wine['free sulfur dioxide'] = wine['free sulfur dioxide'] / maximum[5]
wine['total sulfur dioxide'] = wine['total sulfur dioxide'] / maximum[6]
wine['density'] = wine['density'] / maximum[7]
wine['pH'] = wine['pH'] / maximum[8]
wine['sulphates'] = wine['sulphates'] / maximum[9]
wine['alcohol'] = wine['alcohol'] / maximum[10]

#Changing wine2 to match version 2 specifications
#Good/Bad categories of quality
wine2['good'] = wine2['quality'] >=6

#Drop quality column
wine.drop(columns = 'quality')

#Changing wine3 to match version 3 specifications
wine3['fixed acidity'] = wine['fixed acidity'] / maximum[0]
wine3['volatile acidity'] = wine['volatile acidity'] / maximum[1]
wine3['citric acid'] = wine['citric acid'] / maximum[2]
wine3['residual sugar'] = wine['residual sugar'] / maximum[3]
wine3['chlorides'] = wine['chlorides'] / maximum[4]
wine3['free sulfur dioxide'] = wine['free sulfur dioxide'] / maximum[5]
wine3['total sulfur dioxide'] = wine['total sulfur dioxide'] / maximum[6]
wine3['density'] = wine['density'] / maximum[7]
wine3['pH'] = wine['pH'] / maximum[8]
wine3['sulphates'] = wine['sulphates'] / maximum[9]
wine3['alcohol'] = wine['alcohol'] / maximum[10]

#Split data
from sklearn.model_selection import train_test_split as tts
train, test = tts(wine)
train2, test2 = tts(wine2)
train3, test3 = tts(wine3)
train4, test4 = tts(wine4)

# column order in CSV file
column_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
                'total sulfur dioxide','density','pH','sulphates','alcohol','quality','good']

feature_names = column_names[:-2]
reg_label_name = column_names[-2]
gb_label_name = column_names[-1]

print("Features: {}".format(feature_names))
print("Label: {}".format(reg_label_name))
print("Label: {}".format(gb_label_name))

#Version 1
train_x = train[feature_names]
train_y = train[gb_label_name]

test_x = test[feature_names]
test_y = test[gb_label_name]

#Version2
train_x2 = train2[feature_names]
train_y2 = train2[gb_label_name]

test_x2 = test2[feature_names]
test_y2 = test2[gb_label_name]

#Version3
train_x3 = train3[feature_names]
train_y3 = train3[reg_label_name]

test_x3 = test3[feature_names]
test_y3 = test3[reg_label_name]

#Version4
train_x4 = train4[feature_names]
train_y4 = train4[reg_label_name]

test_x4 = test4[feature_names]
test_y4 = test4[reg_label_name]

#Model A versions 1 & 2 
model = tf.keras.Sequential([
  tf.keras.layers.Dense(1024, activation=tf.nn.relu, input_shape = (11,)), 
  tf.keras.layers.Dense(512, activation = tf.nn.relu),
  tf.keras.layers.Dense(256, activation = tf.nn.relu),
  tf.keras.layers.Dense(2, activation = tf.nn.softmax)])
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
              
#Model B versions 3&4
model2 = tf.keras.Sequential([
  tf.keras.layers.Dense(1024, activation=tf.nn.relu, input_shape = (11,)), 
  tf.keras.layers.Dense(512, activation = tf.nn.relu),
  tf.keras.layers.Dense(256, activation = tf.nn.relu),
  tf.keras.layers.Dense(10, activation = tf.nn.softmax)])
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
```

To fit model, I used model.fit(train_x, train_y, epochs = 10) and substituted the different versions of train_x and train_y depending on what model I was fitting 
To evaluate the model, I used model.evaluate(test_x, test_y) for the corresponding version of the model/train_x and train_y

#### Random Forest

``` R
rm(list=ls(all=TRUE))

library(tidymodels)
library(tidyverse)
library(randomForest)



data <- read.csv(
  "winequality-red (1).csv",
  header=TRUE)

dim(data)

data$quality[data$quality < 6] <- 0
data$quality[data$quality >= 6] <- 1
data$quality <- as.factor(data$quality)
summary(data)
sapply(data, class)

data_split <- initial_split(data, prop = 3/4)

data_train <- training(data_split)
data_test <- testing(data_split)

data_cv <- vfold_cv(data_train)

# define the recipe
data_recipe <- 
  # which consists of the formula (outcome ~ predictors)
  recipe(quality ~., data = data)
data_recipe

data_train_preprocessed <- data_recipe %>%
  # apply the recipe to the training data
  prep(data_train) %>%
  # extract the pre-processed training dataset
  juice()
data_train_preprocessed

rf_model <- 
  # specify that the model is a random forest
  rand_forest() %>%
  # specify that the `mtry` parameter needs to be tuned
  set_args(mtry = tune()) %>%
  # select the engine/package that underlies the model
  set_engine("ranger", importance = "impurity") %>%
  # choose either the continuous regression or binary classification mode
  set_mode("classification") 

rf_workflow <- workflow() %>%
  # add the recipe
  add_recipe(data_recipe) %>%
  # add the model
  add_model(rf_model)

# specify which values eant to try
rf_grid <- expand.grid(mtry = c(3, 4, 5))
# extract results
rf_tune_results <- rf_workflow %>%
  tune_grid(resamples = data_cv, #CV object
            grid = rf_grid, # grid of values to try
            metrics = metric_set(accuracy, roc_auc) # metrics we care about
  )

rf_tune_results %>%
  collect_metrics()

param_final <- rf_tune_results %>%
  select_best(metric = "accuracy")
param_final

rf_workflow <- rf_workflow %>%
  finalize_workflow(param_final)

rf_fit <- rf_workflow %>%
  # fit on the training set and evaluate on test set
  last_fit(data_split)

test_performance <- rf_fit %>% collect_metrics()
test_performance

### Random Forest Model 2
splits <- initial_split(data, strata = quality)

pns_other <- training(splits)
pns_test  <- testing(splits)

pns_other %>%
  count(quality) %>%
  mutate(prop = n/sum(n))

pns_test  %>%
  count(quality) %>%
  mutate(prop = n/sum(n))

val_set <- validation_split(pns_other,
                            strata = quality,
                            prop = 0.80)
val_set

cores <- parallel::detectCores()
cores

rf_mod <-
  rand_forest(mtry = tune(), trees = 1000) %>%
  set_engine("ranger", num.threads = cores) %>%
  set_mode("classification")

rf_recipe <-
  recipe(quality ~ ., data = pns_other)

rf_workflow <-
  workflow() %>%
  add_model(rf_mod) %>%
  add_recipe(rf_recipe)

rf_mod

rf_mod %>%
  parameters()

rf_res <-
  rf_workflow %>%
  tune_grid(val_set,
            grid = 25,
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(roc_auc))

rf_res %>%
  show_best(metric = "roc_auc")

autoplot(rf_res)
ggsave("rf_res.png")

rf_tune_results %>%
  collect_metrics()

param_final <- rf_tune_results %>%
  select_best(metric = "accuracy")
param_final

rf_workflow <- rf_workflow %>%
  finalize_workflow(param_final)

rf_fit <- rf_workflow %>%
  # fit on the training set and evaluate on test set
  last_fit(data_split)

test_performance <- rf_fit %>% collect_metrics()
test_performance
```
