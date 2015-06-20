# PA81
Coursera Practical Machine Learning - Project


## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, my goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website [here:](http://groupware.les.inf.puc-rio.br/har) (see the section on the Weight Lifting Exercise Dataset).


## Data

The training data for this project are available from [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv).

The test data are available from [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv).

The data for this project comes from this [original source](http://groupware.les.inf.puc-rio.br/har). 

## Objective

* The goal of the project is to predict the manner in which participants did the exercise. This is the "classe" variable in the training set. 
* This report will describe how I built the model, how I used cross validation,  the expected out-of-sample error and explain the choices made.
* I will subsequently use the prediction model to predict 20 different test cases.

