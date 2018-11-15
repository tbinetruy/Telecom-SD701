#!/bin/bash

USER=binetruy
HOST=c133-01
RESULTS_FOLDER=results
MODEL_NAME=$1

# compile
sbt package

# upload
ssh $USER@$HOST "mkdir -p kaggle/$RESULTS_FOLDER"
ssh $USER@$HOST "mkdir -p kaggle/$RESULTS_FOLDER/$MODEL_NAME"
scp target/scala-2.11/simple-project_2.11-1.0.jar $USER@$HOST:~/kaggle/$RESULTS_FOLDER/$MODEL_NAME

# exec
ssh $USER@$HOST "cd kaggle/$RESULTS_FOLDER/$MODEL_NAME && ./../../spark/bin/spark-submit simple-project_2.11-1.0.jar"

# repatriate result
mkdir -p $RESULTS_FOLDER
mkdir -p $RESULTS_FOLDER/$MODEL_NAME
scp $USER@$HOST:~/kaggle/$RESULTS_FOLDER/$MODEL_NAME/results/*.csv $RESULTS_FOLDER/$MODEL_NAME/results.csv
scp $USER@$HOST:~/kaggle/$RESULTS_FOLDER/$MODEL_NAME/f1-score $RESULTS_FOLDER/$MODEL_NAME/f1-score
