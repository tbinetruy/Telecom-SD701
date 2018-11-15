#!/bin/bash

USER=binetruy
HOST=c133-01
MODEL_NAME=$1

# compile
sbt package

# upload
ssh $USER@$HOST "mkdir -p kaggle/$MODEL_NAME"
scp target/scala-2.11/simple-project_2.11-1.0.jar $USER@$HOST:~/kaggle/$MODEL_NAME

# exec
ssh $USER@$HOST "cd kaggle/$MODEL_NAME && ./../spark/bin/spark-submit simple-project_2.11-1.0.jar"

# repatriate result
mkdir -p $MODEL_NAME
scp $USER@$HOST:~/kaggle/$MODEL_NAME/results/*.csv $MODEL_NAME/results.csv
scp $USER@$HOST:~/kaggle/$MODEL_NAME/f1-score $MODEL_NAME/f1-score
