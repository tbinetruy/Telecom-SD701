#!/bin/bash

USER=binetruy
HOST=c133-01

# compile
sbt package

# upload
scp target/scala-2.11/simple-project_2.11-1.0.jar $USER@$HOST:~/kaggle

# exec
ssh $USER@$HOST "cd kaggle && ./spark/bin/spark-submit simple-project_2.11-1.0.jar"

# repatriate result
scp $USER@$HOST:~/kaggle/results/*.csv results.csv

# delete result file on server
ssh $USER@$HOST "rm -rf ~/kaggle/results/"

