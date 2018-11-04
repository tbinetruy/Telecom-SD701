#!/bin/bash

# compile
sbt package

# upload
scp target/scala-2.11/simple-project_2.11-1.0.jar binetruy@c133-07:~/kaggle

# exec
ssh binetruy@c133-07 "cd kaggle && ./spark/bin/spark-submit simple-project_2.11-1.0.jar"

# repatriate result
scp binetruy@c133-07:~/kaggle/results/*.csv result.csv

