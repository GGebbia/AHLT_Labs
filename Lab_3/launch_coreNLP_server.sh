#!/usr/bin/env bash

echo "Launching coreNLP server!"

# default dir to be executed
coreNLP_dir="../../stanford-corenlp-full-2018-10-05/"

# if there is input argument change the default
if [ -z "$1" ]
  then
    coreNLP_dir=$1
fi

# execute command to launch in the directory 
cd $coreNLP_dir && java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
