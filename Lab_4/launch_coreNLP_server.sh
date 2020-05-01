#!/bin/sh
echo "Launching coreNLP server!"

# default dir to be executed
CORENLP_DIR="${HOME}/stanford-corenlp-full-2018-10-05/"

# if there is input argument change the default
if [ ! -z "$1" ];then
	CORENLP_DIR=$1
fi

# execute command to launch in the directory
java -mx4g -cp "${CORENLP_DIR}/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
