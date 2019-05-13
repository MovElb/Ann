#!/bin/bash
mkdir data
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json -O data/train-v2.0.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O data/dev-v2.0.json
wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O data/glove.840B.300d.zip
unzip data/glove.840B.300d.zip -d data
rm data/glove.840B.300d.zip

