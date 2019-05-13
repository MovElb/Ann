#!/bin/bash
mkdir squad2_data
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json -O squad2_data/train-v2.0.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O squad2_data/dev-v2.0.json
wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O squad2_data/glove.840B.300d.zip
unzip squad2_data/glove.840B.300d.zip -d squad2_data
rm squad2_data/glove.840B.300d.zip

