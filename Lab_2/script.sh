#!/bin/sh

NO_KNOWLEDGE_DIR="./extracted_features/no_knowledge"

# Execute feature extractor on Train dataset (feats and megam)
python3 feature_extactor.py --dir data/$1 --type feats --out "$NO_KNOWLEDGE_DIR/$1"
python3 feature_extactor.py --dir data/$1 --type megam --out "$NO_KNOWLEDGE_DIR/$1"

python3 feature_extactor.py --dir data/$2 --type feats --out "$NO_KNOWLEDGE_DIR/$2"
python3 feature_extactor.py --dir data/$2 --type megam --out "$NO_KNOWLEDGE_DIR/$2"


python3 ML_model.py --train "$NO_KNOWLEDGE_DIR/$1" --test "$NO_KNOWLEDGE_DIR/$2" --out "FINAL_RESULTS/no_knowledge_$2"

