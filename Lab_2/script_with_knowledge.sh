#!/bin/sh

KNOWLEDGE_DIR="./extracted_features/knowledge"

# Execute feature extractor on Train dataset (feats and megam)
python3 feature_extactor_with_knowledge.py --dir data/$1 --type feats --out "$KNOWLEDGE_DIR/$1"
python3 feature_extactor_with_knowledge.py --dir data/$1 --type megam --out "$KNOWLEDGE_DIR/$1"

python3 feature_extactor_with_knowledge.py --dir data/$2 --type feats --out "$KNOWLEDGE_DIR/$2"
python3 feature_extactor_with_knowledge.py --dir data/$2 --type megam --out "$KNOWLEDGE_DIR/$2"

python3 ML_model.py --train "$KNOWLEDGE_DIR/$1" --test "$KNOWLEDGE_DIR/$2" --out "FINAL_RESULTS/knowledge_$2"

