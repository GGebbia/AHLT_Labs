# Execute feature extractor on Train dataset (feats and megam)
python feature_extactor_with_knowledge.py --dir data/Train --type feats > /dev/null
python feature_extactor_with_knowledge.py --dir data/Train --type megam > /dev/null

python feature_extactor_with_knowledge.py --dir data/Devel --type feats > /dev/null
python feature_extactor_with_knowledge.py --dir data/Devel --type megam > /dev/null

python ML_model.py
