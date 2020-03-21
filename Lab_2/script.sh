# Execute feature extractor on Train dataset (feats and megam)
python3 feature_extactor.py --dir data/Train --type feats > /dev/null
python3 feature_extactor.py --dir data/Train --type megam > /dev/null

python3 feature_extactor.py --dir data/Devel --type feats > /dev/null
python3 feature_extactor.py --dir data/Devel --type megam > /dev/null

python3 ML_model.py
