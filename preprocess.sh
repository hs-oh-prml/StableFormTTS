export PYTHONPATH=.
CONFIG="configs/models/tts/stableform_tts.yaml";
python data_gen/tts/runs/preprocess.py --config $CONFIG
python data_gen/tts/runs/binarize.py --config $CONFIG
