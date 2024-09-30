export PYTHONPATH=.
DEVICE=3;
CONFIG="configs/models/tts/stableform_tts.yaml";
MODEL_NAME="0930_StableFormTTS_libri16k";

CUDA_VISIBLE_DEVICES=$DEVICE python tasks/run.py \
    --config $CONFIG \
    --exp_name $MODEL_NAME \
    --reset \
    --hparams=$HPARAMS

CUDA_VISIBLE_DEVICES=$DEVICE python tasks/run.py \
    --config $CONFIG \
    --exp_name $MODEL_NAME \
    --infer \
    --hparams=$HPARAMS \
    --reset