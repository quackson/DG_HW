# python3 synthesize.py --text "hello world" --restore_step 900000 \
# --mode single -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/hifigan.yaml -t config/LJSpeech/train.yaml

python3 synthesize.py --text "The shaven face of the priest is a further item to the same effect." --restore_step 900000 --mode single -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/hifigan.yaml -t config/LJSpeech/train.yaml