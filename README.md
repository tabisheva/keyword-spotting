# Keyword Spotting

Attention-based model for detecting trigger words in streaming mode based on [this paper](https://arxiv.org/pdf/1803.10916.pdf)

In this case trigger word is `sheila` (la-la-la)

## Usage

Download [dataset](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz)

Set path to file in config

run ```python inference.py``` and you will get probabilities of trigger word over the audio
