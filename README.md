# Tutorial on Artificial Intelligence with Language Models for Audio

This tutorial aims to cover basics of adapting and fine-tuning language models for text to speech and speech recognition audio applications.

## Text to Speech

As an example, we implement fine-tuning of a transformer model that was pretrained on an English language corpus. The model is to learn to pronounce numbers correctly in new languages that the model has not been pretrained on.

For example, the pretrained model has difficulty pronouncing the Dutch words for numbers correctly.

To keep things very simple, to limit the size of required training data and to make it possible to run the training on a low-end personal computer, we use data with the Dutch text or the number representation of the numbers 1 to 10, as well as audio files with the correct pronunciation. The following examples can be run on a personal computer with only a Celeron CPU and just 8GB RAM. Of course for realistic fine-tuning tasks much larger datasets, more time, more memory and/or GPUs would be necessary.

Here I describe how to fine-tune the `SpeechT5ForTextToSpeech "microsoft/speecht5_tts"` model to correctly pronounce the Dutch word "negen". The pretrained model is decribed here: https://huggingface.co/microsoft/speecht5_tts

Training was done by:

* preparing training and evaluation data with the texts to pronounce and corresponding audio (`.wav`) files
* loading the data with the transformers datasets library 
* loading the `SpeechT5ForTextToSpeech "microsoft/speecht5_tts"` model and `SpeechT5Processor`
* running training with the transformers Training API


Data preparation and fine-tuning is done by running SpeechT5/trainSpeechT5ttsGetallen.py on branch master: https://github.com/JacobAdj/AudioAI/blob/master/SpeechT5/trainSpeechT5ttsGetallen.py

### Required libraries

The following imports are used for fine-tuning:

```python
import os
os.environ["WANDB_MODE"] = "disabled"

import tensorflow as tf
import torch
import torchaudio

from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor, TrainingArguments, Trainer
from transformers import SpeechT5HifiGan
from transformers import TrainerCallback

from datasets import load_dataset , Audio
from datasets import concatenate_datasets
```


If not yet available in your Python installation, these can be installed from your command prompt with:
```shell
pip install tensorflow 
pip install torch
pip install torchaudio

pip install transformers, datasets 
```

### Training and evaluation data

Data for training and/or evaluation are loaded with

```python
DATASET_DIR = "./numberaudiodata/"
AUDIO_DIR = "./numberaudiodata/"

dataset = load_dataset("json", data_files = DATASET_DIR + "trainNL.json")

dataset = dataset["train"]  # Ensure correct split selection
```

The JSON file describing the data looks like

```json
[
    {
        "root_path": "./numberaudiodata",
        "audio_file": "./numberaudiodata/number1.wav",
        "text": "1",
        "spoken_text": "een",
        "speaker_id": "speaker1",
        "duration": 2.0
    },

.......

    {
        "root_path": "./numberaudiodata",
        "audio_file": "./numberaudiodata/number10.wav",
        "text": "10",
        "spoken_text": "tien",
        "speaker_id": "speaker1",
        "duration": 2.0
    }
]
```

There are only 10 data items, which is too few for good training, so we augment the data by replicating the data to make the training dataset 10 times larger:

```python
datasets = []

for d in range(10):
    datasets.append(dataset)

dataset = concatenate_datasets(datasets)

print(len(dataset))  # Check new dataset size
```

The dataset does not yet have actual sound data, which are needed for training. The JSON file has the file names of the sound data, wahic are in `.wav` files. We load these into the dataset as follows:

```python
def update_example(example, audio_id):
    example['audio_id'] = audio_id
    example['language'] = 9
    example['gender'] = 'female'
    example['speaker_id'] = '1122'
    example['is_gold_transcript'] = True
    example['accent'] = 'None'

    waveform, sampling_rate = torchaudio.load(AUDIO_DIR + example['audiofile'])
    # Convert to NumPy
    audio_array = waveform.numpy()
    # print(audio_array.flatten())
    example['audio'] = {}
    example['audio']['array'] = audio_array.flatten()
    example['audio']['sampling_rate'] = sampling_rate

    return example

# Use `enumerate` with `map()` to update dataset
dataset = dataset.map(lambda example, idx: update_example(example, idx + 1), with_indices=True)

dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

print(dataset[3])

```

The sound data, as waveform, are now in the example fields `example['audio']['array']` and their sampling rates are in the fields `example['audio']['sampling_rate']`.
Sampling rates indicate how many times per second the strength of the sound signal is measured and are needed for correct interpretation of the sound signal.
The `print(dataset[3])` statement serves to visually check the the data look right.

