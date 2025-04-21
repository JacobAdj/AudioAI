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

### Training and evaluation data loading

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

### Data pre-processing

The dataset does not yet have actual sound data, which are needed for training. The JSON file has the file names of the sound data, which are in `.wav` files. We load these into the dataset as follows:

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

The sound data, as waveform, are now in the fields `example['audio']['array']` and their sampling rates are in the fields `example['audio']['sampling_rate']` of the data examples.
Sampling rates indicate how many times per second the strength of the sound signal is measured and are needed for correct interpretation of the sound signal.
The `print(dataset[3])` statement serves to visually check that the data look right.

We now have the sound data, but these are not yet in an appropriate form to be used as inputs of a transformer model, a type of neural network. The raw waveform data have to be transformed to a spectrogram. The following code takes care of that:

```python
processor = SpeechT5Processor.from_pretrained(MODEL_NAME , cache_dir=CACHE_DIR)

def prepare_dataset(example):

    audio = example["audio"]

    example = processor(
        text=example["text"],
        audio_target=audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_attention_mask=False,
    )

    # strip off the batch dimension
    example["labels"] = example["labels"][0]

    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation" , cache_dir=CACHE_DIR)

    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"])

    example["speaker_embeddings"] = speaker_embeddings

    return example


processed_example = prepare_dataset(dataset[0])
print(list(processed_example.keys()))
print(processed_example["speaker_embeddings"].shape)
print('type(processed_example["labels"])' , type(processed_example["labels"]))

dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)
print('dataset.column_names' , dataset.column_names)
print('training data negen input_ids' , dataset[8]['input_ids'])
print('training data negen labels' , dataset[8]['labels'][0])

loadedspectrogram = dataset[8]['labels']
print('loadedspectrogram is ' , type(loadedspectrogram))

# Ensure spectrogram is a PyTorch tensor
spectrogram = torch.tensor(loadedspectrogram)

```

The `SpeechT5Processor` does the necessary conversions: `text=example["text"]` converts text to tokens, and `audio_target=audio["array"]` and   `sampling_rate=audio["sampling_rate"]` convert the raw audio signal to a spectrogram.

There is also a bit of code to load speaker embeddings, to ensure pronunciations corresponding to the characteristics of a certain speaker:

```python
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation" , cache_dir=CACHE_DIR)

    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"])

    example["speaker_embeddings"] = speaker_embeddings
```

### Training

We are now ready to do some training by running
```python
  trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = dataset,
        data_collator = data_collator,
        callbacks=[SaveBestModelCallback(model, processor, "D:/LanguageModels/ftT5modelDutchNumbers")]
    )

    trainer.train()
```
after defining the paremeters of the `Trainer` class.

The first parameter, `model`, is the pretrained model to be fine-tuned:
```python
MODEL_NAME = "microsoft/speecht5_tts"

CACHE_DIR = "D:/LanguageModels/cache"

model = SpeechT5ForTextToSpeech.from_pretrained(MODEL_NAME , cache_dir=CACHE_DIR)
```
The `MODEL_NAME` gives the name of the pretrained model to de downloaded from the HuggingFace hub, in this case the Microsoft `speecht5_tts` model.
`CACHE_DIR` is not required, but used here to download the model to a custom cache directory to save space on my C: disk.

The second parameter, `args`, has the arguments of the training algorithm, looking something like this:
```python
training_args = TrainingArguments(
        output_dir="D:/LanguageModels/out_T5tts",
        run_name="tts_exp", 
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        num_train_epochs=100,
        save_steps=100,
        save_total_limit=2,
        # logging_dir="D:/LanguageModels/logs",
        logging_steps=1,
        learning_rate=0.001,
        optim="adamw_torch_fused",
        eval_strategy="no",
        # eval_strategy="steps",
        eval_steps=500,
        weight_decay=0.0,
        fp16=True
    )
```
