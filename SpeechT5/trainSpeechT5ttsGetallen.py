
import os
os.environ["WANDB_MODE"] = "disabled"

import tensorflow as tf
# print(dir(tf))


import torch
import torchaudio

# import soundfile as sf
# import sounddevice as sd


from transformers import TrainingArguments, Trainer

from transformers import SpeechT5HifiGan

from transformers import TrainerCallback

# from peft import get_peft_model, LoraConfig

MODEL_NAME = "microsoft/speecht5_tts"

CACHE_DIR = "D:/LanguageModels/cache"
DATASET_DIR = "D:/LanguageModels/audiodata/"
AUDIO_DIR = "D:/LanguageModels/audiodata/"



from datasets import load_dataset , Audio
from datasets import concatenate_datasets

# dataset = load_dataset("facebook/voxpopuli", "nl", split="train[:10]", trust_remote_code=True, cache_dir=CACHE_DIR)
# len(dataset)

dataset = load_dataset("json", data_files = DATASET_DIR + "trainNL.json")

dataset = dataset["train"]  # Ensure correct split selection

datasets = []

for d in range(10):
    datasets.append(dataset)

# Duplicate the dataset (e.g., twice)
dataset = concatenate_datasets(datasets)

print(len(dataset))  # Check new dataset size

# exit()


# gender_mapping = {"number1.wav": "male", "number2.wav": "female"}  # Example mapping
# dataset = dataset.map(lambda example: {**example, "gender": gender_mapping.get(example["audio"], "unknown")})
print(type(dataset))
print(len(dataset))
print(dataset[0])

audio_id = 1

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
# exit(0)

from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor

processor = SpeechT5Processor.from_pretrained(MODEL_NAME , cache_dir=CACHE_DIR)




# def preprocess(batch):
#     # Tokenize text for model input
#     batch["input_values"] = processor.tokenizer(batch["text"], 
#                                                 return_tensors="pt", 
#                                                 padding="max_length", max_length=21, truncation=True).input_ids

#     print('tokenized text shape' , batch["input_values"].shape)

#     # Convert audio into spectrogram (or mel-spectrogram)
#     speech_array, sampling_rate = torchaudio.load(AUDIO_DIR + batch["audio"])

#     print(f"Audio sample rate: {sampling_rate}")
#     print(f"Waveform min: {speech_array.min()}, max: {speech_array.max()}")

#     mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sampling_rate, n_mels=10, n_fft=512)(speech_array)

#     print(f"Mel Spectrogram Shape: {mel_spectrogram.shape}")
#     # print(mel_spectrogram)
#     if not torch.any(mel_spectrogram):
#         print("Warning: Spectrogram contains only zeros!")
#     else:
#         print('Spectrogram ok')

#     batch["labels"] = mel_spectrogram

#     batch["labels"] = batch["labels"][0]

#     # batch["labels"] = torch.nn.functional.pad(mel_spectrogram, (0, max(0, 150 - mel_spectrogram.shape[-1])), 
#     #                                           mode="constant", value=0)
    
#     # print('tokenized sound ' , batch["labels"])

#     return batch


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

    # print('type(example["labels"]' , type(example["labels"]))
    # # Convert list to torch.Tensor
    # spectrogram = example["labels"]
    # spectrogram_tensor = torch.tensor(spectrogram, dtype=torch.float32)
    # spectrogram_tensor = spectrogram_tensor.unsqueeze(0)
    # example["labels"] = spectrogram_tensor
    # print('type(example["labels"] now' , type(example["labels"]))

    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation" , cache_dir=CACHE_DIR)

    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"])
    # speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    example["speaker_embeddings"] = speaker_embeddings

    # use SpeechBrain to obtain x-vector
    # example["speaker_embeddings"] = create_speaker_embedding(audio["array"])

    return example


processed_example = prepare_dataset(dataset[0])
print(list(processed_example.keys()))
print(processed_example["speaker_embeddings"].shape)
print('type(processed_example["labels"])' , type(processed_example["labels"]))

# exit()
# import matplotlib.pyplot as plt

# plt.figure()
# plt.imshow(processed_example["labels"].T)
# plt.show()

dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)
# dataset.set_format(type="torch", columns=["labels"])  # Ensures labels are tensors
# dataset.set_format(type="torch", columns=["input_ids", "labels", "speaker_embeddings"])


# dataset = dataset.map(prepare_dataset)
print('dataset.column_names' , dataset.column_names)


print('training data negen input_ids' , dataset[8]['input_ids'])
print('training data negen labels' , dataset[8]['labels'][0])


loadedspectrogram = dataset[8]['labels']

print('loadedspectrogram is ' , type(loadedspectrogram))

# Ensure spectrogram is a PyTorch tensor
spectrogram = torch.tensor(loadedspectrogram)
# spectrogram = torch.tensor(loadedspectrogram).unsqueeze(0)  # Add batch dimension if needed

vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan" , cache_dir=CACHE_DIR)
# speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

print('vocoder loaded')

# Generate waveform from spectrogram
speech = vocoder(spectrogram).detach()

print('speech' , speech)
print('speech shape' , speech.shape)



# Play the sound
# sd.play(speech, 16000)
# sd.wait()  # Wait until playback finishes

# sf.write('./output/loaded_negenGetallen.wav', speech, 16000)


# dataset = dataset.train_test_split(test_size=0.1)

from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class TTSDataCollatorWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        label_features = [{"input_values": feature["labels"]} for feature in features]
        
        speaker_features = [feature["speaker_embeddings"] for feature in features]

        # collate the inputs and targets into a batch
        batch = processor.pad(
            input_ids=input_ids, labels=label_features, return_tensors="pt"
        )

        # replace padding with -100 to ignore loss correctly
        batch["labels"] = batch["labels"].masked_fill(
            batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100
        )

        # not used during fine-tuning
        del batch["decoder_attention_mask"]

        # round down target lengths to multiple of reduction factor
        if model.config.reduction_factor > 1:
            target_lengths = torch.tensor(
                [len(feature["input_values"]) for feature in label_features]
            )
            target_lengths = target_lengths.new(
                [
                    length - length % model.config.reduction_factor
                    for length in target_lengths
                ]
            )
            max_length = max(target_lengths)
            batch["labels"] = batch["labels"][:, :max_length]

        # also add in the speaker embeddings
        batch["speaker_embeddings"] = torch.tensor(speaker_features)

        # print('batch["input_ids"]' , batch["input_ids"][3])
        # print('batch["labels"]' , batch["labels"][3])

        return batch
    

data_collator = TTSDataCollatorWithPadding(processor=processor)


training = True


class SaveBestModelCallback(TrainerCallback):
    def __init__(self, model, processor, save_path, best_loss=float("inf")):
        self.model = model
        self.processor = processor
        self.save_path = save_path
        self.best_loss = best_loss

    def on_step_end(self, args, state, control, **kwargs):
        if state.log_history and "loss" in state.log_history[-1]: 
            current_loss = state.log_history[-1]["loss"]
            if current_loss < 1.00 and current_loss < self.best_loss:
                self.best_loss = current_loss
                print(f"New best loss: {self.best_loss} - Saving model...")
                self.model.save_pretrained("D:/LanguageModels/ftT5modelGetallen9")
                self.processor.save_pretrained("D:/LanguageModels/ftT5processorGetallen9")


import torch.nn as nn


class MSETrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        labels = inputs.pop("labels", None)  # Extract target speech outputs
        if labels is None:
            raise ValueError("Labels are missing in inputs!")
        else:
            print('labels' , labels)

# Ensure inputs are valid before passing to model
        for key, value in inputs.items():
            if value is None:
                raise ValueError(f"Input '{key}' is None! Check preprocessing.")

        print(' inputs are valid before passing to model') 

        outputs = model(**inputs)  # Forward pass to get predictions

        print('outputs' , outputs) 

        loss_fn = nn.MSELoss()  # Define MSE loss function
        loss = loss_fn(outputs.logits, labels)  # Compute loss between predictions & labels

        print('loss' , loss) 

        return (loss, outputs) if return_outputs else loss

    # def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

    #     # labels = inputs.pop("labels")  # Extract target speech outputs
    #     # outputs = model(**inputs)  # Model prediction
    #     loss = super().compute_loss(model, inputs, return_outputs)

    #     # loss = torch.nn.MSELoss()(outputs.logits, labels)  # Compute MSE loss

    #     return  loss



if training:

    print('now training')


    model = SpeechT5ForTextToSpeech.from_pretrained(MODEL_NAME , cache_dir=CACHE_DIR)

    print(model.config)
    targetmodules = []
    nall = 0
    ntarget = 0
    for name, module in model.named_modules():
        nall += 1
        if 'speech_decoder_postnet' in name:
            ntarget +=1
            targetmodules.append(name)
            print(name, "->", type(module))

    print('nall' , nall , 'ntarget' , ntarget)

  
    print(dir(model))

    print(model.named_parameters())

    # exit()

    # Fine-tune only the postnet
    for param in model.parameters():
        param.requires_grad = False  # Freeze all layers
    for param in model.speech_decoder_postnet.parameters():
        param.requires_grad = True  # Unfreeze postnet

    # config = LoraConfig(
    #     r=8,  # Rank of LoRA matrices
    #     lora_alpha=32,
    #     lora_dropout=0.1,
    #     # target_modules=targetmodules  # Apply LoRA to decoder layers
    #     target_modules = ["speech_decoder_postnet.feat_out", "speech_decoder_postnet.prob_out"]  # Example names

    #     # target_modules=["decoder.block", "decoder.attention", "speech_decoder_prenet"],  # Adjust based on model structure
    #     # target_modules=["decoder.layers"],  # Apply LoRA to decoder layers
    # )

    # peftmodel = get_peft_model(model, config)

    training_args = TrainingArguments(
        output_dir="D:/LanguageModels/out_T5tts",
        run_name="tts_exp", 
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        num_train_epochs=1000,
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

    # from transformers import DataCollatorWithPadding

    # collator = DataCollatorWithPadding(tokenizer=processor.tokenizer, padding=True)

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = dataset,
        data_collator = data_collator,
        callbacks=[SaveBestModelCallback(model, processor, "D:/LanguageModels/ftT5modelGetallen")]
    )

    

    trainer.train()


    # model.save_pretrained("D:/LanguageModels/ftT5modelGetallen")
    # processor.save_pretrained("D:/LanguageModels/ftT5processorGetallen")



