
import os
os.environ["WANDB_MODE"] = "disabled"

import torch
import torchaudio

# from IPython.display import Audio

import soundfile as sf
import sounddevice as sd
# import librosa
# import librosa.display

from datasets import load_dataset 

from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor
from transformers import SpeechT5HifiGan


MODEL_NAME = "microsoft/speecht5_tts"

CACHE_DIR = "D:/LanguageModels/cache"
DATASET_DIR = "D:/LanguageModels/dataset/"
AUDIO_DIR = "D:/LanguageModels/dataset/audio/"

# model = SpeechT5ForTextToSpeech.from_pretrained("D:/LanguageModels/ftT5modelGetallen")
# processor = SpeechT5Processor.from_pretrained("D:/LanguageModels/ftT5processorGetallen")
model = SpeechT5ForTextToSpeech.from_pretrained(MODEL_NAME , cache_dir=CACHE_DIR)
processor = SpeechT5Processor.from_pretrained(MODEL_NAME , cache_dir=CACHE_DIR)

# for name, param in model.named_parameters():
#     print(name, param.requires_grad) 

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation" , cache_dir=CACHE_DIR)
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

inputs = processor(text="vijf", return_tensors="pt")

print('inputs["input_ids"]' , inputs["input_ids"] )

spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)

print('spectrogram' , spectrogram)
print('spectrogram type' , type(spectrogram))

# # Convert PyTorch tensor to NumPy array
# spectrogramnp = spectrogram.cpu().numpy().T

# print('spectrogramnp.shape' , spectrogramnp.shape)

# # Convert Mel spectrogram back to audio
# speech = librosa.feature.inverse.mel_to_audio(spectrogramnp, sr=16000)

vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan" , cache_dir=CACHE_DIR)
speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

print('speech' , speech)
print('speech shape' , speech.shape)



# Play the sound
sd.play(speech, 16000)
sd.wait()  # Wait until playback finishes

sf.write('./output/negen_T5modelGetallen.wav', speech, 16000)

print("Speech synthesis complete and saved")


# Audio(speech, rate=16000)



