# Tutorial on Artificial Intelligence with Language Models for Audio

This tutorial aims to cover basics of adapting and fine-tuning language models for text to speech and speech recognition audio applications.

## Text to Speech

As an example, we implement fine-tuning of a transformer model that was pretrained on an English language corpus. The model is to learn to pronounce numbers correctly in new languages that the model has not been pretrained on.

For example, the pretrained model has difficulty pronouncing Dutch words correctly.

To keep things very simple, to limit the size of required training data and to make it possible to run the training on a low-end personal computer, we use data with the Dutch text or the number representation of the numbers 1 to 10, as well as audio files with the correct pronunciation.

Here I describe how to fine-tuned the SpeechT5ForTextToSpeech "microsoft/speecht5_tts" model to correctly pronounce the Dutch word "negen".

Training was done by:

* preparing training and evaluation data with the texts to pronounce and corresponding audio (.wav) files
* loading the data with the transformers datasets library 
* loading the SpeechT5ForTextToSpeech "microsoft/speecht5_tts" model and SpeechT5Processor


Data preparation and fine-tuning is done by running SpeechT5/trainSpeechT5ttsGetallen.py on branch master: https://github.com/JacobAdj/AudioAI/blob/master/SpeechT5/trainSpeechT5ttsGetallen.py


