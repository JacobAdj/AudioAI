# Tutorial on Artificial Intelligence with Language Models for Audio

This tutorial aims to cover basics of adapting and fine-tuning language models for text to speech and speech recognition audio applications.

## Text to Speech

Here we implement fine-tuning of a transformer model that was pretrained on an English language corpus. The model is to learn to pronounce numbers correctly in new languages that the model has not been pretrained on.

For example, the pretrained model has difficulty pronouncing Dutch words correctly.

To keep things very simple, to limit the size of required training data and to make it possible to run the training on a low-end personal computer, we use data with the Dutch text or the number representation of the number 1 to 10, as wellas audio files with the correct pronunciation.

I am experimenting to fine-tune Text to Speech models to pronounce numbers correctly in new languages that the models have not been pretrained on.

16 April 2025: I fine-tuned the SpeechT5ForTextToSpeech "microsoft/speecht5_tts" model to correctly pronounce the Dutch word "negen".

Training was done by:

* preparing training and evaluation data with the texts to pronounce and corresponding audio (.wav) files
* loading the data with the transformers datasets library 
* loading the SpeechT5ForTextToSpeech "microsoft/speecht5_tts" model and SpeechT5Processor


running SpeechT5/trainSpeechT5ttsGetallen.py on branch master: https://github.com/JacobAdj/AudioAI/blob/master/SpeechT5/trainSpeechT5ttsGetallen.py
