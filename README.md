# Proof of concept code related to machine learning with audio


I am experimenting to fine-tune Text to Speech models to pronounce numbers correctly in new languages that the models have not been pretrained on.

16 April 2025: I fine-tuned the SpeechT5ForTextToSpeech "microsoft/speecht5_tts" model to correctly pronounce the Dutch word "negen".

Training was done by running SpeechT5/trainSpeechT5ttsGetallen.py on branch master: https://github.com/JacobAdj/AudioAI/blob/master/SpeechT5/trainSpeechT5ttsGetallen.py
