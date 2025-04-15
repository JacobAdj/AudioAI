from gtts import gTTS

import json
# from num2words import num2words
from expandnumbers import getal_in_woorden


# text = "één"  # Number pronunciation in Dutch
# tts = gTTS(text, lang="nl")
# tts.save("D:/LanguageModels/audiodata/number1.wav")


# Function to generate spoken number data
def generate_dutch_numbers(start=1, end=10):

    dataset = []
    
    for num in range(start, end+1):

        spoken_text = getal_in_woorden(num)  # Convert number to Dutch words
        # spoken_text = num2words(num, lang='nl')  # Convert number to Dutch words

        tts = gTTS(spoken_text, lang="nl")
        tts.save(f"D:/LanguageModels/audiodata/number{num}.wav")

        dataset.append(
            {
                "root_path" : "D:/LanguageModels/audiodata",
                "audio_file": f"D:/LanguageModels/audiodata/number{num}.wav",  
                "text": str(num),
                "spoken_text": spoken_text,
                "speaker_id": "speaker1",
                "duration": 2.0  # Placeholder duration
            }
        )
    
    return dataset

# Generate dataset
dutch_dataset = generate_dutch_numbers(1, 10)

# Save to JSON file
with open("D:/LanguageModels/audiodata/dutch_numbers_dataset.json", "w", encoding="utf-8") as f:
    json.dump(dutch_dataset, f, ensure_ascii=False, indent=4)

print("Dutch number dataset saved as 'dutch_numbers_dataset.json'")
