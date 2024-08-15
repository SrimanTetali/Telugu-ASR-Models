import torch
from transformers import pipeline

# Path to the audio file to be transcribed
audio = "/home/srimantetali/Sriman/projects/output.mp3"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Specify the model you want to use. Change this to any Wav2Vec2 model you want to use

model_name = "addy88/wav2vec2-telugu-stt"
#model_name = "Anujgr8/wav2vec2-base-Telugu-large"  
#model_name = "Harveenchadha/vakyansh-wav2vec2-telugu-tem-100"
#model_name = "anuragshas/wav2vec2-large-xlsr-53-telugu"


# Initialize the ASR pipeline
transcribe = pipeline(
    task="automatic-speech-recognition", 
    model=model_name, 
    chunk_length_s=30, 
    device=device
)

# Perform transcription
transcription = transcribe(audio)["text"]

# Remove unwanted tokens
cleaned_transcription = transcription.replace("<s>", "").strip()

# Print the cleaned transcription
print('Transcription: ', cleaned_transcription)
