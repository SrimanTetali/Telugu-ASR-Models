import torch
from transformers import pipeline

# Path to the audio file to be transcribed
audio_path = "/home/srimantetali/Sriman/projects/Generated_Audio.mp3"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Choose the model to use
# model_name = "Anujgr8/w2v-bert-Telugu-large"  
model_name = "cdactvm/telugu_w2v-bert_model"

# Initialize the ASR pipeline
try:
    transcribe = pipeline(
        task="automatic-speech-recognition",
        model=model_name,
        chunk_length_s=30,
        device=device,
    )
except Exception as e:
    print(f"Error initializing pipeline: {e}")
    exit()

# Set the forced decoder IDs if supported by the model
if hasattr(transcribe.model.config, 'forced_decoder_ids') and hasattr(transcribe.tokenizer, 'get_decoder_prompt_ids'):
    transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(language="te", task="transcribe")

# Perform transcription and print the result
try:
    result = transcribe(audio_path)
    print('Transcription:', result["text"])
except Exception as e:
    print(f"An error occurred during transcription: {e}")
