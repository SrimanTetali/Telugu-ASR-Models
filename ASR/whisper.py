import torch
from transformers import pipeline

# Path to the audio file to be transcribed
audio = "/home/srimantetali/Sriman/projects/Kid_imp_of_trees.mp3"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# model_name = "vasista22/whisper-telugu-tiny"
# model_name = "vasista22/whisper-telugu-small"
# model_name = "vasista22/whisper-telugu-base"
# model_name = "vasista22/whisper-telugu-medium"
# model_name = "vasista22/whisper-telugu-large-v2"
model_name = "kowshik/whisper-telugu-medium"
# model_name = "kowshik/whisper-telugu-large-v2"
# model_name = "Mukund017/whisper-small-telugu"
# model_name = "eswardivi/whisper-tiny-fluers_V2_telugu_Augmentation_full_datset_V2_e5"
# model_name = "Anujgr8/Whisper-Anuj-small-Telugu-final"


# Initialize the ASR pipeline
transcribe = pipeline(
    task = "automatic-speech-recognition", 
    model = model_name, 
    chunk_length_s=30, 
    device=device,
)

# Set the forced decoder IDs for Telugu transcription
transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(language="te", task="transcribe")

# Perform transcription and print the result
print('Transcription: ', transcribe(audio)["text"])
