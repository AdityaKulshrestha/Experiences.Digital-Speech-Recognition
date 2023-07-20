import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline


def whisper_transcription(audio_array):
    processor = WhisperProcessor.from_pretrained("openai/whisper-base.en")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en")
    input_features = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features

    # generate token ids
    predicted_ids = model.generate(input_features)
    # decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    print(transcription)
    return transcription[0]


def large_audio_transcription(audio):
    model = pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-base.en",
        device=torch.device("cuda:0"),
        generate_kwargs={"num_beams": 5},  # same setting as `openai-whisper` default
        chunk_length_s=30,
        stride_length_s=(4, 2)
    )
    output = model(audio)
    return output['text']


def transcribe_marathi(audiofile: str, lang: str = 'marathi') -> str:
    model = pipeline(
        task="automatic-speech-recognition",
        model="Aditya02/Vistar_Marathi_Model",
        device=torch.device("cuda:0"),
        # chunk_length_s=30, # if not precised then only generate as much as `max_new_tokens`
        generate_kwargs={"num_beams": 5},  # same setting as `openai-whisper` default
        chunk_length_s=30,
        stride_length_s=(4, 2)
    )

    result = model(audiofile, return_timestamps=True)

    return result["text"]
