import uvicorn
import librosa
from fastapi import FastAPI, File, UploadFile
from utils import whisper_transcription, large_audio_transcription, transcribe_marathi

app = FastAPI()


@app.get('/')
def index():
    return {'message': 'Hello, stranger'}


@app.post("/marathi")
def create_upload_file(file: bytes = File()):
    with open('audio.mp3', 'wb') as f:
        f.write(file)
        f.close()
    return {'filename': transcribe_marathi("audio.mp3")}


@app.post("/english")
def create_upload(file: UploadFile):
    audio_data, sample_rate = librosa.load(file.file)
    output = whisper_transcription(audio_data)
    # model = whisper.load_model('base', download_root='models_whisper/', device="cuda")
    # transcription = model.transcribe(file.filename, temperature=0)
    complete_text = large_audio_transcription(audio_data)
    return {'complete transcription': complete_text}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)


