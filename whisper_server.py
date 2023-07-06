import os
import time
from flask import Flask, jsonify, request, redirect
import whisper
import re
import threading
import talkToGpt
import voicevox

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'m4a','mp3','wav'}
WHISPER_MODEL_NAME = 'small' # tiny, base, small, medium
WHISPER_DEVICE = 'cuda' # cpu, cuda

print('loading whisper model', WHISPER_MODEL_NAME, WHISPER_DEVICE)
whisper_model = whisper.load_model(WHISPER_MODEL_NAME, device=WHISPER_DEVICE)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app = Flask(__name__, static_url_path='/')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# rinnaを使用する際にコメントを解除してください
# app.config['TOKENIZER'] = talkToGpt.get_tokenizer()
# app.config['MODEL'] = talkToGpt.get_model()

lock = threading.Lock()

@app.route('/')
def index():
   return redirect('/index.html')

@app.route('/api/transcribe', methods=['POST'])
def transcribe():
   print('start transcribe')
   file = request.files['file']
   ext = file.filename.rsplit('.', 1)[1].lower()
   if ext and ext in ALLOWED_EXTENSIONS:
       filename = str(int(time.time())) + '.' + ext
       saved_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
       file.save(saved_filename)
       lock.acquire()
       result = whisper_model.transcribe(saved_filename, fp16=False, language='ja')
       lock.release()
       print(".\\" + saved_filename)
       os.remove(saved_filename)
       return result, 200
   result={'error':'something wrong'}
   return result, 400



prompt = "あなたは面接官です。まずあなたから質問を投げかけて面接を行ってください。" 
@app.route('/api/callgpt', methods=['POST'])
def callgpt():
   try:
      global prompt
      question = request.form.get('question')
      # rinnnaを使用する際にコメントを解除してください
      # answer = talkToGpt.talk_to_rinna(app.config['TOKENIZER'], app.config['MODEL'], question)
      
      # OpenAI APIを使用する際にコメントを解除してください
      lock.acquire()
      ret = talkToGpt.talk_to_gpt(prompt, question)
      lock.release()
      answer = ret[0]
      prompt = prompt + ret[1]
   except Exception as e:
      print(e)
      return {'error':'something wrong'}, 400
   return {'answer':answer}, 200


@app.route('/api/callvoicevox', methods=['POST'])
def callvoicevox():
   try:
      lock.acquire()
      talk_text = request.form.get('talk_text')
      output_filepath = voicevox.speak_voicevox(talk_text)
      lock.release()
   except Exception as e:
      print(e)
      return {'error':'something wrong'}, 400
   return {'output_filepath':output_filepath}, 200

app.run(host='localhost', port=9000)
