import requests
import json
import time

def speak_voicevox(input_texts):
    speaker     = 14 # speaker id
    filename    = str(int(time.time()))
    output_path = "./static/voicevox_output"

    # audio_query (音声合成用のクエリを作成するAPI)
    res1 = requests.post('http://localhost:50021/audio_query',
                        params={'text': input_texts, 'speaker': speaker})
    # synthesis (音声合成するAPI)
    res2 = requests.post('http://localhost:50021/synthesis',
                        params={'speaker': speaker},
                        data=json.dumps(res1.json()))
    # wavファイルに書き込み
    filepath = output_path + '/' + filename + '.wav'
    with open(filepath, mode='wb') as f:
        f.write(res2.content)
    filepath = filepath.replace("./static","")
    return filepath