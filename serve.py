#!/usr/bin/env python3
# /// script
# dependencies = [
#   "msgpack",
#   "websockets",
#   "numpy",
#   "huggingface-hub",
#   "kroko_onnx @ git+https://github.com/kroko-ai/kroko-onnx",
# ]
# ///
import asyncio
import msgpack
import json
import math
import websockets
import kroko_onnx
import numpy as np
import traceback
import os
from huggingface_hub import hf_hub_download 
from time import time

def download_model_if_needed():
    model_filename = "Kroko-FR-Community-64-L-Streaming-001.data"
    model_path = os.path.join(os.path.dirname(__file__), model_filename)

    if not os.path.exists(model_path):
        print(f"Downloading model from Hugging Face...")
        model_path = hf_hub_download(
            repo_id="Banafo/Kroko-ASR",
            filename="Kroko-FR-Community-64-L-Streaming-001.data",
            repo_type="model"
        )
        print(f"Model downloaded to: {model_path}")
    return model_path

model_path = download_model_if_needed()
recognizer = kroko_onnx.OnlineRecognizer.from_transducer(
    model_path=model_path,
    num_threads=1,
    provider="cpu",
    sample_rate=16000,
    decoding_method="greedy_search",
    modeling_unit='',
    blank_penalty=0.0,
    enable_endpoint_detection=True,
    rule1_min_trailing_silence=2.4,
    rule2_min_trailing_silence=1.2,
    rule3_min_utterance_length=20.0,
)

async def send_json(websocket, data):
    await websocket.send(msgpack.packb(data))

async def unmute_stt(ws):
    await send_json(ws, {"type": "Ready"})
    s = recognizer.create_stream()
    try:
        last_result = []
        step = 0
        pause_prediction = 1.0
        t = 0
        while True:
            msg = await ws.recv()
            msg = msgpack.unpackb(msg)
            if msg['type'] == 'Audio':
                s.accept_waveform(16000, np.array(msg['pcm']))

                while recognizer.is_ready(s):
                    recognizer.decode_streams([s])

                result = json.loads(recognizer.get_result_as_json_string(s))
                result = result['elements']['words']

                new_words = result[len(last_result):]
                #print("new_words", new_words)
                for word in new_words:
                    await send_json(ws, {"type": "Word", "text": word['text'], "start_time": word['startedAt'] })
                last_result = result

                dt = len(msg['pcm']) / 16000.0
                t += dt
                # Target "no voice" in 0.5s. if few words, 1s if many words
                alpha = math.exp( - dt / 1.0 * math.log(1/0.6))
                if new_words:
                    if pause_prediction > 0:
                        pause_prediction = 0
                    elif pause_prediction > -1:
                        pause_prediction += -1 * alpha
                    if pause_prediction < -1:
                        pause_prediction = -1.0
                else:
                    pause_prediction = 1.0 * (1-alpha) + alpha * pause_prediction

                # unmute uses prs[2]
                if pause_prediction < 0:
                    prs = [-1, -1, 0]
                else:
                    prs = [-1, -1, pause_prediction]
                #print(t, "pause", pause_prediction)
                await send_json(ws, {"type":"Step", "step_idx": step, "prs": prs})

                step += 1
                


    except:
        print("Closing unmute stt websocket")
        traceback.print_exc()


async def transcribe(websocket):
    print(f"Client connected for transcription at {websocket.request.path}")
    if websocket.request.path == '/api/asr-streaming':
        return await unmute_stt(websocket)

    if websocket.request.path != '/transcribe':
        raise "meh"
    s = recognizer.create_stream()
    last_result = None
    try:
        while True:
            message = await websocket.recv()
            if isinstance(message, bytes):
                samples_int16 = np.frombuffer(message, dtype=np.int16)
                samples_float32 = samples_int16.astype(np.float32)
                samples_float32 = samples_float32 / 32768
                s.accept_waveform(16000, samples_float32)

                while recognizer.is_ready(s):
                    recognizer.decode_streams([s])
                
                result = recognizer.get_result(s)
                if result and result != last_result:
                    last_result = result
                    await websocket.send(json.dumps({"type":"partial","content":result}))
            elif isinstance(message, str):
                print("|", message)
                j = json.loads(message)
                if 'eof' in j and j['eof'] == 1:
                    s.input_finished()
                    while recognizer.is_ready(s):
                        recognizer.decode_streams([s])
                    final_result = recognizer.get_result(s)
                    await websocket.send(json.dumps({"type":"final","content":final_result}))
                    break
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")
    print("Finishing websocket")

async def main():
    async with websockets.serve(transcribe, "::", 8765):
        print("WebSocket server started on ws://localhost:8765")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
