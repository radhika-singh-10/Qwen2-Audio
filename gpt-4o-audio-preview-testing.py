import os
import csv
import json
import base64
import aiohttp
import asyncio
import aiofiles
import requests
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import openai

def process_audio_with_gpt_4o(session, base64_encoded_audio, output_modalities, system_prompt, apikey, file_name):
    # url = "https://api.openai.com/v1/chat/completions"

    # headers = {
    #     "Content-Type": "application/json",
    #     "Authorization": f"Bearer {apikey}"
    # }

    # data = {
    #     "model": "gpt-4o-audio-preview",
    #     "modalities": output_modalities,
    #     "audio": {
    #         "voice": "alloy",
    #         "format": "mp3"
    #     },
    #     "messages": [
    #         {"role": "system", "content": system_prompt},
    #         {
    #             "role": "user",
    #             "content": {
    #                 "type": "input_audio",
    #                 "input_audio": {
    #                     "data": base64_encoded_audio,
    #                     "format": "mp3"
    #                 }
    #             }
    #         }
    #     ]
    # }

    # response = session.post(url, headers=headers, json=data)
    # if response.status_code == 200:
    #     return response.json()
    # else:
    #     print(f"Error {response.status_code}: {response.text}")
    #     return None
    openai.api_key=apikey
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4o-audio-preview",
            modalities=output_modalities,
            audio={"voice": "alloy", "format": "mp3"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64_encoded_audio,
                                "format": "mp3"
                            }
                        }
                    ]
                },
            ]
        )
        #print(completion.choices[0].message)
        return completion.choices[0].message
    
    except Exception as e:
        print(f"Error processing audio with GPT-4o for file {file_name}: {e}")
        return "Error in processing"


def process_file(session, file_path, modalities, apikey, csv_writer, processed_files):
    try:
        with open(file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
        base64_audio = base64.b64encode(audio_bytes).decode('utf-8')


        filename = os.path.basename(file_path)

        # Skip if file is already processed
        if filename in processed_files:
            print(f"Skipping already processed file: {filename}")
            return

        print("filename : ",filename)
        transcript_prompt = "Transcribe the following audio."
        print("Transcription Prompt ",transcript_prompt)
        transcription_response = process_audio_with_gpt_4o(session, base64_audio, modalities, transcript_prompt, apikey,filename)
        transcription_analysis = transcription_response.get("audio", {}).get("transcript", None)
        print(transcription_analysis)
        # (
        #     transcription_response['audio']['transcript']
        #     if transcription_response
        #     else "Error in transcription"
        # )
        #print(f"Transcription: {transcription_response}", transcription_analysis)
        toxicity_prompt = "Is the audio toxic? If yes, what kind of toxic class does this audio belong to?"
        print("Toxicity Prompt ",toxicity_prompt)
        toxicity_response = process_audio_with_gpt_4o(session, base64_audio, modalities, toxicity_prompt, apikey,filename)
        toxicity_analysis = toxicity_response.get('audio', {}).get('transcript',None)
        print(toxicity_analysis)
        # (
        #     toxicity_response['audio']['transcript']
        #     if toxicity_response
        #     else "Error in toxicity analysis"
        # )
        #print(f"Toxicity Analysis: {toxicity_response}" , toxicity_analysis)
        csv_writer.writerow([filename, transcription_analysis, toxicity_analysis])
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")


def process_folder(folder_path, modalities, apikey, session, csv_writer, processed_files):
    try:
        
        for filename in os.listdir(folder_path):
            print(filename)
            if filename.endswith(".mp3"):
                file_path = os.path.join(folder_path, filename)
                print(f"Processing file: {file_path}")
                process_file(session, file_path, modalities, apikey, csv_writer, processed_files)
    except Exception as e:
        print(f"Error processing folder {folder_path}: {e}")


def extract_toxic_content():
    try:
        # df = pd.read_csv(csv_file, encoding='latin1')
        # condition = df['Dataset'].str.strip().isin(['LJ Speech', 'MELD', 'Common Voice'])
        # filtered_df = df[condition]
        apikey = ""
        modalities = ["text","audio"]
        folder_path = "detoxy_pilot_data/detoxy_pilot_data"
        output_csv_path = "audio_analysis_results_detoxy.csv"
        processed_files = set()
        if os.path.exists(output_csv_path):
            with open(output_csv_path, mode="r", newline="") as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader, None)  
                for row in csv_reader:
                    if row:  
                        processed_files.add(row[0])
                        print(row)
                    print(row)
        with open(output_csv_path, mode="a", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            if csvfile.tell() == 0:
                csv_writer.writerow(["Filename", "Transcript Analysis", "Toxicity Analysis"])
            with requests.Session() as session:
                process_folder(folder_path, modalities, apikey, session, csv_writer, processed_files)
    except Exception as e:
        print(f"Error in extract_toxic_content: {e}")



def compute_online_metrics(csv_file):
    df = pd.read_csv(csv_file)
    ground_truth,predicted_label = df.iloc[:, -1],df.iloc[:, -2]
    print("accuracy score ",accuracy_score(ground_truth,predicted_label))
    print("recall score ",recall_score(ground_truth,predicted_label))
    print("f1 socre ",f1_score(ground_truth,predicted_label))
    print("precision score ",precision_score(ground_truth,predicted_label))

def select_random_true_positives(csv_file):
    df=pd.read_csv(csv_file)
    true_positive_data = df[df['CLASS']=='FP']
    with open('random_false_positves.csv', mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        print(writer)




def main():
    #select_random_true_positives("/home/rsingh57/models/detoxy_audio_analysis_results.csv")
    #compute_online_metrics("audio_analysis_results.csv")
    extract_toxic_content()
    

    

        
if __name__ == "__main__":
    main()
