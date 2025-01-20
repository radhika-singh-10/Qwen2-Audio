import csv
import os
import librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from argparse import ArgumentParser

DEFAULT_CKPT_PATH = 'Qwen/Qwen2-Audio-7B-Instruct'

def _get_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, default=DEFAULT_CKPT_PATH,
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("--cpu-only", action="store_true", help="Run demo with CPU only")
    parser.add_argument("--audio-folder", type=str, required=True, default="./detoxy_test_data",
                        help="Path to the folder containing audio files to process.")
    parser.add_argument("--output-csv", type=str, default="qwen2audio_results.csv",
                        help="Path to the output CSV file.")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt to guide the audio processing.", 
                        default="Is the audio toxic? If yes, what kind of toxic class does this audio belong to?")
    return parser.parse_args()

def process_audio_files(audio_folder, output_csv, prompt, model, processor, device):
    """Process audio files in a folder and write results to a CSV file."""
    with open(output_csv, mode="a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if csv_file.tell() == 0:
            writer.writerow(["Audio File", "Prompt", "Response"])

        for audio_file in os.listdir(audio_folder):
            if audio_file.lower().endswith(('.wav', '.mp3')):
                audio_path = os.path.join(audio_folder, audio_file)
                print(f"Processing: {audio_path}")

                # Load the audio file
                audio, _ = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)

                # Prepare inputs
                inputs = processor(text=prompt, audios=[audio], return_tensors="pt", padding=True)
                if not device == "cpu":
                    inputs["input_ids"] = inputs.input_ids.to(device)

                # Generate response
                generate_ids = model.generate(**inputs, max_length=256)
                generate_ids = generate_ids[:, inputs.input_ids.size(1):]
                response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

                # Write to CSV
                writer.writerow([audio_file, prompt, response])

if __name__ == "__main__":
    args = _get_args()

    device = "cpu" if args.cpu_only else "cuda"

    # Load model and processor
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.checkpoint_path,
        torch_dtype="auto",
        device_map="auto" if device != "cpu" else None,
        resume_download=True
    ).eval()
    processor = AutoProcessor.from_pretrained(args.checkpoint_path, resume_download=True)

    # Process audio files
    process_audio_files(args.audio_folder, args.output_csv, args.prompt, model, processor, device)

    print(f"Processing complete. Results saved to {args.output_csv}")
