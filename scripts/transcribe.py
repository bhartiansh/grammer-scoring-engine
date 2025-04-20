import os
import pandas as pd
import whisper
from tqdm import tqdm

def transcribe_audio(input_dir, csv_file, output_csv):
    model = whisper.load_model("base")
    df = pd.read_csv(csv_file)
    transcripts = []

    for filename in tqdm(df["filename"], desc="Transcribing"):
        audio_path = os.path.join(input_dir, filename)
        if not os.path.exists(audio_path):
            print(f"[!] File not found: {filename}")
            continue
        try:
            result = model.transcribe(audio_path)
            transcripts.append({"filename": filename, "transcript": result["text"].strip()})
        except Exception as e:
            print(f"[!] Error processing {filename}: {e}")
            transcripts.append({"filename": filename, "transcript": None})

    transcript_df = pd.DataFrame(transcripts)
    transcript_df.to_csv(output_csv, index=False)
    print(f"[✓] Transcriptions saved to {output_csv}")
