
import os
import whisper
from pyannote.audio import Pipeline
from pyannote.core import Segment
import csv
import json

# Set up Hugging Face Token
HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")

# Define helper functions
def get_text_with_timestamp(transcribe_res):
    timestamp_texts = []
    for item in transcribe_res['segments']:
        start = item['start']
        end = item['end']
        text = item['text']
        timestamp_texts.append((Segment(start, end), text))
    return timestamp_texts

def add_speaker_info_to_text(timestamp_texts, ann):
    spk_text = []
    for seg, text in timestamp_texts:
        spk = ann.crop(seg).argmax()
        spk_text.append((seg, spk, text))
    return spk_text

def merge_cache(text_cache):
    sentence = ''.join([item[-1] for item in text_cache])
    spk = text_cache[0][1]
    start = text_cache[0][0].start
    end = text_cache[-1][0].end
    return Segment(start, end), spk, sentence

PUNC_SENT_END = ['.', '?', '!']

def merge_sentence(spk_text):
    merged_spk_text = []
    pre_spk = None
    text_cache = []
    for seg, spk, text in spk_text:
        if spk != pre_spk and pre_spk is not None and len(text_cache) > 0:
            merged_spk_text.append(merge_cache(text_cache))
            text_cache = [(seg, spk, text)]
            pre_spk = spk
        elif text and len(text) > 0 and text[-1] in PUNC_SENT_END:
            text_cache.append((seg, spk, text))
            merged_spk_text.append(merge_cache(text_cache))
            text_cache = []
            pre_spk = spk
        else:
            text_cache.append((seg, spk, text))
            pre_spk = spk
    if len(text_cache) > 0:
        merged_spk_text.append(merge_cache(text_cache))
    return merged_spk_text


def diarize_text(transcribe_res, diarization_result):
    timestamp_texts = get_text_with_timestamp(transcribe_res)
    spk_text = add_speaker_info_to_text(timestamp_texts, diarization_result)
    res_processed = merge_sentence(spk_text)
    return res_processed

def process_audio_file(audio_file_path, pipeline, model):
    print(f"Processing file: {audio_file_path}")

    try:
        print("Processing audio with Whisper ASR model...")
        asr_result = model.transcribe(audio_file_path)

        print("Performing speaker diarization...")
        diarization_result = pipeline(audio_file_path)

        print("Merging transcription and speaker diarization results...")
        final_result = diarize_text(asr_result, diarization_result)

        # Prepare the response
        response = []
        for seg, spk, sent in final_result:
            response.append({
                "start_time": f'{seg.start:.2f}',
                "end_time": f'{seg.end:.2f}',
                "speaker": spk,
                "transcript": sent
            })

        print("Processing complete")
        return response

    except Exception as e:
        print(f"Error processing {audio_file_path}: {e}")
        return None

def main():
    # Define and create model directory if it doesn't exist
    model_dir = "./pretrained_models"
    os.makedirs(model_dir, exist_ok=True)

    print("Loading models...")
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                            use_auth_token=HF_AUTH_TOKEN,
                                            cache_dir=model_dir)
    except Exception as e:
        print(f"Error loading Pyannote model: {e}")
        print("Falling back to default cache location for Pyannote.")
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                            use_auth_token=HF_AUTH_TOKEN)

    try:
        model = whisper.load_model("tiny.en", download_root=model_dir)
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        print("Falling back to default cache location for Whisper.")
        model = whisper.load_model("tiny.en")

    print("Models loaded successfully!")

    # Define input and output directories
    input_dir = "./input_audio"
    output_dir = "./transcriptions"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process all audio files in the input directory
    for filename in os.listdir(input_dir):

        if filename.lower().endswith((".wav", ".mp3")):  # Add or remove audio formats as needed
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_text.csv")

            result = process_audio_file(input_path, pipeline, model)

            if result:
                with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = ['start_time', 'end_time', 'speaker', 'transcript']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                    writer.writeheader()
                    for row in result:
                        writer.writerow(row)

                print(f"Processed results saved to: {output_path}")
            else:
                print(f"Failed to process: {filename}")

if __name__ == "__main__":
    main()
