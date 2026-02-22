import json
from langdetect import detect, LangDetectException

def is_italian(text):
    try:
        return detect(text) == 'it'
    except LangDetectException:
        return False

def process_json_to_lyrics(json_file, output_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    lyrics_list = []
    total_songs = 0
    italian_songs = 0
    
    for line in lines:
        try:
            song_data = json.loads(line)
            total_songs += 1
            if 'lyrics' in song_data:
                lyrics = song_data['lyrics']
                # Replace newlines with spaces to keep each song on one line
                lyrics = lyrics.replace('\n', ' ')
                if is_italian(lyrics):
                    lyrics_list.append(lyrics)
                    italian_songs += 1
        except json.JSONDecodeError:
            print(f"Error decoding JSON: {line}")

    with open(output_file, 'w', encoding='utf-8') as f:
        for lyrics in lyrics_list:
            f.write(f"{lyrics}\n")

    print(f"Processed {total_songs} songs.")
    print(f"Found {italian_songs} Italian songs.")
    print(f"Filtered out {total_songs - italian_songs} non-Italian songs.")
    print(f"Italian lyrics saved to {output_file}")

# Usage
input_file = './data/the_italian_music_dataset.json'
output_file = './data/italian_lyrics.txt'
process_json_to_lyrics(input_file, output_file)