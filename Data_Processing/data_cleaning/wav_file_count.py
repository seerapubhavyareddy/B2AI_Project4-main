import os

def count_wav_files_with_word(directory, word):
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.wav') and word.lower() in file.lower():
                count += 1
    return count

def count_wav_files(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.wav'):
                count += 1
    return count



# Replace this with the path to your folder
# folder_path = "/home/b/bhavyareddyseerapu/silence_removal/filtered_stridor_data"
# folder_path = "/home/b/bhavyareddyseerapu/silence_removal/filtered_b2ai_data"
# folder_path = "/home/b/bhavyareddyseerapu/filtered_stridor_data"
# folder_path = "/home/b/bhavyareddyseerapu/filtered_b2ai_data"
folder_path = "/home/b/bhavyareddyseerapu/chunk_data/chunk_b2ai"
folder_path = "/home/b/bhavyareddyseerapu/chunk_data/chunk_b2ai"


total_wav_files = count_wav_files(folder_path)
print(f"Total number of .wav files: {total_wav_files}")

search_words = ["fimo", "rp", "reg", "deep", "rmo"]  # Add all words you want to search for

for word in search_words:
    count = count_wav_files_with_word(folder_path, word)
    print(f"Total number of .wav files containing '{word}': {count}")