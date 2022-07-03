# Imports
import os
# Music21 enables us to manipulate symbolic music data efficiently, and helps in the conversion of music files to a specified format.
import music21 as m21
import json
import tensorflow.keras as keras
import numpy as np


# ##### In-case the path to MuseScore is not set, use the code below to do so for making functionalities like `m21.show()` work.
# us = m21.environment.UserSettings()
# us['musicxmlPath'] = 'C:\\Program Files\\MuseScore 3\\bin\\MuseScore3.exe'
# us['musescoreDirectPNGPath'] = 'C:\\Program Files\\MuseScore 3\\bin\\MuseScore3.exe'
# us['musicxmlPath']

# Constants
KERN_DATASET_PATH = "Kern_Dataset/erk"
SAVE_DIR = "Dataset"
SINGLE_FILE_DATASET = "Dataset_file"
MAPPING_PATH = "Mapping.json"
# Length of sequence being passed to the LSTM
SEQUENCE_LENGTH = 64

# Expressed in quarter-note length. Here, 1 = 1 quarter-note value.
ACCEPTABLE_DURATIONS = [
    # Here we use only the most common notes, so as to simplify everything, and make it easier for the model to recognize the patterns.
    0.25, # Sixteenth-note
    0.5, # Eighth-note
    0.75,
    1,
    1.5,
    2,
    3,
    4
]

# Loading the folk songs in kern
def load_songs_in_kern(dataset_path):
    # Go through all the files in the dataset and load them with music21.
    
    songs = []
    
    # 'os.walk()' goes through all the files and folders recursively given the parent folder.
    for path, subdir, files in os.walk(dataset_path):
        for file in files:
            # Filtering the kern files out of all the files. Done by checking the last three letters i.e., the extension.
            if file[-3:] == "krn":
                song =  m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs

# Checking whether all the notes and rests of the song are acceptable or not.
def has_acceptable_durations(song, acceptable_durations):
    # '.flat' function of music21 flattens the list, and 'notesAndRests' returns only the notes and rests out of the flattened list.
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in ACCEPTABLE_DURATIONS:
            return False
    return True


# The reason for transposing the songs to C-major or A-minor is so that we can learn patterns from a smaller dataset too. For learning in all the 24 keys, larger dataset will be required and will be computationally way too expensive.

# Transpose songs to C-major if in major mode, and in A-minor if in minor mode.
def transpose(song):
    # Get the key from the song, if mentioned.
    parts = song.getElementsByClass(m21.stream.Part) # Parts is analogous to bars of the song.
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]
    
    # Estimate the key of the song using music21, if not mentioned in the song.
    if not isinstance(key, m21.key.Key):
        # If the key retrieved is not a key-object of music21:
        key = song.analyze("key")
        
    # Get the interval of transposition. E.g., If the song has the key of B-major, calculating distance between B-major and C-major(target key).
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
        
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))
    
    # Transposing the song by the calculated interval.
    transposed_song = song.transpose(interval)
    
    return transposed_song


# ### Encoding the song
# Input will be a song as a music21 object and it will return the song as a string which will be the song encoded in time series representation. 
# 
# For example, suppose a song has pitch = 60 (middle C), and duration = 1 (in quarter notes). So this will be encoded as a list initially where each item corresponds to a **sixteenth note**. 
# 
# So list = \[60, _, _, _ \] where the first element is the MIDI note for the pitch and the underscore denotes that the note is held for that specific timestamp.
# 
# The duration adds to 1 as each note is 16th note and 4 of those makes a quarter-note which is equal to the duration of 1. So setting time_step=0.25, defaults it to a sixteenth note.

def encode_song(song, time_step=0.25):
    
    encoded_song = []
    
    for event in song.flat.notesAndRests:
        # Handling notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi
        
        # Handling rests
        elif isinstance(event, m21.note.Rest):
            symbol = "r"
            
        # Converting to time-series representation
        # Casted to int because a for-loop needs to be run for number of steps.
        steps = int(event.duration.quarterLength / time_step)
        
        # We need either the MIDI number or the symbol 'r', for the *first* step. Then for the note being held, we want underscore.
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")
                
    # Converting encoded song to string. 'map()' will convert individual elements to strings, if not already.
    encoded_song = " ".join(map(str, encoded_song))
    
    return encoded_song

def preprocess(dataset_path):
    # Loading the folk songs.
    print("Loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.")
    
    for i, song in enumerate(songs):
        # Filter out the songs that don't have acceptable durations.
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            # To make an if statement test if something didn’t happen, you can put the not operator in front of the condition at hand.
            continue # Skip the song if it does not have acceptable notes and rests.
        
        # Transposing songs to C-major/A-minor
        song = transpose(song)
        
        # Encoding the song to time-series representation
        encoded_song = encode_song(song)
        
        # Save the song to a txt file
        save_path = os.path.join(SAVE_DIR, str(i))
        
        with open(save_path, "w") as fp:
            fp.write(encoded_song)


# ### Merging separate string music files into one file

# Utility function for the function below
def load(file_path):
    with open(file_path, "r") as fp:
        song = fp.read()
    return song

# Function to merge all the different string music files into one file.
def create_single_file_dataset(dataset_path, dataset_file_path, sequence_length):
    # Having number of slashes equal to the sequence length
    new_song_delimeter = "/ " * sequence_length
    songs = ""
    
    # Load encoded songs and add delimeters
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs = songs + song + " " + new_song_delimeter
    
    # As the new song delimeter is sequence length number of slashes each followed by a space, we don't want space after the last slash, hence slicing the string.
    songs = songs[:-1]
    
    with open(dataset_file_path, "w") as fp:
        fp.write(songs)
        
    return songs


# ### Mapping function
# The songs in form of strings have some values other than integers like "_" and "r". A neural network can only work with numbers. Hence this function will map the non-integer values to an integer value which can be interpreted by the neural network.

def create_mapping(songs, mapping_path):
    mappings = {}
    
    # Identify the vocabulary
    songs = songs.split()
    vocab = list(set(songs))
    
    # Creating mappings
    for i, symbol in enumerate(vocab):
        mappings[symbol] = i
    
    # Saving the vocabulary to a JSON file
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent=4) # 'indent=4' makes the file more readable by not storing everything in one line.

def convert_songs_to_int(songs):
    int_songs = []
    
    # Load mappings
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)
        
    # Cast the songs string to a list
    songs = songs.split()
    
    # Mapping songs to int
    for symbol in songs:
        int_songs.append(mappings[symbol])
        
    return int_songs

def generate_training_sequences(sequence_length):
    # Load songs and map them to integers
    songs = load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)
    
    # Generate the training sequences
    inputs = []
    targets = []
    
    # Number of sequences that can be made from a given number of songs
    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])
    
    # One-hot encoding the sequences
    vocabulary = len(set(int_songs))
    # Dimensions of the inputs will be (Number of sequences, sequence length, vocabulary size)
    inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary)
    targets = np.array(targets)
    
    return inputs, targets

def main():
    preprocess(KERN_DATASET_PATH)
    songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
    create_mapping(songs, MAPPING_PATH)
#     inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)

if __name__ == "__main__":
    main()