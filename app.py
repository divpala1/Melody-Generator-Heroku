# Imports
import json
import tensorflow.keras as keras
import numpy as np
import music21 as m21
from preprocessing import SEQUENCE_LENGTH, MAPPING_PATH

from flask import Flask, request, render_template, send_file

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

class MelodyGenerator:
    
    def __init__(self, model_path="model.h5"):
        
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)
        
        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)
            
        self._start_symbols = ["/"] * SEQUENCE_LENGTH
    
    def _sample_with_temperature(self, probabilities, temperature):
        predictions = np.log(probabilities) / temperature
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions))
        
        choices = range(len(probabilities))
        index = np.random.choice(choices, p=probabilities)
        
        return index
        
    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
        
        # Create seed with start symbols
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed
        
        # Map seed to integers
        seed = [self._mappings[symbol] for symbol in seed]
        
        for _ in range(num_steps):
            # Limit the seed to the max sequence length
            seed = seed[-max_sequence_length:]
            
            # One-hot encoding the seed
            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))
            # Adding extra-dimension because model.predict() expects a 3-d input. The onehot_seed's dimension currently is (max_sequence_length, vocabulary size)
            onehot_seed = onehot_seed[np.newaxis, ...]
            
            # Making prediction
            probabilities = self.model.predict(onehot_seed)[0]
            
            output_int = self._sample_with_temperature(probabilities, temperature)
            
            # Update the seed
            seed.append(output_int)
            
            # Map integers to our encoding
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]
            
            # Checking if we are at the end of the melody
            if output_symbol == "/":
                break
                
            # Updating the melody
            melody.append(output_symbol)
        
        return melody
    
    def save_melody(self, melody, step_duration=0.25, format="midi", file_name="melody.mid"):
        # Create a music21 stream
        stream = m21.stream.Stream()
        
        # Parse all the symbols in melody and create note/rest objects
        start_symbol = None
        step_counter = 1
        
        for i, symbol in enumerate(melody):
            # Case-1: Note/rest
            if symbol != "_" or i + 1 == len(melody):
                
                # Passing the first note/rest
                if start_symbol is not None:
                    
                    quarter_length_duration = step_duration * step_counter
                    
                    # Rest
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength = quarter_length_duration)
                    
                    # Note
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength = quarter_length_duration)
                   
                    stream.append(m21_event)
                    
                    # Resetting the step counter
                    step_counter = 1
                    
                start_symbol = symbol
            
            # Case-2: Prolongation sign "_"
            else:
                step_counter += 1
                
            
        # Write the m21 stream to a midi file
        stream.write(format, file_name)

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    mg = MelodyGenerator()
    seed = request.form.get("melody")
    melody = mg.generate_melody(seed, 500, SEQUENCE_LENGTH, 0.3)
    
    mg.save_melody(melody)
    
    melody = " ".join(melody)
    
    return render_template('index.html', predicted_melody='The generated melody: {}'.format(melody), can_download=True)

@app.route('/download')
def download_file():
    p = "GeneratedMusic/melody.mid"
    return send_file(p, as_attachment=True)

if __name__ == "__main__":
        app.run(debug=True)

