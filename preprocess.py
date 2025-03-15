
import os
import pandas as pd
import pickle
from dataset.midi import MidiDataset

# Tempo are so wired that they can't be processed
__unsupporrted_songs = ['ChrisPotter_InASentimentalMood_FINAL.csv',
                        'EricDolphy_Serene_FINAL.csv',
                        'DavidMurray_ChelseaBridge_FINAL.csv'
                        ]

global_midi2note = None

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Choose dataset')
    parser.add_argument('--dataset',type=str,help='The directory where the dataset (.csv) is stored',default='csv/jazzvar_csv')
    return parser.parse_args()


def csv2seq(path):
    songs = {}
    columns = ['track','time','tipe','channel','note','velocity','duration','placeholder']
    for path,subdirs,files in os.walk(path):
        for file in files:
            if(file in __unsupporrted_songs):
                continue
            if file.endswith('.csv'):
                file_path = os.path.join(path,file)
                with open(file_path,'r') as f:
                    line = f.readline()
                
                # Normalize the time to 480
                division = int(line.split(",")[-1])
                scale = 480/division
                df = pd.read_csv(file_path,names=columns)
                
                # Filter the dataframe to only contain note_on and note_off
                df=df.loc[df['channel'].isin([' 0'])]
                df = df.loc[df['tipe'].isin([" Note_on_c", " Note_off_c"])]
                # Normalize the time
                df['time']=df['time'].apply(lambda x: round(int(x)*scale)) 
                # Convert midi to note
                df['note'] = df['note'].apply(lambda x:int(x))
                df['note'] = df['note'].apply(lambda x: global_midi2note[x])
                # Drop unnecessary columns
                df.loc[df['velocity']==" 0",'tipe'] = " Note_off_c"
                df.drop(columns=['track','channel','velocity'],inplace=True)
                # Calculate the duration and pause for each note
                df['duration'] = df['time'].diff()
                df['duration'] = df['duration'].shift(-1)
                if(df['duration']<0).any():
                    print(f"Negative duration in {file}")
                df['pause'] = df['duration'].shift(-1)
                df = df.loc[df['tipe'].isin([" Note_on_c"])]
                df.fillna(0,inplace=True)
                df['duration'] = df['duration'].apply(lambda x: int(x))
                df['pause'] = df['pause'].apply(lambda x: int(x))

                # Generate the sequence like "note,duration,pause" for each note
                seq="" 
                for _,row in df.iterrows():
                    seq+=row['note']+","+str(row['duration'])+","+str(row['pause'])+" "
                songs[file] = seq
                
    return songs

def create_dict(path):
    midi2note_df = pd.read_csv(path)
    # Remove the different notation for the same midi note, e.g. F#/Gb
    midi2note_df['note'] = midi2note_df['note'].apply(lambda x: x.split("/")[0])
    # Convert to dict
    midi2note = dict(zip(midi2note_df['midi'],midi2note_df['note']))
    # Also, create the reverse dict
    note2midi = dict(zip(midi2note_df['note'],midi2note_df['midi']))
    return midi2note,note2midi
    

if __name__ == "__main__":
    args = parse_args()
    
    # Get the midi to note and note to midi conversion dictionary
    global_midi2note,local_note2midi=create_dict('csv/conversion.csv')
    pickle.dump(global_midi2note,open('midi2note.pkl','wb'))
    pickle.dump(local_note2midi,open('note2midi.pkl','wb'))
    
    # Convert the csv files to sequence
    songs = csv2seq(args.dataset)
    seq = ''
    for index,song in songs.items():
        seq+=song
        
    # Get unique char mapping
    unique_chars = tuple(set(seq))
    id2char = dict(enumerate(unique_chars))
    char2id = {char:id for id,char in id2char.items()}
    
    # Save the dataset
    with open('dataset.txt','w') as f:
        f.write(seq)
    seq = [char2id[char] for char in seq]
    pickle.dump(seq,open('songs.pkl','wb'))
    pickle.dump(char2id,open('char2id.pkl','wb'))
    pickle.dump(id2char,open('id2char.pkl','wb'))

        