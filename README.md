# ml_music_generator
A charRNN based music generator using PyTorch for CS 205 final project.

## Structure
- `csv/`: contains the data used for training the model. These files are converted from MIDI files by using [midicsv](https://www.fourmilab.ch/webtools/midicsv/) tool.
    - [Jazzvar Dataset](https://zenodo.org/records/11264676)
    - [Jazzomat Dataset](https://jazzomat.hfm-weimar.de/dbformat/dboverview.html)
- `dataset/`: contains the dataset class for the music generator.
- `model/`: contains the model class for the music generator.
- `pretrained/`: contains the pretrained model we've provided.
