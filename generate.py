import pickle
import pandas as pd
import torch
import numpy as np

line_1 = "0, 0, Header, 1, 1, 480\n"
line_2 = "1, 0, Start_track\n"
line_3 = "1, 0, Tempo, 500000\n"
line_4 = "1, 0, Time_signature, 4, 2, 24, 8\n"
line_5 = "1, 0, Key_signature, -4, \"major\"\n"
line_end = "0, 0, End_of_file"

class Generator():
    def __init__(self,char2id,id2char):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.char2id = pickle.load(open(char2id,'rb'))
        self.id2char = pickle.load(open(id2char,'rb'))

        

    def one_hot_encode(self,data,vocab_len):
        encoded = torch.zeros(1,len(data),vocab_len)
        for i,symbol in enumerate(data):
            encoded[0][i][int(symbol)] = 1
        return encoded
    def predict_next(self,model, ch,h,top_k):
        vocab_len = model.input_dim
        x = self.char2id[ch]
        x  = self.one_hot_encode([x],vocab_len)
        x = x.to(self.device)
        h = tuple([each.data for each in h])
        out,h = model(x,h)
        p = torch.nn.functional.softmax(out, dim=1).data

        if torch.cuda.is_available():
            p = p.cpu()
        p,top_char = p.topk(top_k)
        p = p.numpy().squeeze()
        top_char = top_char.numpy().squeeze()
        ret = np.random.choice(top_char, p=p/p.sum())
        return self.id2char[ret],h

    def generate_song(self,model,trigger,length,top_k):
        model.eval()
        chars = [ch for ch in trigger]
        h = model.init_hidden(1)
        for ch in chars:
            char,h = self.predict_next(model,ch,h,top_k)
        chars.append(char)
        for i in range(length):
            char,h = self.predict_next(model,chars[-1],h,top_k)
            chars.append(char)
        return ''.join(chars)

    def seq2csv(self,notes,path):
        # with open('test.txt','r') as f:
        #     notes = f.read()
        songs = notes.split(' ')
        note2midi = pickle.load(open('note2midi.pkl','rb'))
        # print(note2midi)
        notes = []
        duration = []
        pause = []
        for song in songs:
            if(song==''):
                continue
            song = song.split(',')
            if len(song)!=3:
                # print('LSTM generated song is note in the right format, delete this note')
                continue
            if(song[0] not in note2midi):
                # print('Note not found in note2midi, delete this note')
                continue
            try:
                a = int(song[1])
                b = int(song[2])
            except:
                # print('Duration is not an integer, delete this note')
                continue
            notes.append(" "+str(note2midi[song[0]]))
            duration.append(int(song[1]))
            pause.append(int(song[2]))
        df = pd.DataFrame({'note':notes,'duration':duration,'pause':pause})
        res = line_1+line_2+line_3+line_4+line_5
        on_time = 0
        for i,row in df.iterrows():
            off_time = row['duration']+on_time
            t1 = '1, '+str(on_time)+', Note_on_c, 0,'+row['note']+', 80\n'
            t2 = '1, '+str(off_time)+', Note_off_c, 0,'+row['note']+', 80\n'
            on_time = off_time+row['pause']
            res+=t1+t2
        res += '1, '+str(on_time+1)+', End_track\n'
        res+=line_end

        with open(path,'w') as f:
            f.write(res)


    