import torch
import pickle

from dataset.midi import MidiDataset
from model.LSTM import LSTM

import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--data_dir',type=str,help='The directory where the data is stored',default='./')
    parser.add_argument('--max_seq',type=int,help='The maximum sequence length',default=50)
    parser.add_argument('--batch_size',type=int,help='The batch size',default=512)
    parser.add_argument('--n_epoch',type=int,help='The number of epochs',default=200)
    parser.add_argument('--clip',type=int,help='The gradient clipping value',default=5)
    parser.add_argument('--learning_rate',type=float,help='The learning rate',default=1e-3)
    parser.add_argument('--print_every',type=int,help='Print every n steps',default=10)
    parser.add_argument('--save_every',type=int,help='Save every n steps',default=500)
    return parser.parse_args()


def train(model,train_set,test_set,batch_size,n_epoch,lr,clip,print_every,save_every):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    citerion = torch.nn.CrossEntropyLoss()
    
    train = torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=False,drop_last=True)
    test = torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=True,drop_last=True)
    
    counter = 0
    loss_history = []
    eval_loss_history = []
    for i in range(n_epoch):
        for x,y in train:
            counter+=1
            
            model.train()
            h = model.init_hidden(batch_size)
            x = x.to(device)
            y = y.to(device)
            h = tuple([each.data.to(device) for each in h])
            
            optimizer.zero_grad()
            output,h= model(x,h)
            loss = citerion(output,y.view(y.shape[0]*y.shape[1],-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),clip)
            optimizer.step()
            if(counter%print_every==0):
                val_h = model.init_hidden(batch_size)
                val_h = tuple([each.data.to(device) for each in val_h])
                model.eval()
                for x,y in test:
                    x = x.to(device)
                    y = y.to(device)
                    output,val_h = model(x,val_h)
                    eval_loss = citerion(output,y.view(y.shape[0]*y.shape[1],-1))
                loss_history.append(loss.item())
                eval_loss_history.append(eval_loss.item())
                print(f"Epoch: {i+1}/{n_epoch} - Step: {counter} - train loss: {loss.item()} - eval loss:{eval_loss.item()}")
            if(counter%save_every==0):
                torch.save(model.state_dict(),f'model/checkpoint_{counter}.pth')
    torch.save(model.state_dict(),'model/model_lstm_latest.pth')
    pickle.dump(loss_history,open('loss_history.pkl','wb'))
    pickle.dump(eval_loss_history,open('eval_loss_history.pkl','wb'))

def getXAndY(data,max_seq):
    x = []
    y = []
    for i in range(0,len(data)-max_seq,max_seq):
        x.append(data[i:i+max_seq])
        y.append(data[i+1:i+max_seq])
        y[-1].append(x[-1][0])
    x = np.array(x)
    y = np.array(y)
    return x,y

if __name__ == "__main__":
    # Parse the arguments
    args = parse_args()
    
    # Load the dataset
    dataset = MidiDataset(args.data_dir,getXAndY,args.max_seq)
    train_set, test_set = torch.utils.data.random_split(dataset,
                                                        lengths=[0.8,0.2])
    
    # Train the model
    vocab_len = len(pickle.load(open('char2id.pkl','rb')))
    model = LSTM(input_dim=vocab_len,
                 hidden_dim=512,
                 layer_dim=2,
                 drop_prob=0.5,
                 output_dim=vocab_len)
    
    train(model,train_set,test_set,
          args.batch_size,args.n_epoch,
          args.learning_rate,args.clip,
          args.print_every,args.save_every)
    