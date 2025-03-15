from torch import nn

class LSTM(nn.Module):
    def __init__(self,input_dim, hidden_dim, layer_dim, drop_prob, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, dropout=drop_prob,batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        
    def forward(self, x, hidden):
        r_output,hidden = self.lstm(x,hidden)
        out = self.dropout(r_output)
        out = out.contiguous().view(-1,self.hidden_dim)
        out = self.fc(out)
        return out,hidden

    def init_hidden(self,batch_size):
        weight = next(self.parameters()).data
        
        hidden = (weight.new(self.layer_dim,batch_size,self.hidden_dim).zero_(),
                  weight.new(self.layer_dim,batch_size,self.hidden_dim).zero_())
        return hidden
    
    
