import torch
import torch.nn as nn
import torch.nn.functional as F


# MODELS WITHOUT EXTRENAL DATA


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob, device="cpu"):
        super(LSTMModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.device=device

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob, device=self.device
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, h0=None, c0=None, base_val=None, pprint=False, stochastic=False):
        if h0 is None:
            # Initializing hidden state for first input with zeros
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=self.device)#.requires_grad_()

        if c0 is None:
            # Initializing cell state for first input with zeros
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=self.device)#.requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.lstm(x, (h0, c0))# (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        if pprint:
            print(out)
            print(out + base_val.reshape(-1,1))

        if base_val is None:
            return out #+ x[:,-1,0].reshape(-1,1)
        else:
            return out + base_val.reshape(-1,1)



class SALSTM4(nn.Module):
    def __init__(self, input_dim, seq_length, hidden_dim, layer_dim, output_dim, dropout_prob, attention_mode="A", device="cpu"):
        super().__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.device=device
        self.seq_length = seq_length
        self.attention_mode = attention_mode

        self.attention_layer = nn.Linear(self.hidden_dim, 1, bias=False)
        self.mixed_att_weigths = torch.nn.Parameter(torch.randn(1,2))
        torch.nn.init.xavier_uniform_(self.mixed_att_weigths)
        self.dropout = nn.Dropout(p=dropout_prob)

        # LSTM layers
        if layer_dim>1:
            self.lstm = nn.LSTM(
                input_dim, hidden_dim, layer_dim, batch_first=True, device=self.device, dropout=dropout_prob
            )
        else:
            self.lstm = nn.LSTM(
                input_dim, hidden_dim, layer_dim, batch_first=True, device=self.device
            )

        # Fully connected layer
        self.mean_fc1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.mean_mid = nn.Linear(hidden_dim//2, hidden_dim//2, bias=False)
        self.mean_fc2 = nn.Linear(hidden_dim//2, 1, bias=False)

        self.std_fc1 = nn.Linear(hidden_dim, hidden_dim//2, bias=False)
        self.std_mid = nn.Linear(hidden_dim//2, hidden_dim//2, bias=False)
        self.std_fc2 = nn.Linear(hidden_dim//2, 1, bias=False)

        self.dev_fc1 = nn.Linear(hidden_dim+2, hidden_dim//2+1)
        self.dev_fc2 = nn.Linear(hidden_dim//2+1, 1)
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout_prob)


    def forward(self, x, h0=None, c0=None, base_val=None, pprint=False, stochastic=True, perc=False, retstd=False):

        if pprint:
            print("\n####")

        x = x.clone()
        maxx = x.abs().max(dim=1)[0].flatten()
        nx = x[:,:,0:2]/(maxx[0:2]+0.000001)
        x[:,:,0:2] = nx


        if h0 is None:
            # Initializing hidden state for first input with zeros
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=self.device)#.requires_grad_()

        if c0 is None:
            # Initializing cell state for first input with zeros
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=self.device)#.requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.lstm(x, (h0, c0))# (h0.detach(), c0.detach()))


        if self.attention_mode == "None":
            out = self.dropout(out[:, -1, :])                               # NO ATTENTION
        elif self.attention_mode == "A":
            att_out, alphat = self.attention(out)
            out = self.dropout(att_out)                                     # FULL ATTENTION 
            if pprint:
                print(f"alphat: ",alphat)
        elif self.attention_mode == "MA":
            if pprint:
                print(f"alphat: ",alphat)
            att_out, alphat = self.attention(out)
            out = 0.5*self.dropout(att_out) +0.5*out[:, -1, :]              # MIXED ATTENTION
        
        elif self.attention_mode == "TMA":
            att_out, alphat = self.attention(out)
            maw = F.softmax(self.mixed_att_weigths, dim=1)                  # mixed attention weights
            if pprint:
                print(f"alphat: ",alphat)
                print("MAW:", maw)
            out = maw[0,0]*self.dropout(att_out) +maw[0,1]*out[:, -1, :]    # TRAINABLE MIXED ATTENTION
        out = torch.relu(out) 
    
        out_m = self.mean_fc1(out)
        out_m = torch.relu(out_m)
        out_m = torch.tanh(self.mean_mid(out_m))
        out_m = self.dropout(out_m)
        out_m = self.mean_fc2(out_m)

        out_std = self.std_fc1(out)
        out_std = torch.relu(out_std)
        out_std = torch.tanh(self.std_mid(out_std))
        out_std = self.dropout(out_std)
        out_std = self.std_fc2(out_std)
        out_std = torch.exp(out_std)
        
        out_dev = self.dev_fc1(torch.concat((out, out_m, out_std), dim=-1))
        out_dev = torch.tanh(out_dev) #relu
        out_dev = self.dropout(out_dev)
        out_dev = self.dev_fc2(out_dev)
        #out_dev = torch.abs(out_dev)
        out_dev = torch.exp(out_dev/4)
       
        out = self.fc(out)

        if stochastic:
            eps = torch.normal(mean=torch.zeros_like(out, device=out.device), std=torch.ones_like(out,device=out.device))
        else:
            eps = torch.zeros_like(out, device=out.device)
        randout = torch.einsum("ij,ij -> ij", eps, out_std) + out_m

        if perc and base_val is not None:
            #out = torch.tanh(randout)*(out_dev)**2*base_val/maxx[0]*0.05
            out = torch.tanh(randout)*(out_dev)*base_val/maxx[0]*0.05
        else:
            out = torch.tanh(randout)*(1+out_dev)**2

        if pprint:
            print("eps ", eps)
            print("mean ", out_m)
            print("std ",out_std)
            print("dev ",out_dev)
            print("r-out ",randout)
            print()
            print("d-out ", out)
            print("out ",out + base_val.reshape(-1,1)) #+ x[:,-1,0].reshape(-1,1))


        out = out*maxx[0]

        if base_val is None:
            base_val = 0
        else: 
            base_val = base_val.reshape(-1,1)

        if retstd:
            return out + base_val, out_std
        else:
            return out + base_val


    
    def attention(self, x):
        et = self.attention_layer(x)
        et = torch.tanh(et).squeeze(2)
        alphat = torch.softmax(et,dim=1)
        att_output = torch.einsum("ij, ijk -> ik",alphat,x)
        return att_output, alphat



class ALSTM(nn.Module):
    def __init__(self, input_dim, seq_length, hidden_dim, layer_dim, output_dim, dropout_prob, device="cpu"):
        super(ALSTM, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.device=device
        self.seq_length = seq_length

        self.attention_layer = nn.Linear(self.hidden_dim,1, bias=False)
        self.dropout = nn.Dropout(p=dropout_prob)

        # LSTM layers
        if layer_dim>1:
            self.lstm = nn.LSTM(
                input_dim, hidden_dim, layer_dim, batch_first=True, device=self.device, dropout=dropout_prob
            )
        else:
            self.lstm = nn.LSTM(
                input_dim, hidden_dim, layer_dim, batch_first=True, device=self.device
            )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.out_act = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # batchnorm on seq length insetad of mini batches :v
        # we'd like to learn a local normalization
        self.batchnorm = nn.BatchNorm1d(1) #affine=False


    def forward(self, x, h0=None, c0=None, base_val=None, pprint=False, stochastic=False):

        if pprint:
            print("\n####")

        x = self.batchnorm(x.transpose(0,1)).transpose(1,0)
            
        if h0 is None:
            # Initializing hidden state for first input with zeros
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=self.device)#.requires_grad_()

        if c0 is None:
            # Initializing cell state for first input with zeros
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=self.device)#.requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.lstm(x, (h0, c0))# (h0.detach(), c0.detach()))

        att_out, alphat = self.attention(out)
        if pprint:
            print(alphat)
        out = self.dropout(att_out)
        out = torch.relu(out) 
        out = self.fc2(out)
        if pprint:
            print(out)
            print(out + base_val.reshape(-1,1)) #+ x[:,-1,0].reshape(-1,1))


        if base_val is None:
            return out
        else:
            return out + base_val.reshape(-1,1)

    
    def attention(self, x):
        et = self.attention_layer(x)
        et = torch.tanh(et).squeeze(2)
        alphat = torch.softmax(et,dim=1)
        att_output = torch.einsum("ij, ijk -> ik",alphat,x)
        return att_output, alphat



class PaperModel(nn.Module):
    def __init__(self, input_dim, seq_length, hidden_dim, layer_dim, output_dim, dropout_prob, device="cpu"):
        super().__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.device=device
        self.seq_length = seq_length
        self.input_dim = input_dim

        self.embedd = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim)
        ) 

        self.attention_layer = nn.Linear(self.hidden_dim,1, bias=False)
        self.dropout = nn.Dropout(p=dropout_prob)

        # LSTM layers
        if layer_dim>1:
            self.lstm = nn.LSTM(
                input_dim, hidden_dim, layer_dim, batch_first=True, device=self.device, dropout=dropout_prob
            )
        else:
            self.lstm = nn.LSTM(
                hidden_dim, hidden_dim, layer_dim, batch_first=True, device=self.device
            )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # batchnorm 
        self.batchnorm = nn.BatchNorm1d(1) #affine=False


    def forward(self, x, h0=None, c0=None, base_val=None, pprint=False, stochastic=False):


        #x = self.batchnorm(x.transpose(0,1)).transpose(1,0)
        x = self.embedd(x)

        if h0 is None:
            # Initializing hidden state for first input with zeros
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=self.device)

        if c0 is None:
            # Initializing cell state for first input with zeros
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=self.device)


        out, (hn, cn) = self.lstm(x, (h0, c0))

        att_out, alphat = self.attention(out)
        out = self.dropout(att_out)
        out = self.fc(out)
        out = torch.relu(out)
        #out = self.batchnorm(out)
        out = self.fc2(out)

        if base_val is None:
            return out 
        else:
            return out + base_val.reshape(-1,1)
    
    def attention(self, x):
        et = self.attention_layer(x)
        et = torch.tanh(et).squeeze(2)
        alphat = torch.softmax(et,dim=1)
        att_output = torch.einsum("ij, ijk -> ik",alphat,x)
        return att_output, alphat






