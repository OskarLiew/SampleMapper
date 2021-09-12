import torch.nn as nn
import numpy as np


class LSTMEncoder(nn.Module):
    
    def __init__(self, input_size, first_layer_size, encoding_size, num_layers):
        super(LSTMEncoder, self).__init__()
        self.encoder_lstm_list = nn.ModuleList([])
        
        # Get evenly spaced layer sizes
        layer_sizes = np.linspace(first_layer_size, encoding_size, num_layers, dtype=int)
        layer_sizes = np.append(input_size, layer_sizes)
        
        for i in range(num_layers):
            encoder_lstm = nn.LSTM(
                input_size=layer_sizes[i],
                hidden_size=layer_sizes[i + 1],
                batch_first=True,                
            )
            self.encoder_lstm_list.append(encoder_lstm)

    def forward(self, x):
        for encoder in self.encoder_lstm_list:
            x, _ = encoder(x)
        return x[:,-1,:].unsqueeze(1)

    
class LSTMDecoder(nn.Module):
    
    def __init__(self, output_size, last_layer_size, encoding_size, num_layers):
        super(LSTMDecoder, self).__init__()
        self.decoder_lstm_list = nn.ModuleList([])
        
        # Get evenly spaced layer sizes
        layer_sizes = np.linspace(encoding_size, last_layer_size, num_layers, dtype=int)
        for i in range(num_layers - 1):
            decoder_lstm = nn.LSTM(
                input_size=layer_sizes[i],
                hidden_size=layer_sizes[i + 1],
                batch_first=True,                
            )
            self.decoder_lstm_list.append(decoder_lstm)
        self.output_layer = nn.Linear(last_layer_size, output_size)
        
    def forward(self, x):
        for decoder in self.decoder_lstm_list:
            x, _ = decoder(x)
        x = self.output_layer(x)
        return x
    
    
class LSTMAutoEncoder(nn.Module):
    
    def __init__(self, input_size, first_layer_size, encoding_size, encoder_layers=1):
        super(LSTMAutoEncoder, self).__init__()
        self.encoding_size = encoding_size
        
        self.lstm_encoder = LSTMEncoder(input_size, first_layer_size, encoding_size, encoder_layers)
        self.lstm_decoder = LSTMDecoder(input_size, first_layer_size, encoding_size, encoder_layers)

    def encode(self, x):
        return self.lstm_encoder(x)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = self.lstm_encoder(x)
        x = x.expand(batch_size, seq_len, self.encoding_size)
        x = self.lstm_decoder(x)
        return x
