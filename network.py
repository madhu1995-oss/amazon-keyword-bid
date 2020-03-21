import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, device, num_inputs: int, num_output: int, num_hidden_layers: int, 
                hidden_layer_info: int(), activation_function: str):
        super().__init__()

        torch.set_grad_enabled(True)

        self.fc_list = []
        input = num_inputs
        assert(num_hidden_layers == len(hidden_layer_info))
        key_item = 1
        for item in hidden_layer_info:
            output = item
            self.fc_list.append(nn.Linear(in_features=input, out_features=output).to(device))
            key = "fc" + str(key_item)
            setattr(self, key, self.fc_list[-1])
            input = output
            key_item += 1

        output = num_output
        self.output_fc_layer = nn.Linear(in_features=input, out_features=output).to(device)

        self.num_inputs = num_inputs
        self.num_output = num_output
        self.num_hidden_layers = num_hidden_layers
        self.activation_function: str = activation_function

    def _activation(self, t: torch.Tensor, activation_function: str) -> torch.Tensor:
        if activation_function.lower() == 'relu':
            return F.relu(t)
        elif activation_function.lower() == 'sigmoid':
            return torch.sigmoid(t)
        elif activation_function.lower() == 'tanh':
            return F.tanh(t)
        else:
            return t

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        input = self.num_inputs
        t = t.reshape(-1, input) 

        for item in self.fc_list:
            t = item(t)
            t = self._activation(t, self.activation_function)

        t = self.output_fc_layer(t)

        return t        