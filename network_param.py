class NetworkParam:
    def __init__(self, num_input, num_output, num_hidden_layers, auto_compute_hidden_layer_info=True, hidden_layer_info=None):
        self.num_input = num_input
        self.num_output = num_output
        self.num_hidden_layers = num_hidden_layers
        if auto_compute_hidden_layer_info:
            self.hidden_layer_info = []
            mulitpler = 1
            for i in range(self.num_hidden_layers):
                self.hidden_layer_info.append(self.num_input // mulitpler)
                mulitpler = mulitpler * 1
        else:
            self.hidden_layer_info = hidden_layer_info

        assert(self.num_hidden_layers == len(self.hidden_layer_info))

    def get_model_filename(self):
        return str(self.num_input) + "_" + str(self.num_hidden_layers) + "_" + str(self.num_output) + ".pt"