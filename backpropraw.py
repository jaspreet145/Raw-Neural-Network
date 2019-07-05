import numpy as np
class neural_network:
    def __init__(self,input_layer,hidden,output,lr,epoch):
        self.input_layer = input_layer
        self.out_layer = output
        self.hidden_layer = hidden
        
        self.weight_input = np.random.random((self.input_layer,self.hidden_layer))
        self.weight_hidden = np.random.random((self.hidden_layer,self.out_layer))
        self.epoch = epoch
        self.learn_rate = lr
        
    def predict(self,x):
        output_hidden = self.activation_function(np.dot(x,self.weight_input))
        output = self.activation_function(np.dot(output_hidden,self.weight_hidden))
        return output
        
    def activation_function(self,x,deriv = False):
        if(deriv==True):
            return x*(1-x)
        return 1/(1+np.exp(-x))
        
    def forward_propogate(self,x,y):
        output_hidden = self.activation_function(np.dot(x,self.weight_input))
        output = self.activation_function(np.dot(output_hidden,self.weight_hidden))
        #print(output)
        error = y - output
        return error,output_hidden,output
        
    def back_propogate(self,error,out_hidden,out,x):
        out_delta = error * self.activation_function(out,True)
        self.weight_hidden += (out_hidden.T.dot(out_delta)*self.learn_rate)

        hidden_layer_error = out_delta.dot(self.weight_hidden.T)
        hidden_delta = hidden_layer_error* self.activation_function(out_hidden,True)

        self.weight_input += (x.T.dot(hidden_delta)*self.learn_rate)
        
    def train(self,x,y):
        for _ in range(self.epoch):
            error,output_hidden,output = self.forward_propogate(x,y)
            self.back_propogate(error,output_hidden,output,x)        
