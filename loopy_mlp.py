# loopy_mlp.py

from collections import OrderedDict, defaultdict
import cPickle as pkl
import sys
import time
import argparse
import copy

from sklearn.metrics import classification_report
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from dataloader import *
from rnn_util_mark_2 import *
import util
from neural_network import NeuralNetwork

#===
"""
the gradient is always 0
the storing of the gradient works, indicating that the gradient itself is 0
changing the loss function didn't seem to help

"""

class LoopyMlp(NeuralNetwork):
# class LoopyMlp():    
    def __init__(self,
                get_data_reader = None,
                hdims = [5,4, 2],
                input_dim = 7,
                loops = [(-2, 0)],
                n_unrolls = 2,
                lrate = 0.001,
                L1_reg = 0.0001):
        """
        :param function get_data_reader: a function that returns a generator which yields a tuple of vector, label.
            This function takes as an argument the name of the dataset.
            I'm sorry.
        :param list loops: for each of these, insert a single connection from
             the first layer specified to the second layer specified.  e.g. [(-2, 0)] 
             creates a single loop from the last hidden layer to the first layer. 
             (-1 would make a loop from the output layer :P)
        """

        if get_data_reader is None: 
            get_data_reader = lambda fname: stsv_reader(fname)

        self.debug=True
        self.debug_model_color = "teal" #color in which statements about the model are printed

        #convert negative indices to positive indices.  This is useful later somewhere.
        for loop_i in range(len(loops)):
            beg, end = loops[loop_i]
            if beg < 0: beg = len(hdims) + beg
            loops[loop_i] = (beg, end)

        self.get_data_reader = get_data_reader


        self.hdims = hdims
        self.loops = loops
        self.n_unrolls = n_unrolls
        self.lrate = lrate
        self.L1_reg = L1_reg
        
        #Note that the cost is a mean so we can use minibatches to train
        self.loss_function = lambda h, y: -T.log(h[:,y]).mean()
        # self.loss_function = lambda h, y: h.mean() + y*0      

        # self.optimizer = self.sgd
        self.optimizer = self.adadelta      

        # initialize weight matrices, etc
        # +1 for intercept
        self._init_params(input_dim, hdims, loops)

        # convert them to theano shared variables
        self._init_tparams()

        self._init_regularizations()    

        # create computational graph.  include 1- and x nodes for the U matrices
        self.input = T.matrix('input')

        #below: embedded_input is identical to input for this model, but for other types of 
        # models (for instance sequence models) this might not be so.
        self.embedded_input = self.input

        self.y = T.vector('y', dtype='int64') #TODO: make y a float?
        self.cost = self.build_architecture(self.input)    

        #=====================================================================
        # Up until now we have built the symbolic variables necessary.  Now we 
        # will compile the functions!
        self.compile_model()

        self.compute_gradients()

        # creates a function that's used by the class AdversarialNetwork
        self.create_input_gradient_function()

    def create_input_gradient_function(self):
        """
        creates the function f_input_grad.  This is a function that takes a batch input and returns the derivative of the cost 
        function with respect to that input.

        i.e. 

        \nabla_x J(\theta, x, y)
        """
        grad_wrt_input = T.grad(self.cost, wrt=self.input)
        self.f_input_grad = theano.function([self.input, self.y], grad_wrt_input)



    def _init_regularizations(self):
        self.L1 = sum(
                abs(self.tparams[w_name]).sum() #+ 
                # abs(self.tparams[self._b_name_from_w_name(w_name)]).sum() 
                for w_name in self._get_w_matrix_names())
        self.L1 += sum(
                abs(self.tparams[u_name]).sum() #+ 
                # abs(self.tparams[self._b_name_from_w_name(u_name)]).sum() 
                for u_name in self._get_loop_matrix_names())        


    def parse_data_line(self, line):
        strvec, label = line.split("\t")
        label = int(label)
        assert label >= 0
        example_vec = np.array([[float(v_i) for v_i in strvec.split()]])#.astype('int32')
        return example_vec.T, label

    def _get_minibatch(self, batch_reader, batch_size):
        """
        returns a minibatch of examples and labels.
        if the reader runs out in the middle, then we won't complain.
        if we get handed an empty reader, we will complain!  We complain by returning False as the second value.
        """
        batch = []
        success = True
        for i in range(batch_size):
            pair = next(batch_reader, None)
            if pair is None:
                if not i:
                    return [], [], False
                else:
                    continue
            batch.append(pair)

        # outut dim x n_samples
        batch_x, batch_y = zip(*batch)
        batch_y = list(batch_y)
        batch_x = np.hstack(batch_x).astype(theano.config.floatX)

        return batch_x, batch_y, success

    def update_params(self, x, y, mask=None, scale=1.0):
        cur_cost = self.f_populate_gradients(x, y)
        self.f_update_params(self.lrate, scale)
        return cur_cost

    def train_model(self, data_fname, epochs=1, batch_size=16):
        """
        """
        for epoch in range(epochs):
            print "epoch %s..."%epoch
            # batch_reader = copy.copy(data_reader)
            batch_reader = self.get_data_reader(data_fname)

            #===========================================================================
            # let's do a batch update! 
            while True:
                # batch_x is of shape (input_dim, batch_size).
                # batch_y is a list of length batch_size
                batch_x, batch_y, success = self._get_minibatch(batch_reader, batch_size)
                if not success: break;
                # batch_x = self._prepend_intercept(batch_x)                
                cur_cost_val = self.update_params(batch_x, batch_y)


    def classify_batch(self, data_fname):
        data_reader = self.get_data_reader(data_fname)
        y_pred = []
        y_true = []
        for example_vec, label in data_reader:
            # example_vec = self._prepend_intercept(example_vec)
            y_true.append(label)
            pred = self.f_pred(example_vec)[0]
            y_pred.append(pred)

        #===============================================================================
        # below is some kick-arse debugging tools
        if len(set(y_pred)) == 1:
            util.colorprint("Warning: all predictions are of value %s"%y_pred[0], "flashing")
            util.colorprint("Here are the hidden layer activations for the lasst training example:", "flashing")
            self.print_sample_activations(example_vec)

        return y_true, y_pred


    def build_architecture(self, input):
        w_names = self._get_w_matrix_names()
        u_names = self._get_loop_matrix_names()
        all_hidden_activations = []
        if self.debug: util.colorprint("building architecture...", self.debug_model_color)
        # loop_inputs = a mapping from index to vector (dict), initialized to nothing
        # loop_outputs = a mapping from index to vector (dict), initialized to ones

        loop_inputs = {}
        loop_outputs = defaultdict(list)

        #======================================================
        # note that "output_layer" means output FROM the loop, and 
        # input_layer means input TO the loop.  Perhaps better names are in order.
        for input_layer, output_layer in self.loops:
            loop_inputs[input_layer] = None
            loop_output_shape = (self.all_layer_dims[output_layer], 1)
            loop_outputs[output_layer].append(np.ones(loop_output_shape))
            if self.debug:
                util.colorprint("creating loop output of shape %s from layer %s to layer %s"%(loop_output_shape, input_layer, output_layer), self.debug_model_color)

        for _ in range(self.n_unrolls):
            hidden_activation = input
            # hidden_activation = self._prepend_intercept(input)            
            #====================================================
            # put in all the weight matrices and populate the inputs to the loops
            for layer_i, w_name in enumerate(w_names):
                #optional addendum: terminate when the last loop is reached, unless it's the last unroll
                # b_name = self._b_name_from_w_name(w_name)
                if self.debug:
                    util.colorprint("Inserting parameters %s and (layer %s) into the graph..."%(w_name, layer_i), self.debug_model_color)

                if layer_i in loop_outputs:
                    for loop_output in loop_outputs[layer_i]:
                        if self.debug:
                            util.colorprint("\tInserting incoming loop activation...", self.debug_model_color)
                        hidden_activation *= loop_output
                    loop_outputs[layer_i] = []

                hidden_activation = T.dot(self.tparams[w_name], hidden_activation) 

                # print (1, hidden_activation.shape[1])
                # hidden_activation += T.tile(self.tparams[b_name], (1, hidden_activation.shape[1]))
                # hidden_activation += self.tparams[b_name]
                # hidden_activation = T.tanh(hidden_activation)
                hidden_activation = T.nnet.sigmoid(hidden_activation)
                # hidden_activation = self._prepend_intercept(hidden_activation)

                #---------------------------------------------------
                if layer_i in loop_inputs:
                    if self.debug:
                        util.colorprint("\tStoring outgoing loop activation...", self.debug_model_color)
                    loop_inputs[layer_i] = hidden_activation

                #---------------------------------------------------                    
                all_hidden_activations.append(hidden_activation)
            #====================================================
            # calculate the outputs 
            for u_i, u_name in enumerate(u_names):
                input_layer, output_layer = self.loops[u_i]
                # b_name = self._b_name_from_w_name(u_name)
                if self.debug:
                    util.colorprint("inserting %s and %s into the graph, ready to feed into layer %s"%(u_name, b_name, output_layer), self.debug_model_color)
                loop_output = T.dot(self.tparams[u_name], loop_inputs[input_layer])# +  self.tparams[b_name]
                loop_output = T.nnet.sigmoid(loop_output)
                loop_outputs[output_layer].append(loop_output)

        # final_activation = all_hidden_activations[-1]
        self.final_activation = T.nnet.softmax(all_hidden_activations[-1].T)
        all_hidden_activations.append(self.final_activation)

        self.all_hidden_activations = all_hidden_activations

        # self.final_activation = 1.0/(1.0 + T.nnet.sigmoid(final_activation))

        # off = 1e-8
        # if final_activation.dtype == 'float16':
        #     off = 1e-6

        # self.cost = -T.log(final_activation[self.y, 0] + off)        
       
        cost = self.loss_function(self.final_activation, self.y) + self.L1_reg * self.L1

        return cost


    def compile_model(self):
        self.f_pred_prob = theano.function([self.input], self.final_activation, name='f_pred_prob')
        self.f_pred_prob_debug = theano.function([self.input], self.all_hidden_activations, name='f_pred_prob')
        #TODO: I think the axis is right, but I am uncertain!! self.final_activation has shape (n, 1)
        self.f_pred =      theano.function([self.input], self.final_activation.argmax(axis=1), name='f_pred')

    def compute_gradients(self):
        # maybe doesn't need to be a class variable
        self.grads = T.grad(self.cost, wrt=self.tparams.values())

        #lrate: learning rate
        self.f_populate_gradients, self.f_update_params = self.optimizer()


        # =====================================================================
        # print out the computational graph and make an image of it too
        if self.debug and False:
            # util.colorprint("Following is the graph of the final hidden layer:", "blue")
            # final_activation_fn = theano.function([self.input], final_activation)
            # theano.printing.debugprint(final_activation_fn.maker.fgraph.outputs[0])   
            # util.colorprint("Also, saving png of computational graph:", "blue")
            # theano.printing.pydotprint(final_activation_fn, 
            #     outfile="output/lmlp_final_act_viz.png", 
            #     compact=True,
            #     scan_graphs=True,
            #     var_with_name_simple=True)
            util.colorprint("Following is the graph of the first of the derivatives:", "blue")
            final_grad_fn = theano.function([self.input, self.y], self.grads[0])
            theano.printing.debugprint(final_grad_fn.maker.fgraph.outputs[0]) 
            util.colorprint("Yay colorprinted:", "blue")
            print theano.pp(self.final_activation)
            util.colorprint("Also, saving png of computational graph:", "blue")
            theano.printing.pydotprint(final_grad_fn, 
                outfile="output/lmlp_final_grad_viz.png", 
                compact=True,
                scan_graphs=True,
                var_with_name_simple=True)            


    def _get_w_matrix_names(self):
        """
        note that this includes all non-looping weight matrices, including that connecting to the output
        """
        return ["W_%s"%i for i in range(len(self.hdims))]

    def _get_loop_matrix_names(self):
        return ["U_%s"%i for i in range(len(self.loops))]    

    def _init_params(self, input_dim, hdims, loops):

        self.params = {}
        w_names = self._get_w_matrix_names()
        u_names = self._get_loop_matrix_names()
        #================================================================        
        # create parameter matrices:
        self.all_layer_dims = [input_dim] + hdims
        for layer_i, l_size in enumerate(self.all_layer_dims):
            if layer_i==0: continue
            #TODO: intercepts
            param_name = w_names[layer_i-1]

            # IMPORTANT: the +1 comes from the bias/intercept
            param_shape = (l_size, self.all_layer_dims[layer_i - 1])
            if self.debug:
                print "Creating a matrix named %s of shape %s"%(param_name, param_shape)
            self.params[param_name] = self._weight_init(param_shape, include_intercept=False)

            #------------------------------------------------------------
            # b_name = self._b_name_from_w_name(param_name)
            # self.params[b_name] =  np.zeros((param_shape[0],1)).astype(theano.config.floatX)
            # self.params[b_name] = np.zeros((param_shape[0],1), dtype=theano.config.floatX)

        #================================================================
        # create matrices for the loops
        for loop_i, (loop_beginning, loop_end) in enumerate(loops):            
            param_name = u_names[loop_i]
            param_shape = (self.all_layer_dims[loop_end], hdims[loop_beginning])
            if self.debug:
                print "Creating a matrix named %s of shape %s"%(param_name, param_shape)
            self.params[param_name] = self._weight_init(param_shape, include_intercept=False)
            #------------------------------------------------------------
            # b_name = self._b_name_from_w_name(param_name)
            # self.params[b_name] = np.zeros((param_shape[0],1), dtype=theano.config.floatX)

    def _init_tparams(self):
        self.tparams = {}
        for param_name in self.params.keys():
            self.tparams[param_name] = theano.shared(self.params[param_name],
                         # broadcastable=(False, True),
                         name=param_name)
    
    def _weight_init(self, param_shape, include_intercept=False):
        """
        :param bool include_intercept: if true, the first column will be initialized as zeros.
            I don't know why the intercept is initialized differently.  Maybe it oughtn't to be.
            Actually, I'm not going to do anythign differently.  But I might.  What I might do is
            have this function called recursively on the remaining part of the matrix with 
            include_intercept=False.
            Whatever.

            Actually, it's important to zero the first column, come to think of it, in case the intercept is really big!
            I'll just do that.
        """
        #TODO: make orthonormal
        # return np.random.random(param_shape)
        # return np.random.random(param_shape) - 0.5
        rng = np.random.RandomState(1234)
        fan_sum = param_shape[0] + param_shape[1]
        W_values = np.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / fan_sum),
                    high=numpy.sqrt(6. / fan_sum),
                    size=param_shape
                ),
                dtype=theano.config.floatX
            )
        # if activation == theano.T.nnet.sigmoid:
        #     W_values *= 4
        if include_intercept:
            W_values[:, 0]*=0
        return W_values

    def print_sample_activations(self, example_vec=None):
        if example_vec==None:
            example_vec = 0.01*np.random.random((self.input_dim, 1))
        w_names = self._get_w_matrix_names()

        all_activations =  self.f_pred_prob_debug(example_vec)
        util.print_matrix(example_vec, color="red")
        for layer_i, (w_name, activation) in enumerate(zip(w_names, all_activations)):
            print '\n|\nV\n'
            # util.colorprint("for layer %s:"%layer_i, "bold")
            # neuron_mask_name = self._get_layer_mask_name(layer_i)
            # weight_mask_name = self._get_mask_name(w_name, "weight")            
            util.print_matrix(self.tparams[w_name].get_value(), color="blue", newline=False) 
            # if weight_mask_name in self.masks:
            #     print 'X'
            #     util.print_matrix(self.masks[weight_mask_name], color="graphite")  

            print '\n|\nV\n'

            util.print_matrix(activation.T, color="magenta", newline=False)
            # if neuron_mask_name in self.masks:
            #     print 'X'
            #     util.print_matrix(self.masks[neuron_mask_name].T, color="graphite") 
    # def _b_name_from_w_name(self, w_name):
    #     return w_name + "_b"

if __name__=="__main__":
    # train_data = "binary_toy_data"
    # train_data = "toy_data_deleteme"

    # train_data = "data/weird_data_train"
    # test_data = "data/weird_data_test"

    if 1:
        train_data = "data/binary_toy_data"
        test_data = "data/binary_toy_data"    
        N_EPOCHS = 20
        # INPUT_DIM = 24
        # INPUT_DIM, OUTPUT_DIM = 24, 2
        INPUT_DIM, OUTPUT_DIM = 7, 2 
        HDIMS = [10, OUTPUT_DIM]
    elif 0:
        train_data = "data/d100_dot_train" 
        test_data = "data/d100_dot_test"     

        N_EPOCHS = 20
        # INPUT_DIM = 24
        # INPUT_DIM, OUTPUT_DIM = 24, 2
        INPUT_DIM, OUTPUT_DIM = 100, 2 
        HDIMS = [100, 100, 50, OUTPUT_DIM]        

    elif 1:
        train_data = "data/d20_dot_train" 
        test_data = "data/d20_dot_test"  

        N_EPOCHS = 20
        # INPUT_DIM = 24
        # INPUT_DIM, OUTPUT_DIM = 24, 2
        INPUT_DIM, OUTPUT_DIM = 20, 2 
        HDIMS = [20, 20, OUTPUT_DIM]


    # INPUT_DIM, OUTPUT_DIM = 784, 10    
    # model = LoopyMlp(input_dim=7, lrate=0.01, n_unrolls = 2, loops=[(2, 0), (1, 0)], hdims = [8, 5,2])
    get_data_reader = lambda fname: stsv_reader(fname)    
    models = [
            # LoopyMlp(input_dim=INPUT_DIM, lrate=0.01, n_unrolls = 1, loops=[], hdims = [18, 9, OUTPUT_DIM]),
            # LoopyMlp(input_dim=INPUT_DIM, lrate=0.01, n_unrolls = 2, loops=[], hdims = [18, 9, OUTPUT_DIM]),
            # LoopyMlp(input_dim=INPUT_DIM, lrate=0.01, n_unrolls = 2, loops=[(2, 0)], hdims = [200, 100, 40, OUTPUT_DIM]),
            LoopyMlp(get_data_reader = get_data_reader, input_dim=INPUT_DIM, lrate=001, n_unrolls = 1, loops=[], hdims = HDIMS),            
            # LoopyMlp(input_dim=INPUT_DIM, lrate=0.01, n_unrolls = 3, loops=[], hdims = [18, 9,OUTPUT_DIM]),
            # LoopyMlp(input_dim=INPUT_DIM, lrate=0.01, n_unrolls = 3, loops=[(2, 0)], hdims = [18, 9, OUTPUT_DIM]),            
            ]      

    for model_no, model in enumerate(models):
        print "="*80
        print "model %s, train then test:"%model_no

        # get_train_reader = lambda: mnist_generator("train")
        model.train_model(train_data, epochs=N_EPOCHS, batch_size=1)


        y_true, y_pred =  model.classify_batch(train_data)
        print classification_report(y_true, y_pred) 

        y_true, y_pred =  model.classify_batch(test_data)
        print classification_report(y_true, y_pred)  

                     
