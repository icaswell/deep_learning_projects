#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# File: adversarial_neural_network_symbolic.py
# @Author: Isaac Caswell
# @created: 10 Jan 2016
#
#===============================================================================
# DESCRIPTION:
#
# A wrapper class that will modify a NeuralNetwork class to train with 
# adversarial examples.
# 
# Note that this will work for any class yo ufeed in, as long as it contains the
# following fields and methods:
#
# cost - a theano symbolic expression for the cost of the model
# embedded_input -  a symbolic variable representing a batch of input vectors.
# train_model(*args, **kwargs) - a function that will train the model 
# build_architecture(embedded_input) - a method that will take an embedded input
#                    and build the model architecture, returning the cost.
# compute_gradients() - a method that will compute whatever gradients etc. the 
#                   NeuralNetwork needs, and store than.
# classify_batch() - what it sounds like
#
#===============================================================================
# USAGE:
# from adversarial_neural_network_symbolic import AdversarialNeuralNetworkSymbolic
# 
# get_data_reader = lambda fname: stsv_reader(fname)
# model = LoopyMlp(get_data_reader = get_data_reader, input_dim=20, lrate=001, hdims = [20, 20, 2])
#
# model = AdversarialNeuralNetworkSymbolic(model, alpha=1.0, epsilon=0.8)
# model.train_model(train_data, epochs=N_EPOCHS, batch_size=16)
#===============================================================================


from dataloader import *
import util
from neural_network import NeuralNetwork
import theano.tensor as T
import theano

class AdversarialNeuralNetworkSymbolic(NeuralNetwork):
    def __init__(self, model, alpha=0.5, epsilon=0.5):
        """
        :param NeuralNetwork model: an instantiated instance of a NeuralNetwork
        :param float alpha: the constant by which to weight the adversarial objective
        :param float epsilon: the size of the perturbation used to make adversarial examples
        """


        #===============================================================================
        # make sure that the NeuralNetwork passed in has all the methods and fields that
        # it will need
        required_class_methods = set(["train_model", "cost", "embedded_input", "build_architecture", "compute_gradients", "classify_batch"])            
        assert len(required_class_methods - set(dir(model))) == 0   


        util.colorprint("building adversarial cost...", 'red')
        
        #===============================================================================
        # modifies the model's cost function to include an adversarial term                 
        self.make_cost_adversarial(model, alpha, epsilon)

        #===============================================================================
        # Now that we have modified the cost, recompute the gradients
        model.compute_gradients()

        self.model = model


    def make_cost_adversarial(self, model, alpha, epsilon):
        """
        Modifies the model's cost to include an adversarial term.
        """
        leaf_grads = T.grad(model.cost, wrt=model.embedded_input)

        anti_example = T.sgn(leaf_grads)

        # make the batch of adversarial examples
        adv_batch = model.embedded_input + epsilon*anti_example

        # stop the gradient here
        adv_batch = theano.gradient.disconnected_grad(adv_batch)

        adv_cost = model.build_architecture(adv_batch)

        model.cost = alpha*model.cost + (1-alpha)*adv_cost

    def train_model(self, *args, **kwargs):
        self.model.train_model(*args, **kwargs)


    def classify_batch(self, **kwargs):
        return self.model.classify_batch(**kwargs)
    



if __name__ == "__main__":
    
    from loopy_mlp import LoopyMlp
    from sklearn.metrics import classification_report


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

    get_data_reader = lambda fname: stsv_reader(fname)
    model = LoopyMlp(get_data_reader = get_data_reader, input_dim=INPUT_DIM, lrate=.001, n_unrolls = 1, loops=[], hdims = HDIMS)

    model = AdversarialNeuralNetworkSymbolic(model, alpha=0.5, epsilon=0.1)

    print "="*80
    print "model scores, train then test:"

    # get_train_reader = lambda: mnist_generator("train")
    model.train_model(data_fname=train_data, epochs=N_EPOCHS, batch_size=1)


    y_true, y_pred =  model.classify_batch(data_fname=train_data)
    print classification_report(y_true, y_pred) 

    y_true, y_pred =  model.classify_batch(data_fname=test_data)
    print classification_report(y_true, y_pred) 

    # rnn.train_model(
    #     saveto="saved_models/deleteme.npz",
    #     dataset = "data/imdb.pkl",
    #     max_epochs = 2,
    #     l2_reg_U = 0.0,
    #     optimizer = "adadelta",
    #     batch_size = 3,
    #     wemb_init = "random"
    # )


    # from rnn_mark_2 import Rnn

    # rnn = Rnn(adversarial=False, 
    #     hidden_dim = 6,
    #     word_dim = 4,
    #     maxlen = 50,
    #     weight_init_type = "ortho_1.0",
    #     debug=False,
    #     grad_clip_thresh=1.0,
    #     what_to_do_with_long_reviews = "filter",
    #     encoder = "lstm",
    #     )

    # rnn = AdversarialNeuralNetwork(rnn)


    # rnn.train_model(
    #     saveto="saved_models/deleteme.npz",
    #     dataset = "data/imdb.pkl",
    #     max_epochs = 2,
    #     l2_reg_U = 0.0,
    #     optimizer = "adadelta",
    #     batch_size = 3,
    #     wemb_init = "random"
    # )

