#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# File: adversarial_neural_network_make_examples.py
# @Author: Isaac Caswell
# @created: 7 Jan 2016
#
#===============================================================================
# DESCRIPTION:
#
# A wrapper class that will modify a NeuralNetwork class to train with 
# adversarial examples.
# 
# Note that this will work for any class yo ufeed in, as long as it contains the
# following methods:
#
# f_input_grad - a function which takes as input a training batch and their true
#                labels, and returns the gradient of the cost with respect to the
#                input
# get_data_reader - a function which when called yields a reader for a dataset. 
#                   The reader is a generator object that yields a tuple of (column 
#                   vector, label)
# classify_batch -  a method which takes a batch of examples and returns their labels
# update_params -  a method which takes a batch of examples and their labels, and
#                   performs a training update therewith
# train_model - what it sounds like yo
#
#===============================================================================
# USAGE:
# from adversarial_neural_network_make_examples import AdversarialNeuralNetworkMakeExamples
# 
# get_data_reader = lambda fname: stsv_reader(fname)
# model = LoopyMlp(get_data_reader = get_data_reader, input_dim=20, lrate=001, hdims = [20, 20, 2])
#
# model = AdversarialNeuralNetwork(model, alpha=1.0, epsilon=0.8)
# model.train_model(train_data, epochs=N_EPOCHS, batch_size=1)
#===============================================================================
# PROBLEMS:
# the create-adversarial-examples approach doesn't seem to work with a word vector
# approach if you want to train the embeddings.  The reason is that in order to 
# train the embeddings, you need to use actual vectors from the word embedding matrix.


from dataloader import *
import util
from neural_network import NeuralNetwork

class AdversarialNeuralNetworkMakeExamples(NeuralNetwork):
    def __init__(self, model, alpha=0.5, epsilon=0.5):
        """
        :param str type: "symbolic" or "make_examples".  The former modifies the cost function. The latter simulates actual examples.
        """

        if implementation_type == "make_examples":
            required_class_methods = set(["f_input_grad", "classify_batch", "get_data_reader", "update_params"])
            assert len(required_class_methods - set(dir(model))) == 0   
        elif implementation_type=="symbolic":
            required_class_methods = set(["f_input_grad", "train_model", "cost", "embedded_input", "build_architecture", "calculate_gradients", "classify_batch"])            
            assert len(required_class_methods - set(dir(model))) == 0            
        else:
            print "Error: unknown type %s"%implementation_type
            sys.exit(1)
        self.alpha = alpha
        self.epsilon = epsilon
        self.model = model

    def get_adversarial_examples(self, batch_x, batch_y):
        """
        model.f_input_grad is a function that takes a batch input and returns the derivative of the cost 
            function with respect to that input.
            \nabla_x J(\theta, x, y)
        """
        if "embed_batch" in dir(self.model):


        return batch_x + self.epsilon*np.sign(self.model.f_input_grad(batch_x, batch_y))


    # def train_model(self, **kwargs):

    # def train_model(self, data_fname, epochs=1, batch_size=16):
    #     """
    #     """
    #     for epoch in range(epochs):
    #         print "epoch %s..."%epoch
    #         # batch_reader = copy.copy(data_reader)
    #         batch_reader = self.get_data_reader(data_fname)

    #         #===========================================================================
    #         # let's do a batch update! 
    #         while True:
    #             # batch_x is of shape (input_dim, batch_size).
    #             # batch_y is a list of length batch_size
    #             batch_x, batch_y, success = self._get_minibatch(batch_reader, batch_size)
    #             if not success: break;
    #             # batch_x = self._prepend_intercept(batch_x)                
    #             cur_cost_val = self.update_params(batch_x, batch_y)

    def subsample_batch(self, batch_x, batch_y, alpha):
        """
        this could def be done better
        """
        sub_batch_x, sub_batch_y = [], []
        for i in range(len(batch_y)):
            if np.random.rand() < alpha:
                sub_batch_x.append(batch_x[:,i])
                sub_batch_y.append(batch_y[i])
        return np.array(sub_batch_x), batch_y



    def train_model(self, data_fname, 
                        epochs=1,
                        batch_size=16, 
                        max_epochs=500,
                        saveFreq=1110,
                        dispFreq=500, 
                        ):
        """
        :param function get_data_reader: a function that returns a generator which yields a tuple of vector, label.
            I'm sorry.
        :param int epochs: the number of full sweeps through the data to perform
        :param dict **kwargs: any other key-word arguments to be passed in to the 
        """
        update_index = 0

        for epoch in range(epochs):
            print "epoch %s..."%epoch
            # batch_reader = copy.copy(data_reader)

            #note that get_data_reader() can choose to shuffle the data
            batch_reader = self.model.get_data_reader(data_fname)

            #===========================================================================
            # let's do a batch update! 
            while True: 
                update_index += 1
                #===============================================================================
                # normal update
                batch_x, batch_y, success = self._get_minibatch(batch_reader, batch_size)
                # batch_x = self._prepend_intercept(batch_x)
                if not success: break;


                cur_cost_val = self.model.update_params(batch_x, batch_y, scale=self.alpha)

                if np.isnan(cur_cost_val) or np.isinf(cur_cost_val):
                    util.colorprint('something screwy happened on update %s, iteration %s: cur_cost_val = %s'%(update_index, epoch, cur_cost_val), "red")
                    return None

                #===============================================================================
                # adversarial update
                sub_batch_x, sub_batch_y = self.subsample_batch(batch_x, batch_y, self.alpha)

                adv_batch_x = self.get_adversarial_examples(batch_x, batch_y)   
                cur_adv_cost_val = self.model.update_params(adv_batch_x, batch_y, scale=1.0-self.alpha) 


                #===============================================================================
                #                 
                if np.mod(update_index, dispFreq) == 0:
                    print 'Epoch ', epoch, 'Update ', update_index, 'Cost ', cur_cost_val, 'adv. Cost ', cur_adv_cost_val




    def classify_batch(self, reader):
        return self.model.classify_batch(reader)
    

    #============================================================================================================
    # nice inheritable functions!     


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
        batch_x = np.hstack(batch_x)

        return batch_x, batch_y, success


if __name__ == "__main__":
    
    from loopy_mlp import LoopyMlp
    from sklearn.metrics import classification_report


    if 0:
        train_data = "data/binary_toy_data"
        test_data = "data/binary_toy_data"    
        N_EPOCHS = 20
        # INPUT_DIM = 24
        # INPUT_DIM, OUTPUT_DIM = 24, 2
        INPUT_DIM, OUTPUT_DIM = 7, 2 
        HDIMS = [10, OUTPUT_DIM]
    elif 1:
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
    model = LoopyMlp(get_data_reader = get_data_reader, input_dim=INPUT_DIM, lrate=001, n_unrolls = 1, loops=[], hdims = HDIMS)

    model = AdversarialNeuralNetwork(model, alpha=1.0, epsilon=0.8)

    print "="*80
    print "model scores, train then test:"

    # get_train_reader = lambda: mnist_generator("train")
    model.train_model(train_data, epochs=N_EPOCHS, batch_size=1)


    y_true, y_pred =  model.classify_batch(train_data)
    print classification_report(y_true, y_pred) 

    y_true, y_pred =  model.classify_batch(test_data)
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

