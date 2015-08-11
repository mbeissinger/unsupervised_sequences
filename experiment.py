import math
import theano
import theano.tensor as T
import numpy as np
from opendeep import config_root_logger
from opendeep.data import MNIST
from opendeep.models import Model, GSN, RNN
from opendeep.monitor import Monitor
from opendeep.optimization import RMSProp

def stack_mnist(matrix):
    return np.asarray(
        [np.vstack((matrix[i], matrix[i + 1])) for i in range(matrix.shape[0] - 1)]
    )

class RNN_GSN(Model):
    def __init__(self):
        super(RNN_GSN, self).__init__()

        # input is a 3-dimensional matrix in the form (n_sequence, sequence_len, dimensionality)
        xs = T.tensor3("Xs")
        xs = xs.dimshuffle(1, 0, 2)
        self.input = xs
        # self.input now is matrix of form (sequence_len, n_sequence, dimensionality)

        gsn_inputs = 784
        gsn_hiddens = 1000
        gsn_layers = 1
        gsn_walkbacks = 1

        # Create the GSN that will encode the input space
        gsn = GSN(
            input_size=gsn_inputs,
            hidden_size=gsn_hiddens,
            layers=gsn_layers,
            walkbacks=gsn_walkbacks,
            visible_activation='sigmoid',
            hidden_activation='tanh',
            image_height=28,
            image_width=28,
            input_noise_level=0.5
        )
        # grab the input arguments
        gsn_args = gsn.args.copy()
        # grab the parameters it initialized
        gsn_params = gsn.get_params()

        # encode each timestep batched input to its hidden representation
        def encode_step(x):
            gsn = GSN(
                inputs_hook=(gsn_inputs, x),
                params_hook=gsn_params,
                **gsn_args
            )
            return gsn.get_hiddens()

        hiddens, encode_updates = theano.scan(
            fn=encode_step,
            sequences=[self.input],
            outputs_info=[None]
        )

        # make the rnn to go from hiddens -> hiddens
        self.rnn = RNN(
            inputs_hook=(gsn_hiddens * (math.ceil(gsn_layers / 2.)), hiddens),
            hidden_size=100,
            # needs to output hidden units for odd layers of GSN
            output_size=gsn_hiddens * (math.ceil(gsn_layers / 2.)),
            layers=1,
            activation='tanh',
            hidden_activation='relu',
            weights_init='uniform', weights_interval='montreal',
            r_weights_init='identity'
        )

        # decode the rnn's output gsn hiddens to the next x value, and return the cost and output
        def decode_step(hiddens, x):
            gsn = GSN(
                inputs_hook=(gsn_inputs, x),
                hiddens_hook=(gsn_hiddens, hiddens),
                params_hook=gsn_params,
                **gsn_args
            )
            return gsn.get_train_cost(), gsn.get_outputs(), gsn.show_cost

        (costs, outputs, recon_costs), decode_updates = theano.scan(
            fn=decode_step,
            sequences=[self.rnn.get_outputs(), self.input[1:]],
            outputs_info=[None,None,None]
        )

        self.monitor = T.mean(recon_costs)

        self.outputs = outputs

        self.updates = dict()
        self.updates.update(self.rnn.get_updates())
        self.updates.update(encode_updates)
        self.updates.update(decode_updates)

        self.cost = costs.sum()
        self.params = gsn_params + self.rnn.get_params()

    # Model functions necessary for training
    def get_inputs(self):
        return [self.input]

    def get_params(self):
        return self.params

    def get_train_cost(self):
        return self.cost

    def get_updates(self):
        return self.updates

    def get_outputs(self):
        return self.outputs



def main():
    # get an mnist dataset sequenced 0-9 repeating
    mnist = MNIST(sequence_number=1, seq_3d=True, seq_length=13)
    # transform the mnist dataset into pairs of change images [0,1],[1,2],[2,3], etc.
    train_data = stack_mnist(mnist.train_inputs)
    valid_data = stack_mnist(mnist.valid_inputs)
    test_data = stack_mnist(mnist.test_inputs)

    rnngsn = RNN_GSN()

    recon = Monitor(name="binary_crossentropy", expression=rnngsn.monitor)

    optimizer = RMSProp(
        model=rnngsn,
        dataset=mnist,
        epochs=500,
        batch_size=100,
        save_freq=10,
        stop_patience=30,
        stop_threshold=.9995,
        learning_rate=1e-6,
        decay=.95,
        max_scaling=1e5,
        grad_clip=5.,
        hard_clip=False
    )
    optimizer.train(monitor_channels=recon)


if __name__ == '__main__':
    config_root_logger()
    main()
