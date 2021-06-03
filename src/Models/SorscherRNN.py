import tensorflow as tf
from tensorflow.keras.layers import Dense, SimpleRNN


class SorscherRNN(tf.keras.Model):
    """
    Model based on:
    https://github.com/ganguli-lab/grid-pattern-formation/blob/master/model.py
    """
    def __init__(
        self,
        Ng=4096,
        Np=512,
        weight_decay=1e-4,
        activation="relu",
        **kwargs
    ):
        super(SorscherRNN, self).__init__(**kwargs)
        self.Ng = Ng
        self.Np = Np

        # define network architecture
        self.encoder = Dense(self.Ng, name="encoder", use_bias=False)
        self.RNN = SimpleRNN(
            self.Ng,
            return_sequences=True,
            activation=activation,
            recurrent_initializer="glorot_uniform",
            recurrent_regularizer=tf.keras.regularizers.L2(weight_decay),
            name="RNN",
            use_bias=False,
        )
        # Linear read-out weights
        self.decoder = Dense(self.Np, name="decoder", use_bias=False)

    def g(self, inputs):
        """
        Compute grid cell activations.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
        Returns:
            g: Batch of grid cell activations with shape [batch_size, sequence_length, Ng].
        """
        v, p0 = inputs
        init_state = self.encoder(p0)
        return self.RNN(v, initial_state=init_state)

    def call(self, inputs, softmax=False):
        """
        Predict place cell code.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
        Returns:
            place_preds: Predicted place cell activations with shape
                [batch_size, sequence_length, Np].
        """
        place_preds = self.decoder(self.g(inputs))

        return (
            place_preds
            if not softmax
            else tf.keras.activations.softmax(place_preds)
        )








