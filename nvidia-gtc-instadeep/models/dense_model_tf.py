from tensorflow.keras import initializers, Model
from tensorflow.keras.layers import Dense, Layer


class DenseNetwork(Layer):
    """Implements a dense network as a layer for use as a backbone."""
    def __init__(self, hidden_layers, num_outputs=None, activation_fn="relu", name=None, **kwargs):
        super(DenseNetwork, self).__init__(name=name)
        self.layers = [
            Dense(
                size,
                name=f"fc_{k}",
                activation=activation_fn,
                kernel_initializer=initializers.glorot_normal(),
                bias_initializer=initializers.constant(0.1),
            )
            for k, size in enumerate(hidden_layers)
        ]
        if num_outputs:
            self.layers.append(
                Dense(
                    num_outputs,
                    name="fc_out",
                    activation=None,
                    kernel_initializer=initializers.glorot_normal(),
                    bias_initializer=initializers.constant(0.1),
                )
            )

    def call(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


class DenseModel(Model):
    """Wraps DenseNetwork as a tf.Model for use by itself."""
    def __init__(self, num_outputs, name="DenseModel", params=None):
        super(DenseModel, self).__init__(name=name)
        self.params = params
        self.dense_layers = DenseNetwork(num_outputs=num_outputs, name="Dense", **self.params)

    def call(self, inputs):
        return self.dense_layers(inputs)
