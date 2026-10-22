"""Define the neural network used for Stage 2 temperature correction."""

from keras import layers, Model

class UCTModel(Model):
    """Build a dense regression model for the Urban Climate Twin."""

    def __init__(self, input_dim, hidden_units=[128, 32], dropout_rate=0.1):
        """Initialize the feature layers and regression head."""
        super().__init__(name="UCTModel")

        self.input_dim = input_dim
        
        # Extract features from the input.
        self.feature_layers = []
        for units in hidden_units:
            self.feature_layers.append(layers.Dense(units, activation='relu'))

        # Predict the target value.
        self.regressor_layers = [
            layers.Dropout(dropout_rate),
            layers.Dense(1)
        ]

    def build(self, input_shape):
        """Build each model layer for the given input shape."""
        x_shape = input_shape
        for layer in self.feature_layers:
            layer.build(x_shape)
            x_shape = layer.compute_output_shape(x_shape)
        for layer in self.regressor_layers:
            layer.build(x_shape)
            x_shape = layer.compute_output_shape(x_shape)
        super().build(input_shape)

    def call(self, x, training=False):
        """Pass inputs through the feature layers and regression head."""
        for layer in self.feature_layers:
            x = layer(x, training=training)

        for layer in self.regressor_layers:
            x = layer(x, training=training)
        return x