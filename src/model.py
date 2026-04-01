import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, Dense, Flatten, Dropout,
                                     GlobalAveragePooling2D, BatchNormalization,
                                     Multiply, MaxPooling2D, Reshape)
from tensorflow.keras.models import Model

def se_block(input_tensor, ratio=16):
    """Squeeze-and-Excitation block for channel-wise attention."""
    channels = input_tensor.shape[-1]
    
    squeeze = GlobalAveragePooling2D()(input_tensor)
    excitation = Dense(channels // ratio, activation="relu")(squeeze)
    excitation = Dense(channels, activation="sigmoid")(excitation)
    excitation = Reshape((1, 1, channels))(excitation)
    
    return Multiply()([input_tensor, excitation])

def multi_scale_conv(input_tensor, filters):
    """Multi-scale convolution block using 3x3, 5x5, and 7x7 filters."""
    conv3 = Conv2D(filters, (3, 3), padding="same", activation="relu")(input_tensor)
    conv5 = Conv2D(filters, (5, 5), padding="same", activation="relu")(input_tensor)
    conv7 = Conv2D(filters, (7, 7), padding="same", activation="relu")(input_tensor)
    
    output = tf.keras.layers.Add()([conv3, conv5, conv7])
    output = BatchNormalization()(output)
    
    return output

def build_msa_net(input_shape=(128, 128, 3)):
    """Builds the complete Multi-Scale Attention Network (MSA-Net)."""
    inputs = Input(shape=input_shape)
    
    x = multi_scale_conv(inputs, 64)
    x = se_block(x)
    
    x = multi_scale_conv(x, 128)
    x = se_block(x)
    
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(1, activation="sigmoid")(x)
    
    model = Model(inputs, outputs, name="MSA-Net")
    return model

if __name__ == "__main__":
    # Test the model structure by running this script directly
    model = build_msa_net()
    model.summary()
