# Imports
import numpy as np
from PIL import Image
import tensorflow as tf
from scipy.optimize import fmin_l_bfgs_b


class StyleTransfer:

    def __init__(self):
        self.pretrained_model = None
        self.content_layers = ['block5_conv2']
        self.style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        self.num_style_layers = len(self.style_layers)
        self.num_content_layers = len(self.content_layers)
        self.style_extraction_model = None
        self.content_extraction_model = None
        self.preprocess_input = None

        self.load_vgg19()
        self.style_transfer_model = self.get_feature_extraction_model(self.style_layers + self.content_layers)

    def load_vgg19(self):
        """
        Loads pretrained model. The intermediate layers of this model is used to get the content and style representations of the image.
        :return:
        """
        self.pretrained_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
        self.preprocess_input = tf.keras.applications.vgg19.preprocess_input
        print(f"Layers of network: {[layer.name for layer in self.pretrained_model.layers]}")

    def get_feature_extraction_model(self, layers):
        """
        Creates a vgg model that returns a list of intermediate output values.
        """
        self.pretrained_model.trainable = False
        outputs = [self.pretrained_model.get_layer(layer).output for layer in layers]
        return tf.keras.Model([self.pretrained_model.input], outputs)

    def __call__(self, inputs):
        # Expects float input in [0,1]
        inputs = inputs * 255.0
        preprocessed_input = self.preprocess_input(inputs)
        outputs = self.style_transfer_model(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [self.gram_matrix(style_output) for style_output in style_outputs]
        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}

    def get_layer_statistics(self, layers, outputs):
        """
        Look at the statistics of each layer's output
        :param layers:
        :param outputs:
        :return:
        """
        for name, output in zip(layers, outputs):
            print(name)
            print("  shape: ", output.numpy().shape)
            print("  min: ", output.numpy().min())
            print("  max: ", output.numpy().max())
            print("  mean: ", output.numpy().mean())
            print("\n")

    def gram_matrix(self, input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / (num_locations)


if __name__ == "__main__":
    style_transfer = StyleTransfer()
    style_transfer.load_vgg19()
