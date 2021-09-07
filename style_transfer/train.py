import os
from typing import Any

from style_transfer.style_transfer import StyleTransfer
import tensorflow as tf
import time
import IPython.display as display
from tqdm import tqdm
from style_transfer.utils import tensor_to_image, clip_0_1

# style_weight=1e-2
# content_weight=1e4

# style_weight=0.02
# content_weight=4.5

dir_path = os.path.dirname(__file__)


class Trainer:

    def __init__(self, style_image: Any, content_image: Any, content_weight: float = 0.01,
                 style_weight: float = 1, total_variation_weight: float = 0.995,
                 total_variation_loss_factor: float = 1.25):
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.total_variation_weight = total_variation_weight
        self.total_variation_loss_factor = total_variation_loss_factor

        self.style_transfer = StyleTransfer()
        self.style_targets = self.style_transfer(style_image)['style']
        self.content_targets = self.style_transfer(content_image)['content']

        self.image = tf.Variable(content_image)

    def style_content_loss(self, outputs):
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - self.style_targets[name]) ** 2)
                               for name in style_outputs.keys()])
        style_loss *= self.style_weight / self.style_transfer.num_style_layers

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - self.content_targets[name]) ** 2)
                                 for name in content_outputs.keys()])
        content_loss *= self.content_weight / self.style_transfer.num_content_layers
        loss = style_loss + content_loss
        return loss

    def train(self, epochs: int = 10, steps_per_epoch: int = 100,
              optimizer=tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)):
        for epoch in tqdm(range(epochs)):
            for step in range(steps_per_epoch):
                self.train_step(optimizer=optimizer)
                # print(".", end='', flush=True)
            # display.clear_output(wait=True)
            # display.display(tensor_to_image(self.image))
            # print("Train step: {}".format(step))
            if (epoch + 1) % 10 == 0:
                save_path = dir_path + "/result"
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                tensor_to_image(self.image).save(save_path + f"/image_{epoch + 1}.png")

    @tf.function()
    def train_step(self, optimizer):
        with tf.GradientTape() as tape:
            outputs = self.style_transfer(self.image)
            loss = self.style_content_loss(outputs)
            loss += self.total_variation_weight * tf.image.total_variation(self.image)

        grad = tape.gradient(loss, self.image)
        optimizer.apply_gradients([(grad, self.image)])
        self.image.assign(clip_0_1(self.image))
