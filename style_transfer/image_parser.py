# Imports
from typing import List, Tuple
import os
import numpy as np
from PIL import Image
import requests
from io import BytesIO

from keras import backend
from keras.models import Model
from keras.applications.vgg16 import VGG16

from scipy.optimize import fmin_l_bfgs_b

dir_path = os.path.dirname(__file__)


class ImageParser:

    def __init__(self, image_size: int = 224, channels: int = 3,
                 imagenet_mean_rgb_values: Tuple[float] = (123.68, 116.779, 103.939)):
        self.image_size = image_size
        self.image_width = image_size
        self.image_height = image_size
        self.channels = channels
        self.imagenet_mean_rgb_values = imagenet_mean_rgb_values

    def __download(self, save_path: str, url_path: str):
        input_image = Image.open(BytesIO(requests.get(url_path).content))
        input_image = input_image.resize((self.image_width, self.image_height))
        save_path_dir = save_path.rpartition("/")[0]
        if not os.path.exists(save_path_dir):
            os.mkdir(save_path_dir)
        input_image.save(save_path)

    def download_images_from_url(self, content_img_url: str = None, style_img_url: str = None):
        if content_img_url is None and style_img_url is None:
            raise ValueError(f"Please specify content_img_url (or) style_img_path to download image")
        if content_img_url:
            self.__download(save_path=f"{dir_path}/content/content_img.png", url_path=content_img_url)
        if style_img_url:
            self.__download(save_path=f"{dir_path}/style/style_img.png", url_path=style_img_url)


if __name__ == "__main__":
    parser = ImageParser()
    parser.download_images_from_url(
        content_img_url="https://www.economist.com/sites/default/files/images/print-edition/20180602_USP001_0.jpg",
        style_img_url="http://meetingbenches.com/wp-content/flagallery/tytus-brzozowski-polish-architect-and-watercolorist-a-fairy-tale-in-warsaw/tytus_brzozowski_13.jpg")
