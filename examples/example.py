import os

from style_transfer.train import Trainer
from style_transfer.utils import load_img

content_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "style_transfer/content/content_img.png")
style_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "style_transfer/style/style_img.png")

if __name__ == "__main__":
    content_image = load_img(content_path)
    style_image = load_img(style_path)
    trainer = Trainer(content_image=content_image, style_image=style_image)
    trainer.train(epochs=10, steps_per_epoch=10)