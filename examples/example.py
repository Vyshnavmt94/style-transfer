import os

from style_transfer.train import Trainer
from style_transfer.utils import load_img

# content_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "style_transfer/content/content_1.png")
# style_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "style_transfer/style/style_1.png")

content_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "style_transfer/content/deadpool.jpg")
style_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "style_transfer/style/guernica.jpg")

if __name__ == "__main__":
    content_image = load_img(content_path)
    style_image = load_img(style_path)
    trainer = Trainer(content_image=content_image, style_image=style_image)
    trainer.train(epochs=50, steps_per_epoch=100)
