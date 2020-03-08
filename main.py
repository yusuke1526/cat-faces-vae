import numpy as np
from kivy.app import App
from kivy.core.window import Window
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle
from kivy.graphics.texture import Texture
from kivy.properties import StringProperty, ObjectProperty
from kivy.config import Config
from model.models import load_model_and_weights
from functions.data import fileRead
import cv2


class app(App):
    def build(self):
        return root

class ImageWidget(Widget):
    rate = 6
    def __init__(self, **kwargs):
        img = kwargs.get('img')
        position = kwargs.get('position')
        del kwargs['img']
        del kwargs['position']
        super(ImageWidget, self).__init__(**kwargs)
        texture = self.image_to_texture(img)
        print(self.pos)
        position = [position*400, self.pos[0]]
        print(position)
        with self.canvas:
            Rectangle(pos=position, size=texture.size, texture=texture)

    def image_to_texture(self, img):
        img = (img * 255).astype(np.uint8)
        img = cv2.resize(img, None, fx=self.rate, fy=self.rate, interpolation=cv2.INTER_NEAREST)
        texture = Texture.create(size=(img.shape[1], img.shape[0]))
        texture.blit_buffer(img.tostring())
        texture.flip_vertical()
        return texture

encoder = load_model_and_weights('model/encoder_0108')
decoder = load_model_and_weights('model/decoder_0108')

root = BoxLayout(orientation='vertical')
label = Label(text='test')
label.size_hint = (1.0, 0.1)
imageArea = BoxLayout(orientation='horizontal')
img = fileRead('src/1.jpg')
decoded = decoder.predict(encoder.predict(np.expand_dims(img, 0)))[0]
im1 = ImageWidget(img=img, position=0)
imageArea.add_widget(im1)
im2 = ImageWidget(img=decoded, position=1)
imageArea.add_widget(im2)
root.add_widget(label)
root.add_widget(imageArea)

if __name__ == "__main__":
    app().run()