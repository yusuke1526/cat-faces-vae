import numpy as np
from kivy.app import App
from kivy.core.window import Window
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.uix.textinput import TextInput
from kivy.uix.checkbox import CheckBox
from kivy.uix.button import Button
from kivy.uix.colorpicker import ColorPicker
from kivy.uix.stencilview import StencilView
from kivy.graphics import Color, Rectangle
from kivy.graphics.texture import Texture
from kivy.properties import StringProperty, ObjectProperty
from kivy.config import Config
from model.models import load_model_and_weights
from functions.data import fileRead
import cv2

Config.set('graphics', 'resizable', False)
Config.set('input', 'mouse', 'mouse,disable_multitouch')
#Window.size = (820, 820)

class MainApp(App):
    def build(self):
        return root

class RootLayout(BoxLayout):
    def __init__(self, **kwargs):
        super(RootLayout, self).__init__(**kwargs)
        self.size = Window.size
        with self.canvas.before:
            Color(0.6, 0.6, 0.6, 1)
            self.rect = Rectangle(size=self.size, pos=self.pos)

class ImageWidget(Widget):
    rate = 6
    color = [0, 0, 0]
    def __init__(self, **kwargs):
        self.img = kwargs.get('img')
        if type(self.img) is str:
            self.img = fileRead(img)
        self.img_size = self.img.shape[0]
        self.img_mod = self.img.copy()
        del kwargs['img']
        super(ImageWidget, self).__init__(**kwargs)
        self.update_image()

    def image_to_texture(self, img):
        img = (img * 255).astype(np.uint8)
        img = cv2.resize(img, None, fx=self.rate, fy=self.rate, interpolation=cv2.INTER_NEAREST)
        texture = Texture.create(size=(img.shape[1], img.shape[0]))
        texture.blit_buffer(img.tostring())
        texture.flip_vertical()
        return texture

    def update_image(self):
        texture = self.image_to_texture(self.img_mod)
        with self.canvas:
            Rectangle(pos=(0, 0), size=texture.size, texture=texture)

    def mask_rectangle(self, x1, y1, x2, y2):
        # 座標を画像上の位置に変換
        x1 = int(x1 / self.rate)
        x2 = int(x2 / self.rate)
        y1 = 64 - int(y1 / self.rate)
        y2 = 64 - int(y2 / self.rate)

        if (x1<0) | (x2>=self.img_size) | (y1<0) | (y2>=self.img_size):
            return
        if chbox.active:
            for i in range(y1, y2):
                for j in range(x1, x2):
                    self.img_mod[i, j, :] = np.random.uniform(size=3)
        else:
            self.img_mod[y1:y2, x1:x2, :] = self.color
        self.update_image()

    def on_touch_down(self, touch):
        self.x_start = touch.x
        self.y_start = touch.y

    def on_touch_move(self, touch):
        self.mask_rectangle(self.x_start, self.y_start, touch.x, touch.y)

    def on_touch_up(self, touch):
        update_im2()


def update_im2():
    decoded = decoder.predict(encoder.predict(np.expand_dims(im1.img_mod, 0)))[0]
    cv2.imwrite('src/.tmp.png', (cv2.cvtColor(decoded, cv2.COLOR_RGB2BGR)*255).astype(np.uint8))
    im2.reload()

def set_im1(num):
    img = fileRead(f'src/{num}.jpg')
    im1.img_mod = img.copy()
    im1.img = img.copy()
    im1.update_image()
    update_im2()

#モデルの読み込み
encoder = load_model_and_weights('model/encoder_0108')
decoder = load_model_and_weights('model/decoder_0108')

# ボタンエリア
def callback_reset_button(self):
    im1.img_mod = im1.img.copy()
    im1.update_image()
    update_im2()
    
def on_text(self, value):
    if value.isdecimal() & 0<int(value)<=100:
        set_im1(value)

button = Button(text='reset')
button.bind(on_release=callback_reset_button)

textinput = TextInput(text='1', multiline=False)
textinput.bind(text=on_text)

label = Label(text='random pattern')
chbox = CheckBox()

buttonArea = BoxLayout(orientation='horizontal')
button.size_hint = (0.9, 1.0)
textinput.size_hint = (0.1, 1.0)
label.size_hint = (0.3, 1.0)
chbox.size_hint = (0.1, 1.0)
buttonArea.add_widget(button)
buttonArea.add_widget(textinput)
buttonArea.add_widget(label)
buttonArea.add_widget(chbox)
buttonArea.size_hint = (1.0, 0.1)

# カラーピッカー
def on_color(self, value):
    im1.color = value[:3]

clr_picker = ColorPicker()
clr_picker.bind(color=on_color)

# イメージエリア
img = fileRead('src/1.jpg')
im1 = ImageWidget(img=img)
im2 = Image(source='src/.tmp.png', allow_stretch=True)
set_im1(1)
imageArea = BoxLayout(orientation='horizontal')
imageArea.add_widget(im1)
imageArea.add_widget(im2)

# ルート
root = RootLayout(orientation='vertical')
clr_picker.size_hint_y = 0.3
buttonArea.size_hint_y = 0.05
imageArea.size_hint_y = 0.63
root.add_widget(clr_picker)
root.add_widget(buttonArea)
root.add_widget(imageArea)


if __name__ == "__main__":
    MainApp().run()