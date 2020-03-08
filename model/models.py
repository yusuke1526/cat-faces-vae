import numpy as np
import os
from tensorflow.keras.layers import Add, Input, Dense, MaxPooling2D, Flatten, Lambda, Reshape, UpSampling2D, BatchNormalization, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from model.layers import CustomVariationalLayer
from tensorflow.keras.models import model_from_json


class VAE:
    def __init__(self, image_w, image_h, ch, latent_dim=50, intermediate_dim=256, filters=32, blocks=3, layers=3):
        self.image_w = image_w
        self.image_h = image_h
        self.ch = ch
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.filters = filters
        self.blocks = blocks
        self.kernel_size = 3
        self.layers = layers
        
        self.create_encoder()
        encoder_input = self.encoder.input
        z_mean, z_log_var = self.encoder(encoder_input)
        
        #sampling
        z = Lambda(self.sampling, output_shape=(self.latent_dim,), name='encoded')([z_mean, z_log_var])
        
        #decoder
        decoder_input = Input(K.int_shape(z)[1:])
        x = Dense(np.prod(self.shape_before_flattening[1:]), activation='relu')(decoder_input)
        x = Reshape(self.shape_before_flattening[1:])(x)
        for i in range(self.blocks):
            #x = Conv2DTranspose(filters, self.kernel_size, padding='same', activation='relu', strides=2)(x)
            x = UpSampling2D(2)(x)
            x = self.create_block(x, self.filters, en=False)
            x = BatchNormalization()(x)
            self.filters //= 2
        
        x = Conv2D(ch, self.kernel_size, padding='same', activation='sigmoid')(x)
        self.decoder = Model(decoder_input, x)
        z_decoded = self.decoder(z)
        
        
        cvl = CustomVariationalLayer()
        cvl.set_z([z_mean, z_log_var])
        cvl.set_original_dim(image_w * image_h * ch)
        y = cvl([encoder_input, z_decoded])
        self.vae = Model(encoder_input, y) # xをinputにyを出力, 出力は実質関係ない
        
    def create_block(self, input, filters, en):
        x = input
        for i in range(self.layers):
            x = Conv2D(filters, self.kernel_size, padding="same", activation='relu')(x)
            #x = LeakyReLU(0.2)(x)
        return x
        
    def sampling(self, args):
        z_mean = args[0]
        z_log_var = args[1]
        epsilon = K.random_normal(shape=(self.latent_dim,), mean=0., stddev=1.0)
        return z_mean + K.exp(z_log_var) * epsilon
    
    def get_vae(self):
        return self.vae
    
    def create_encoder(self):
        encoder_input = Input(shape=(self.image_w, self.image_h, self.ch))
        x = encoder_input
        for i in range(self.blocks):
            self.filters *= 2
            x = self.create_block(x, self.filters, en=True)
            x = MaxPooling2D(2)(x)
        self.shape_before_flattening = K.int_shape(x)
        h = Flatten()(x)
        h = Dense(self.intermediate_dim)(h)
        z_mean = Dense(self.latent_dim)(h) # 潜在変数の平均 μ
        z_log_var = Dense(self.latent_dim)(h) #潜在変数の分散 σのlog
        
        #sampling
        z = Lambda(self.sampling, output_shape=(self.latent_dim,), name='encoded')([z_mean, z_log_var])
        
        self.encoder = Model(encoder_input, [z_mean, z_log_var])
    
    def get_encoder(self):
        return self.encoder
    
    def get_decoder(self):
        return self.decoder
    
class VAE_inception(VAE):
    def __init__(self, image_w, image_h, ch, latent_dim=50, intermediate_dim=256, filters=32, blocks=3, layers=3):
        super().__init__(image_w, image_h, ch, latent_dim, intermediate_dim, filters, blocks, layers)
        
    def create_block(self, input, filters, en):
        x = input
        for i in range(self.layers):
            x3 = Conv2D(filters, 3, padding="same", activation='relu')(x)
            x5 = Conv2D(filters, 5, padding="same", activation='relu')(x)
            #x7 = Conv2D(filters, 7, padding="same", activation='relu')(x)
            x = Add()([x3, x5])
            #x = LeakyReLU(0.2)(x)
        return x
            

from tensorflow.keras.models import model_from_json

def save_model_and_weights(model, file_path):
    if os.path.exists(f'{file_path}.json') | os.path.exists(f'{file_path}.h5'):
        print(f'{file_path} already exists.')
    else:
        json_string = model.to_json()
        open(f'{file_path}.json', 'w').write(json_string)
        model.save_weights(f'{file_path}.h5')

def load_model_and_weights(file_path):
    if os.path.exists(f'{file_path}.json') & os.path.exists(f'{file_path}.h5'):
        json_string = open(f'{file_path}.json').read()
        model = model_from_json(json_string)
        model.load_weights(f'{file_path}.h5')
        return model
    else:
        print(f'{file_path} does not exist.')
        
        
