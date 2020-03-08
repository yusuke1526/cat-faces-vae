class VAE:
    def __init__(self, image_w, image_h, ch, latent_dim=50, intermediate_dim=256, filters=32, blocks=3, layers=3):
        self.latent_dim = latent_dim
        self.kernel_size = 3
        self.layers = layers
        
        #encoder
        encoder_input = Input(shape=(image_w, image_h, ch))
        x = encoder_input
        for i in range(blocks):
            filters *= 2
            #x = BatchNormalization()(x)
            x = self.create_block(x, filters, en=True)
            x = MaxPooling2D(2)(x)
        shape_before_flattening = K.int_shape(x)
        h = Flatten()(x)
        h = Dense(intermediate_dim)(h)
        z_mean = Dense(latent_dim)(h) # 潜在変数の平均 μ
        z_log_var = Dense(latent_dim)(h) #潜在変数の分散 σのlog
        
        #sampling
        z = Lambda(self.sampling, output_shape=(self.latent_dim,), name='encoded')([z_mean, z_log_var])
        
        self.encoder = Model(encoder_input, [z_mean, z_log_var])
        
        #decoder
        decoder_input = Input(K.int_shape(z)[1:])
        x = Dense(np.prod(shape_before_flattening[1:]), activation='relu')(decoder_input)
        x = Reshape(shape_before_flattening[1:])(x)
        for i in range(blocks):
            #x = Conv2DTranspose(filters, self.kernel_size, padding='same', activation='relu', strides=2)(x)
            x = UpSampling2D(2)(x)
            x = self.create_block(x, filters, en=False)
            #x = BatchNormalization()(x)
            filters //= 2
        
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
        encoder_input = Input(shape=(image_w, image_h, ch))
        x = encoder_input
        for i in range(blocks):
            filters *= 2
            x = self.create_block(x, filters, en=True)
            x = MaxPooling2D(2)(x)
        shape_before_flattening = K.int_shape(x)
        h = Flatten()(x)
        h = Dense(intermediate_dim)(h)
        z_mean = Dense(latent_dim)(h) # 潜在変数の平均 μ
        z_log_var = Dense(latent_dim)(h) #潜在変数の分散 σのlog
        
        #sampling
        z = Lambda(self.sampling, output_shape=(self.latent_dim,), name='encoded')([z_mean, z_log_var])
        
        self.encoder = Model(encoder_input, [z_mean, z_log_var])
    
    def get_encoder(self):
        return self.encoder
    
    def get_decoder(self):
        return self.decoder