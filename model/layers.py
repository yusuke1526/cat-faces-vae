from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras import metrics

class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded):
        x = K.flatten(x)
        x_decoded = K.flatten(x_decoded)
        xent_loss = self.original_dim * metrics.binary_crossentropy(x, x_decoded) # 復元誤差: Reconstruction Error
        kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1) # 正則化項: KL Divergence
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded = inputs[1]
        loss = self.vae_loss(x, x_decoded)
        self.add_loss(loss, inputs=inputs) # Layer class のadd_lossを利用
        return x # 実質的には出力は利用しない
    
    def set_z(self, args):
        self.z_mean, self.z_log_var = args
        
    def set_original_dim(self, original_dim):
        self.original_dim = original_dim