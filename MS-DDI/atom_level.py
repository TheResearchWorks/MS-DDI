import tensorflow as tf
from tensorflow import keras
import time
import numpy as np
import matplotlib.pyplot as plt

def rescale_distance_matrix(w): 
    constant_value = tf.constant(1.0,dtype=tf.float32) 
    return (constant_value+tf.math.exp(constant_value))/(constant_value+tf.math.exp(constant_value-w))

def gelu(x):
    return 0.5 * x * (1.0 + tf.math.erf(x / tf.sqrt(2.)))

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask_atom(batch_data):
    padding_mask = tf.cast(tf.math.equal(tf.reduce_sum(batch_data,axis=-1), 0), tf.float32)
    return padding_mask[:, tf.newaxis, tf.newaxis, :]

def scaled_dot_product_attention(q, k, v, mask,adjoin_matrix,dist_matrix):

    if dist_matrix is not None:
        matmul_qk = tf.nn.relu(tf.matmul(q, k, transpose_b = True))
        dist_matrix = rescale_distance_matrix(dist_matrix)

        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = (tf.multiply(matmul_qk,dist_matrix)) / tf.math.sqrt(dk)
    else:
        matmul_qk = tf.matmul(q, k, transpose_b = True)
        
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    if adjoin_matrix is not None:
        scaled_attention_logits += adjoin_matrix 

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  
    output = tf.matmul(attention_weights, v)

    return output, attention_weights

class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, d_model, num_heads,**kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask,adjoin_matrix,dist_matrix):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  
        k = self.wk(k) 
        v = self.wv(v)  
        q = self.split_heads(q, batch_size)  
        k = self.split_heads(k, batch_size)  
        v = self.split_heads(v, batch_size)  


        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask,adjoin_matrix,dist_matrix)
        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  
        output = self.dense(concat_attention)  
        return output, attention_weights

def feed_forward_network(d_model, dff):
    return keras.Sequential([
        keras.layers.Dense(dff, activation=gelu),
        keras.layers.Dense(d_model)
    ])
class EncoderLayer(keras.layers.Layer):

    def __init__(self, d_model, num_heads, dff,rate,**kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.mha1 = MultiHeadAttention(int(d_model/2), num_heads)
        self.mha2 = MultiHeadAttention(int(d_model/2), num_heads)
        self.ffn = feed_forward_network(d_model, dff)
        self.layer_norm1 = keras.layers.LayerNormalization(
            epsilon = 1e-6)
        self.layer_norm2 = keras.layers.LayerNormalization(
            epsilon = 1e-6)
        
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
    def call(self, x, training, encoder_padding_mask,adjoin_matrix,dist_matrix):
        x1,x2 = tf.split(x,2,-1)
        x_l,attention_weights_local = self.mha1(x1, x1, x1, encoder_padding_mask,adjoin_matrix,dist_matrix = None)
        x_g,attention_weights_global = self.mha2(x2, x2, x2, encoder_padding_mask,adjoin_matrix = None,dist_matrix = dist_matrix)
        attn_output = tf.concat([x_l,x_g],axis=-1)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layer_norm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layer_norm2(out1 + ffn_output)
        x_l_g = out2
        return x_l_g,attention_weights_local,attention_weights_global
        
class EncoderModel_atom(keras.layers.Layer):
    def __init__(self, num_layers, 
                 d_model, num_heads, dff, max_length=1000, 
                 scale_levels=3, rate=0.1, **kwargs):
        super(EncoderModel_atom, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_length = max_length
        self.scale_levels = scale_levels
        
        self.embedding = keras.layers.Dense(self.d_model, activation='relu')
        self.position_embedding = positional_encoding(max_length, self.d_model)
        
        self.scale_layers = []
        for i in range(scale_levels):
            kernel_size = 2**(i+1) - 1
            

            feature_dim = d_model // scale_levels
            

            self.scale_layers.append(keras.layers.Conv1D(
                filters=feature_dim,
                kernel_size=kernel_size,
                padding='same',  
                activation='relu',
                name=f'scale_conv_{kernel_size}'
            ))
        
        
        self.feature_fusion = keras.layers.Dense(d_model)
        
        self.position_gate = self.add_weight(
            name='position_gate',
            shape=(),
            initializer=tf.keras.initializers.Constant(0.5),
            trainable=True
        )
        
        self.multiscale_gate = self.add_weight(
            name='multiscale_gate',
            shape=(),
            initializer=tf.keras.initializers.Constant(0.7),  
            trainable=True
        )
        

        self.global_embedding = keras.layers.Dense(dff, activation='relu')
        self.global_gate = self.add_weight(
            name='global_gate',
            shape=(),
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True
        )
        

        self.layer_norm = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_multiscale = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = keras.layers.Dropout(rate)
        

        self.encoder_layers = [
            EncoderLayer(int(d_model), num_heads, dff, rate)
            for _ in range(self.num_layers)
        ]

    def call(self, x, training, adjoin_matrix=None,
             dist_matrix=None, atom_match_matrix=None, sum_atoms=None):

        encoder_padding_mask = create_padding_mask_atom(x)
        
        if adjoin_matrix is not None:
            adjoin_matrix = adjoin_matrix[:,tf.newaxis,:,:]
        if dist_matrix is not None:
            dist_matrix = dist_matrix[:,tf.newaxis,:,:]
        
        x_embedded = self.embedding(x)
        
        seq_len = tf.shape(x_embedded)[1]
        position_info = self.position_embedding[:, :seq_len, :]
        pos_gate_value = tf.sigmoid(self.position_gate)
        x_with_pos = x_embedded + pos_gate_value * position_info
        
        scale_features = []
        for scale_layer in self.scale_layers:
            scale_feature = scale_layer(x_with_pos)
            scale_features.append(scale_feature)
        
        multiscale_concat = tf.concat(scale_features, axis=-1)
        
        multiscale_features = self.feature_fusion(multiscale_concat)
        
        multiscale_features = self.layer_norm_multiscale(multiscale_features)
        
        ms_gate_value = tf.sigmoid(self.multiscale_gate)
        x = x_with_pos * (1 - ms_gate_value) + multiscale_features * ms_gate_value
        
        x = self.dropout(x, training=training)
        
        attention_weights_list_local = []
        attention_weights_list_global = []
        
        for i in range(self.num_layers):
            x, attn_local, attn_global = self.encoder_layers[i](
                x, training, encoder_padding_mask, adjoin_matrix, dist_matrix=dist_matrix
            )
            attention_weights_list_local.append(attn_local)
            attention_weights_list_global.append(attn_global)
        
        aggregated_x = tf.matmul(atom_match_matrix, x) / sum_atoms
        
        global_features = self.global_embedding(aggregated_x)
        global_gate_value = tf.sigmoid(self.global_gate)
        
        if aggregated_x.shape[-1] == global_features.shape[-1]:
            output = self.layer_norm(aggregated_x + global_gate_value * global_features)
        else:
            output = global_features
        
        return output, attention_weights_list_local, attention_weights_list_global, encoder_padding_mask

