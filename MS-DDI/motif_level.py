import tensorflow as tf
from tensorflow import keras
import time
import numpy as np
import matplotlib.pyplot as plt
import sys

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

def create_padding_mask(batch_data):
    padding_mask = tf.cast(tf.math.equal(batch_data, 0), tf.float32)
    return padding_mask[:, tf.newaxis, tf.newaxis, :]


def scaled_dot_product_attention(q, k, v, mask, adjoin_matrix=None, dist_matrix=None):

    
    matmul_qk = tf.matmul(q, k, transpose_b=True)  
    
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    if mask is not None:
        
        try:
            scaled_attention_logits += (mask * -1e9)
        except tf.errors.InvalidArgumentError:
            batch_size = tf.shape(scaled_attention_logits)[0]
            seq_len_q = tf.shape(scaled_attention_logits)[2]
            seq_len_k = tf.shape(scaled_attention_logits)[3]
            
            new_mask = tf.zeros([batch_size, 1, seq_len_q, seq_len_k], dtype=tf.float32)
            scaled_attention_logits += new_mask
    
    if adjoin_matrix is not None:
        try:
            scaled_attention_logits = scaled_attention_logits * adjoin_matrix
        except tf.errors.InvalidArgumentError:
    
    if dist_matrix is not None:
        try:
            scaled_attention_logits = scaled_attention_logits * dist_matrix
        except tf.errors.InvalidArgumentError:
    
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

    def __init__(self, d_model, num_heads, dff, rate, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.mha1 = MultiHeadAttention(int(d_model/2), num_heads)
        self.mha2 = MultiHeadAttention(int(d_model/2), num_heads)
        self.ffn = feed_forward_network(d_model, dff)
        self.layer_norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
        

        self.attention_gate = self.add_weight(
            name='attention_gate',
            shape=(),  
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True
        )
        
        self.ffn_gate = self.add_weight(
            name='ffn_gate',
            shape=(),  
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True
        )
        
    def call(self, x, training, encoder_padding_mask, adjoin_matrix, dist_matrix):
        x1, x2 = tf.split(x, 2, -1)
        
        x_l, attention_weights_local = self.mha1(
            x1, x1, x1, encoder_padding_mask, adjoin_matrix, dist_matrix=None
        )
        
        x_g, attention_weights_global = self.mha2(
            x2, x2, x2, encoder_padding_mask, adjoin_matrix=None, dist_matrix=dist_matrix
        )
        
        attn_output = tf.concat([x_l, x_g], axis=-1)
        attn_output = self.dropout1(attn_output, training=training)
        
        gate_value = tf.sigmoid(self.attention_gate)  
        out1 = self.layer_norm1(x + gate_value * attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        
        gate_value = tf.sigmoid(self.ffn_gate)  
        out2 = self.layer_norm2(out1 + gate_value * ffn_output)
        
        return out2, attention_weights_local, attention_weights_global

class EncoderModel_motif(keras.layers.Layer):
    def __init__(self, num_layers, input_vocab_size,
                 d_model, num_heads, dff, rate=0.1,**kwargs):
        super(EncoderModel_motif, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = keras.layers.Embedding(input_vocab_size,
                                                self.d_model)
        self.dropout = keras.layers.Dropout(rate)
        self.encoder_layers = [
            EncoderLayer(int(d_model), num_heads, dff, rate)
            for _ in range(self.num_layers)]

    def call(self, x, training, atom_level_features, adjoin_matrix=None, dist_matrix=None):
        encoder_padding_mask = create_padding_mask(x) 
        if adjoin_matrix is not None:
            adjoin_matrix = adjoin_matrix[:,tf.newaxis,:,:]
        if dist_matrix is not None:
            dist_matrix = dist_matrix[:,tf.newaxis,:,:]
        x = self.embedding(x) 
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32)) 
        x = self.dropout(x, training=training)
        
        x_sliced = x[:,1:,:]
        seq_shape = tf.shape(x_sliced)
        atom_shape = tf.shape(atom_level_features)
        
        
        if seq_shape[1] != atom_shape[1]:
            if seq_shape[1] > atom_shape[1]:
                padding = seq_shape[1] - atom_shape[1]
                padding_tensor = tf.zeros(
                    [atom_shape[0], padding, atom_shape[2]], 
                    dtype=atom_level_features.dtype
                )
                atom_level_features = tf.concat([atom_level_features, padding_tensor], axis=1)
            else:
                x_sliced = x[:,1:atom_shape[1]+1,:]
        
        x_temp = x_sliced + atom_level_features  
        x = tf.concat([x[:,0:1,:], x_temp], axis=1)  
        
        attention_weights_list_local = []
        attention_weights_list_global = []
        for i in range(self.num_layers):
            x, attention_weights_local, attention_weights_global = self.encoder_layers[i](
                x, training, encoder_padding_mask, adjoin_matrix, dist_matrix=dist_matrix
            ) 
            attention_weights_list_local.append(attention_weights_local) 
            attention_weights_list_global.append(attention_weights_global)
        
        return x, attention_weights_list_local, attention_weights_list_global, encoder_padding_mask







