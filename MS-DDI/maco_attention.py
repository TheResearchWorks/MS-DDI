import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Dense, Dropout, LayerNormalization, BatchNormalization, MultiHeadAttention
from tensorflow.keras.models import Model


class MACoAttention(Layer):
    def __init__(self, d_model, k_dim, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.k_dim = k_dim

    def build(self, input_shape):
        self.W_v = self.add_weight(shape=(self.k_dim, self.d_model), initializer='glorot_uniform', name='W_v')
        self.W_q = self.add_weight(shape=(self.k_dim, self.d_model), initializer='glorot_uniform', name='W_q')
        self.W_m = self.add_weight(shape=(self.k_dim, self.d_model), initializer='glorot_uniform', name='W_m')
        self.W_h = self.add_weight(shape=(1, self.k_dim), initializer='glorot_uniform', name='W_h')
        super().build(input_shape)

    def call(self, inputs):
        V_n, Q_n = inputs[0], inputs[1]

        V_0 = tf.reduce_mean(V_n, axis=1, keepdims=True)
        Q_0 = tf.reduce_mean(Q_n, axis=1, keepdims=True)

        V_r = tf.transpose(V_n, [0, 2, 1])
        Q_r = tf.transpose(Q_n, [0, 2, 1])
        V_0_t = tf.transpose(V_0, [0, 2, 1])
        Q_0_t = tf.transpose(Q_0, [0, 2, 1])

        M_0 = tf.tanh(tf.multiply(V_0_t, Q_0_t))

        H_v = tf.multiply(tf.tanh(tf.matmul(self.W_v, V_r)), tf.tanh(tf.matmul(self.W_m, M_0)))
        H_q = tf.multiply(tf.tanh(tf.matmul(self.W_q, Q_r)), tf.tanh(tf.matmul(self.W_m, M_0)))

        alpha_v = tf.nn.softmax(tf.matmul(self.W_h, H_v), axis=-1)
        alpha_q = tf.nn.softmax(tf.matmul(self.W_h, H_q), axis=-1)

        vector_v = tf.matmul(alpha_v, V_n)
        vector_q = tf.matmul(alpha_q, Q_n)

        return tf.squeeze(vector_v, axis=1), tf.squeeze(vector_q, axis=1)
