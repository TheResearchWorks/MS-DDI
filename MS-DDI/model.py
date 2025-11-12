
import os
import tensorflow as tf
import logging
from utils import *
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from rdkit import RDLogger
import sys
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

from motif_level import * 
from atom_level import *
from maco_attention import MACoAttention  
from tensorflow.keras.layers import MultiHeadAttention


def build_model(
    tokenizer, d_model=512, num_motif_layers=4, num_atom_layers=3, 
    num_heads=8, dropout_rate=0.2, training=True, return_attention=False):

    motif_inputs1 = Input(shape=(None,), name="molecule_sequence1")
    motif_adj_inputs1 = Input(shape=(None,None), name="adj_matrix1") 
    motif_dist_inputs1 = Input(shape=(None,None), name="dist_matrix1")

    atom_inputs1 = Input(shape=(None,61), name="atom_features1")
    atom_adj_inputs1 = Input(shape=(None,None), name="adjoin_matrix1_atom") 
    atom_dist_inputs1 = Input(shape=(None,None), name="dist_matrix1_atom")
    atom_match_matrix1 = Input(shape=(None,None), name="atom_match_matrix1")
    sum_atoms1 = Input(shape=(None,None), name="sum_atoms1")
    
    atom_encoder = EncoderModel_atom(
        num_layers=num_atom_layers, 
        d_model=d_model,  
        dff=d_model, 
        num_heads=num_heads
    )
    
    motif_encoder = EncoderModel_motif(
        num_layers=num_motif_layers, 
        d_model=d_model, 
        dff=d_model*2,
        num_heads=num_heads, 
        input_vocab_size=tokenizer.get_vocab_size
    )
    
    Outseq_atom, *_, encoder_padding_mask_atom = atom_encoder(
        atom_inputs1, adjoin_matrix=atom_adj_inputs1,
        dist_matrix=atom_dist_inputs1, atom_match_matrix=atom_match_matrix1,
        sum_atoms=sum_atoms1, training=training
    )
    
    Outseq_motif, *_, encoder_padding_mask_motif = motif_encoder(
        motif_inputs1, adjoin_matrix=motif_adj_inputs1,
        dist_matrix=motif_dist_inputs1, atom_level_features=Outseq_atom, 
        training=training
    )


    motif_cross_attention = MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=d_model // num_heads, 
        dropout=dropout_rate,
        name="cross_attention_motif_to_atom"
    )
    attn_output1 = motif_cross_attention(
        query=Outseq_motif, 
        value=Outseq_atom, 
        key=Outseq_atom, 
        training=training,
        return_attention_scores=return_attention  
    )
    
    
    enhanced_motif1 = LayerNormalization(epsilon=1e-6)(Outseq_motif + attn_output1)
    
    atom_cross_attention = MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=d_model // num_heads, 
        dropout=dropout_rate,
        name="cross_attention_atom_to_motif"
    )
    attn_output2 = atom_cross_attention(
        query=Outseq_atom, 
        value=Outseq_motif, 
        key=Outseq_motif, 
        training=training,
        return_attention_scores=return_attention
    )
    
    
    enhanced_atom1 = LayerNormalization(epsilon=1e-6)(Outseq_atom + attn_output2)
    
    maco_attention = MACoAttention(d_model=d_model, k_dim=d_model // 2, name="co_attention_block")
    motif_pooled, atom_pooled = maco_attention([enhanced_motif1, enhanced_atom1])

    gate = Dense(d_model, activation='sigmoid')(tf.concat([motif_pooled, atom_pooled], axis=-1))
    combined_features = gate * motif_pooled + (1-gate) * Dense(d_model)(atom_pooled)
    
    x = Dense(d_model, activation='gelu')(combined_features)
    x = Dropout(dropout_rate)(x, training=training)
    x = BatchNormalization()(x, training=training)
    
    skip_connection = x
    
    x = Dense(d_model//2, activation='gelu')(x)
    x = Dropout(dropout_rate)(x, training=training)
    x = BatchNormalization()(x, training=training)
    x = Dense(d_model//4, activation='gelu')(x)
    
    skip_proj = Dense(d_model//4)(skip_connection)
    x = x + skip_proj
    
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dropout(dropout_rate)(x, training=training)
    
    output = Dense(1, activation='sigmoid')(x)
    
    all_inputs = [
        atom_inputs1, atom_adj_inputs1, atom_dist_inputs1,
        atom_match_matrix1, sum_atoms1, motif_inputs1, 
        motif_adj_inputs1, motif_dist_inputs1
    ]
    
    if return_attention:
        attention_dict = {
            'motif_to_atom': attn_weights_motif_to_atom,
            'atom_to_motif': attn_weights_atom_to_motif
        }
        model = Model(inputs=all_inputs, outputs=[output, attention_dict])
    else:
        model = Model(inputs=all_inputs, outputs=[output])
    
    return model

