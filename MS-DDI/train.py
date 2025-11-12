import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
import json
from utils import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from dataset_processed import Graph_Bert_Dataset_fine_tune_DDI as Graph_Bert_Dataset_fine_tune
from cosine_annealing import cosine_annealing_with_warmup
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, Callback
from rdkit import RDLogger
import sys
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
from motif_level import * 
from atom_level import *
from model import build_model


def input_solver(sample, sample2, sample3, sample4, sample5, sample6, 
                sample7, sample8, sample9):
    drug_inputs = {
        'molecule_sequence1': sample,
        'adj_matrix1': sample2,
        'dist_matrix1': sample3,
        'atom_features1': sample4,
        'adjoin_matrix1_atom': sample5,
        'dist_matrix1_atom': sample6,
        'atom_match_matrix1': sample7,
        'sum_atoms1': sample8
    }
    return drug_inputs, sample9

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    pass

dataFolder = '/project/MSDDI/data/warm_start/DrugBank'
tr_dataset = pd.read_csv(dataFolder + '/train.csv')   
val_dataset = pd.read_csv(dataFolder + '/val.csv')
tst_dataset = pd.read_csv(dataFolder + '/test.csv')

tokenizer = Mol_Tokenizer('/project/MSDDI/data/drugbank/token_id.json')
map_dict = np.load('/project/MSDDI/data/drugbank/preprocessed_drug_info.npy',allow_pickle=True).item()

train_dataset_, validation_dataset_, test_dataset_ = Graph_Bert_Dataset_fine_tune(
    train_set=tr_dataset,        
    val_set=val_dataset,         
    tst_set=tst_dataset,         
    tokenizer=tokenizer,
    batch_size=256,
    map_dict=map_dict,
    drug_a_field='drug_A',       
    drug_b_field='drug_B',       
    label_field='DDI'           
).get_data()

train_dataset1 = train_dataset_.map(input_solver)
val_dataset1 = validation_dataset_.map(input_solver) 
test_dataset1 = test_dataset_.map(input_solver)

param = {'name': 'Small', 'num_layers':1, 'num_heads':128 , 'd_model': 256}

class TrainingHistorySaver(Callback):
    
    def __init__(self, json_file_path):
        super().__init__()
        self.json_file_path = json_file_path
        self.training_history = {
            'accuracy': [],
            'val_accuracy': [],
            'loss': [],
            'val_loss': []
        }
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            self.training_history['accuracy'].append(float(logs.get('accuracy', 0)))
            self.training_history['val_accuracy'].append(float(logs.get('val_accuracy', 0)))
            self.training_history['loss'].append(float(logs.get('loss', 0)))
            self.training_history['val_loss'].append(float(logs.get('val_loss', 0)))
            
            try:
                with open(self.json_file_path, 'w') as f:
                    json.dump(self.training_history, f, indent=2)
            except Exception as e:
                pass
    
    def on_train_end(self, logs=None):
        try:
            with open(self.json_file_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
        except Exception as e:
            pass

arch = param
num_layers = arch['num_layers']
num_heads = arch['num_heads']
d_model = arch['d_model']*2 
dff = d_model 
input_vocab_size = tokenizer.get_vocab_size
dropout_rate = 0.1
training = True 


models = build_model(
    tokenizer, 
    d_model=d_model, 
    num_motif_layers=num_layers, 
    num_atom_layers=num_layers, 
    num_heads=num_heads, 
    dropout_rate=dropout_rate,
    training=training
)

models.summary()

total_epochs = 100    
warmup_epochs = 5       
initial_lr = 0.0001     
min_lr = 1e-7           

lr_scheduler = LearningRateScheduler(
    lambda epoch: cosine_annealing_with_warmup(
        epoch, total_epochs, warmup_epochs, initial_lr, min_lr
    ),
    verbose=0
)

checkpoint_path = '/project/MSDDI/save_weights/ckpt'


checkpoint = ModelCheckpoint(
    checkpoint_path, 
    monitor='val_accuracy',                          
    verbose=1,                                  
    save_best_only=True,                        
    mode='max',                                 
    save_weights_only=True,                     
    save_format='tf'
)


opt = Adam(learning_rate=initial_lr)

loss = tf.keras.losses.BinaryCrossentropy() 

models.compile(
    loss=loss, 
    optimizer=opt,
    metrics=[
        'accuracy',                                 
        tf.keras.metrics.AUC(name='auc')             
    ]
)

callbacks = [checkpoint, lr_scheduler]

history = models.fit(
    train_dataset1,
    epochs=total_epochs,
    callbacks=callbacks,
    validation_data=val_dataset1
)





