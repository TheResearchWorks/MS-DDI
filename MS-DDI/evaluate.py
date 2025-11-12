import tensorflow as tf
import numpy as np
import pandas as pd
from utils import *
from dataset_processed import Graph_Bert_Dataset_fine_tune_DDI as Graph_Bert_Dataset_fine_tune
from rdkit import RDLogger
from model import build_model
from sklearn import metrics

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


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

dataFolder = '/project/MSDDI/data/DrugBank'
val_dataset = pd.read_csv(dataFolder + '/val.csv')
tokenizer = Mol_Tokenizer('/project/MSDDI/data/drugbank/token_id.json')
map_dict = np.load('/project/MSDDI/data/drugbank/preprocessed_drug_info.npy',allow_pickle=True).item()

empty_df = pd.DataFrame(columns=['drug_A', 'drug_B', 'DDI'])


_, validation_dataset_, _ = Graph_Bert_Dataset_fine_tune(
    train_set=empty_df,
    val_set=val_dataset,
    tst_set=empty_df,
    tokenizer=tokenizer,
    batch_size=256,
    map_dict=map_dict,
    drug_a_field='drug_A',
    drug_b_field='drug_B',
    label_field='DDI'
).get_data()

val_dataset1 = validation_dataset_.map(input_solver)

param = {'name': 'Small', 'num_layers':1, 'num_heads':128 , 'd_model': 256}

arch = param
num_layers = arch['num_layers']
num_heads = arch['num_heads']
d_model = arch['d_model']*2 
dff = d_model 
input_vocab_size = tokenizer.get_vocab_size
dropout_rate = 0.1
training = False 

models = build_model(
    tokenizer, 
    d_model=d_model, 
    num_motif_layers=num_layers, 
    num_atom_layers=num_layers, 
    num_heads=num_heads, 
    dropout_rate=dropout_rate,
    training=training,
)

models.summary()

checkpoint_path = '/project/MSDDI/save_weights/ckpt'

models.load_weights(checkpoint_path)


def evaluation(preds, truths, ndigits=4, threshold=0.5):

    pred_scores = preds.flatten()
    pred_labels = (pred_scores > threshold).astype(int)

    acc = metrics.accuracy_score(truths, pred_labels)
    auc = metrics.roc_auc_score(truths, pred_scores)
    f1 = metrics.f1_score(truths, pred_labels)
    precision = metrics.precision_score(truths, pred_labels)
    recall = metrics.recall_score(truths, pred_labels)
    ap = metrics.average_precision_score(truths, pred_scores)

    print("ACC:   {:.{}f}".format(acc, ndigits))
    print("AUC:   {:.{}f}".format(auc, ndigits))
    print("F1:    {:.{}f}".format(f1, ndigits))
    print("Prec:  {:.{}f}".format(precision, ndigits))
    print("Rec:   {:.{}f}".format(recall, ndigits))
    print("AP:    {:.{}f} (AUPR)".format(ap, ndigits))


    
pred_res = models.predict(val_dataset1, verbose=0) 
labels = val_dataset['DDI'].tolist() 

evaluation(pred_res, labels)
