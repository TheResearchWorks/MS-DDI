from cProfile import label
import pandas as pd
import numpy as np
import tensorflow as tf
import networkx as nx
from rdkit import Chem
from utils import *

def molgraph_rep(smi,cliques):
    def atom_to_motif_match(atom_order,cliques):
        atom_order = atom_order.tolist()
        temp_matrix = np.zeros((len(cliques),len(atom_order)))
        for th,cli in enumerate(cliques):
            for i in cli:
                temp_matrix[th,atom_order.index(i)] = 1
        return temp_matrix
    def get_adj_dist_matrix(mol_graph,smi):
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        num_atoms = mol.GetNumAtoms() 
        adjoin_matrix_temp = np.eye(num_atoms)
        adj_matrix = Chem.GetAdjacencyMatrix(mol)
        adj_matrix = (adjoin_matrix_temp + adj_matrix)[:,mol_graph['rdkit_ix']][mol_graph['rdkit_ix']]
        dist_matrix = Chem.GetDistanceMatrix(mol)[:,mol_graph['rdkit_ix']][mol_graph['rdkit_ix']]
        return adj_matrix,dist_matrix 
    single_dict = {'input_atom_features':[],
            'atom_match_matrix':[],
            'sum_atoms':[],
            'adj_matrix':[],
            'dist_matrix':[]
            }
    array_rep = array_rep_from_smiles(smi)
    summed_degrees = extract_bondfeatures_of_neighbors_by_degree(array_rep)  
    atom_features = array_rep['atom_features'] 
    all_bond_features = []
    for degree in degrees:
        atom_neighbors_list = array_rep[('atom_neighbors', degree)].astype('int32')
        if len(atom_neighbors_list)==0:
                true_summed_degree = np.zeros((atom_features.shape[0], 10),'float32')
        else:
                true_summed_degree = bond_features_by_degree(atom_features.shape[0],summed_degrees,degree) 
        all_bond_features.append(true_summed_degree) 
    single_dict['atom_match_matrix'] = atom_to_motif_match(array_rep['rdkit_ix'],cliques)
    single_dict['sum_atoms'] = np.reshape(np.sum(single_dict['atom_match_matrix'],axis=1),(-1,1))
    out_bond_features = 0
    for arr in all_bond_features:
        out_bond_features = out_bond_features + arr
    single_dict['input_atom_features'] = np.concatenate([atom_features,out_bond_features],axis=1) 
    adj_matrix,dist_matrix = get_adj_dist_matrix(array_rep,smi)
    single_dict['adj_matrix'] = adj_matrix
    single_dict['dist_matrix'] = dist_matrix 
    single_dict = {key:np.array(value,dtype='float32') for key,value in single_dict.items()} 
    return single_dict

class Graph_Bert_Dataset_fine_tune(object):
    def __init__(self, train_set, val_set, tst_set, tokenizer, batch_size, map_dict, drug_field='drug', label_field='serious'):
        
        self.train_set = train_set
        self.val_set = val_set
        self.tst_set = tst_set
        self.train_set[label_field] = self.train_set[label_field].map(int)
        self.val_set[label_field] = self.val_set[label_field].map(int)
        self.tst_set[label_field] = self.tst_set[label_field].map(int)
        self.label_field = label_field
        self.drug_field = drug_field
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.pad_value = self.tokenizer.vocab['<pad>']
        self.map_dict = map_dict

    def get_data(self):
        self.dataset1 = tf.data.Dataset.from_tensor_slices((self.train_set[self.drug_field], self.train_set[self.label_field]))
        self.dataset1 = self.dataset1.map(self.tf_numerical_seq).padded_batch(self.batch_size, 
            padding_values=(tf.constant(0, dtype=tf.int64),
                tf.constant(0, dtype=tf.float32), tf.constant(-1e9, dtype=tf.float32),
                tf.constant(0, dtype=tf.float32), tf.constant(0, dtype=tf.float32),
                tf.constant(-1e9, dtype=tf.float32), tf.constant(0, dtype=tf.float32),
                tf.constant(1, dtype=tf.float32), tf.constant(0, dtype=tf.int64)),
            padded_shapes=(
                tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([None, None]),
                tf.TensorShape([None, None]), tf.TensorShape([None, None]), 
                tf.TensorShape([None, None]), tf.TensorShape([None, None]),
                tf.TensorShape([None, None]), tf.TensorShape([1]))).prefetch(20)

        self.dataset2 = tf.data.Dataset.from_tensor_slices((self.val_set[self.drug_field], self.val_set[self.label_field]))
        self.dataset2 = self.dataset2.map(self.tf_numerical_seq).padded_batch(self.batch_size,
            padding_values=(tf.constant(0, dtype=tf.int64),
                tf.constant(0, dtype=tf.float32), tf.constant(-1e9, dtype=tf.float32),
                tf.constant(0, dtype=tf.float32), tf.constant(0, dtype=tf.float32),
                tf.constant(-1e9, dtype=tf.float32), tf.constant(0, dtype=tf.float32),
                tf.constant(1, dtype=tf.float32), tf.constant(0, dtype=tf.int64)),
            padded_shapes=(
                tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([None, None]),
                tf.TensorShape([None, None]), tf.TensorShape([None, None]), 
                tf.TensorShape([None, None]), tf.TensorShape([None, None]),
                tf.TensorShape([None, None]), tf.TensorShape([1]))).prefetch(20)

        self.dataset3 = tf.data.Dataset.from_tensor_slices((self.tst_set[self.drug_field], self.tst_set[self.label_field]))
        self.dataset3 = self.dataset3.map(self.tf_numerical_seq).padded_batch(self.batch_size,
            padding_values=(tf.constant(0, dtype=tf.int64),
                tf.constant(0, dtype=tf.float32), tf.constant(-1e9, dtype=tf.float32),
                tf.constant(0, dtype=tf.float32), tf.constant(0, dtype=tf.float32),
                tf.constant(-1e9, dtype=tf.float32), tf.constant(0, dtype=tf.float32),
                tf.constant(1, dtype=tf.float32), tf.constant(0, dtype=tf.int64)),
            padded_shapes=(
                tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([None, None]),
                tf.TensorShape([None, None]), tf.TensorShape([None, None]), 
                tf.TensorShape([None, None]), tf.TensorShape([None, None]),
                tf.TensorShape([None, None]), tf.TensorShape([1]))).prefetch(20)
                
        return self.dataset1, self.dataset2, self.dataset3

    def process_single_molecule(self, smiles):
        nums_list = self.map_dict[smiles]['nums_list'] 
        dist_matrix = self.map_dict[smiles]['dist_matrix'] 
        adjoin_matrix = self.map_dict[smiles]['adj_matrix']
        single_dict_atom = self.map_dict[smiles]['single_dict']
        
        nums_list = [self.tokenizer.vocab['<global>']] + nums_list
        
        temp = np.ones((len(nums_list), len(nums_list)))
        temp[1:, 1:] = adjoin_matrix
        adjoin_matrix = (1 - temp) * (-1e9)
        
        temp_dist = np.ones((len(nums_list), len(nums_list)))
        temp_dist[0][0] = 0
        temp_dist[1:, 1:] = dist_matrix
        dist_matrix = temp_dist
        
        atom_features = single_dict_atom['input_atom_features']
        dist_matrix_atom = single_dict_atom['dist_matrix']
        adjoin_matrix_atom = single_dict_atom['adj_matrix']
        adjoin_matrix_atom = (1 - adjoin_matrix_atom) * (-1e9) 
        atom_match_matrix = single_dict_atom['atom_match_matrix']
        sum_atoms = single_dict_atom['sum_atoms']
        
        x = np.array(nums_list).astype('int64')
        
        return {
            'x': x,
            'adjoin_matrix': adjoin_matrix,
            'dist_matrix': dist_matrix,
            'atom_features': atom_features,
            'adjoin_matrix_atom': adjoin_matrix_atom,
            'dist_matrix_atom': dist_matrix_atom,
            'atom_match_matrix': atom_match_matrix,
            'sum_atoms': sum_atoms
        }


    def merge_molecular_features(self, mol_features_list):

        x_combined = np.concatenate([feat['x'] for feat in mol_features_list])
        
        total_size = sum(feat['adjoin_matrix'].shape[0] for feat in mol_features_list)
        
        adjoin_matrix_combined = np.full((total_size, total_size), 0.0, dtype=np.float32)
        dist_matrix_combined = np.full((total_size, total_size), -1e9, dtype=np.float32)
        
        total_atoms = sum(feat['atom_features'].shape[0] for feat in mol_features_list)
        atom_features_combined = np.zeros((total_atoms, mol_features_list[0]['atom_features'].shape[1]), dtype=np.float32)
        
        adjoin_atom_combined = np.full((total_atoms, total_atoms), 0.0, dtype=np.float32)
        dist_atom_combined = np.full((total_atoms, total_atoms), -1e9, dtype=np.float32)
        
        total_fragments = sum(feat['atom_match_matrix'].shape[0] for feat in mol_features_list)
        atom_match_combined = np.zeros((total_fragments, total_atoms), dtype=np.float32)
        
        sum_atoms_combined = np.zeros((total_fragments, 1), dtype=np.float32)
        
        node_offset = 0
        atom_offset = 0
        frag_offset = 0
        
        for feat in mol_features_list:
            n = feat['adjoin_matrix'].shape[0]
            a = feat['atom_features'].shape[0]
            f = feat['atom_match_matrix'].shape[0]
            
            adjoin_matrix_combined[node_offset:node_offset+n, node_offset:node_offset+n] = feat['adjoin_matrix']
            dist_matrix_combined[node_offset:node_offset+n, node_offset:node_offset+n] = feat['dist_matrix']
            
            atom_features_combined[atom_offset:atom_offset+a] = feat['atom_features']
            adjoin_atom_combined[atom_offset:atom_offset+a, atom_offset:atom_offset+a] = feat['adjoin_matrix_atom']
            dist_atom_combined[atom_offset:atom_offset+a, atom_offset:atom_offset+a] = feat['dist_matrix_atom']
            
            atom_match_combined[frag_offset:frag_offset+f, atom_offset:atom_offset+a] = feat['atom_match_matrix']
            sum_atoms_combined[frag_offset:frag_offset+f] = feat['sum_atoms']
            
            node_offset += n
            atom_offset += a
            frag_offset += f
        
        return (x_combined, adjoin_matrix_combined, dist_matrix_combined, 
                atom_features_combined, adjoin_atom_combined, dist_atom_combined,
                atom_match_combined, sum_atoms_combined)

    
    def numerical_seq(self, smiles_combined, labels):
        smiles_str = smiles_combined.numpy().decode().strip()
        smiles_list = [s.strip() for s in smiles_str.split(',')]
        
        mol_features = []
        for smiles in smiles_list:
            try:
                features = self.process_single_molecule(smiles)
                mol_features.append(features)
            except Exception as e:
                continue
        
        if not mol_features:
            raise ValueError(f"No valid molecules in: {smiles_str}")
            
        merged_features = self.merge_molecular_features(mol_features)

        y = np.array([labels]).astype('int64')
        return (*merged_features, y)

    def tf_numerical_seq(self, smiles, labels):
        x, adjoin_matrix, dist_matrix, atom_features, adjoin_matrix_atom, \
        dist_matrix_atom, atom_match_matrix, sum_atoms, y = tf.py_function(
            self.numerical_seq, 
            [smiles, labels],
            [tf.int64, tf.float32, tf.float32, tf.float32, tf.float32, 
             tf.float32, tf.float32, tf.float32, tf.int64]
        )
        
        x.set_shape([None])
        adjoin_matrix.set_shape([None, None])
        dist_matrix.set_shape([None, None])
        atom_features.set_shape([None, None])
        adjoin_matrix_atom.set_shape([None, None])
        dist_matrix_atom.set_shape([None, None])
        atom_match_matrix.set_shape([None, None])
        sum_atoms.set_shape([None, None])
        y.set_shape([None])
        
        return x, adjoin_matrix, dist_matrix, atom_features, adjoin_matrix_atom, \
               dist_matrix_atom, atom_match_matrix, sum_atoms, y


class Graph_Bert_Dataset_fine_tune_DDI(Graph_Bert_Dataset_fine_tune):
    def __init__(self, train_set, val_set, tst_set, tokenizer, batch_size, map_dict, 
                 drug_a_field='drug_A', drug_b_field='drug_B', label_field='DDI'):
        
        self.drug_a_field = drug_a_field
        self.drug_b_field = drug_b_field
        self.original_label_field = label_field
        
        train_processed = self._preprocess_ddi_data(train_set, drug_a_field, drug_b_field, label_field)
        val_processed = self._preprocess_ddi_data(val_set, drug_a_field, drug_b_field, label_field)
        tst_processed = self._preprocess_ddi_data(tst_set, drug_a_field, drug_b_field, label_field)
        
        super().__init__(
            train_set=train_processed,
            val_set=val_processed, 
            tst_set=tst_processed,
            tokenizer=tokenizer,
            batch_size=batch_size,
            map_dict=map_dict,
            drug_field='drug_combined',
            label_field='reactionoutcome'
        )
    
    def _preprocess_ddi_data(self, dataset, drug_a_field, drug_b_field, label_field):
        required_cols = [drug_a_field, drug_b_field, label_field]
        missing_cols = [col for col in required_cols if col not in dataset.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        processed_df = dataset.copy()
        
        processed_df['drug_combined'] = (
            dataset[drug_a_field].astype(str) + ',' + 
            dataset[drug_b_field].astype(str)
        )
        
        processed_df['reactionoutcome'] = dataset[label_field]
        
        return processed_df
    
    def get_data(self):
        self.dataset1 = tf.data.Dataset.from_tensor_slices((
            self.train_set[self.drug_field], 
            self.train_set[self.label_field]
        ))
        self.dataset1 = self.dataset1.map(self.tf_numerical_seq).padded_batch(
            self.batch_size,
            padding_values=(
                tf.constant(0, dtype=tf.int64),
                tf.constant(0, dtype=tf.float32), tf.constant(-1e9, dtype=tf.float32),
                tf.constant(0, dtype=tf.float32), tf.constant(0, dtype=tf.float32),
                tf.constant(-1e9, dtype=tf.float32), tf.constant(0, dtype=tf.float32),
                tf.constant(1, dtype=tf.float32), 
                tf.constant(0, dtype=tf.int64)
            ),
            padded_shapes=(
                tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([None, None]),
                tf.TensorShape([None, None]), tf.TensorShape([None, None]), 
                tf.TensorShape([None, None]), tf.TensorShape([None, None]),
                tf.TensorShape([None, None]), 
                tf.TensorShape([1])
            )
        ).prefetch(20)

        self.dataset2 = tf.data.Dataset.from_tensor_slices((
            self.val_set[self.drug_field], 
            self.val_set[self.label_field]
        ))
        self.dataset2 = self.dataset2.map(self.tf_numerical_seq).padded_batch(
            self.batch_size,
            padding_values=(
                tf.constant(0, dtype=tf.int64),
                tf.constant(0, dtype=tf.float32), tf.constant(-1e9, dtype=tf.float32),
                tf.constant(0, dtype=tf.float32), tf.constant(0, dtype=tf.float32),
                tf.constant(-1e9, dtype=tf.float32), tf.constant(0, dtype=tf.float32),
                tf.constant(1, dtype=tf.float32), 
                tf.constant(0, dtype=tf.int64)
            ),
            padded_shapes=(
                tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([None, None]),
                tf.TensorShape([None, None]), tf.TensorShape([None, None]), 
                tf.TensorShape([None, None]), tf.TensorShape([None, None]),
                tf.TensorShape([None, None]), 
                tf.TensorShape([1])
            )
        ).prefetch(20)

        self.dataset3 = tf.data.Dataset.from_tensor_slices((
            self.tst_set[self.drug_field], 
            self.tst_set[self.label_field]
        ))
        self.dataset3 = self.dataset3.map(self.tf_numerical_seq).padded_batch(
            self.batch_size,
            padding_values=(
                tf.constant(0, dtype=tf.int64),
                tf.constant(0, dtype=tf.float32), tf.constant(-1e9, dtype=tf.float32),
                tf.constant(0, dtype=tf.float32), tf.constant(0, dtype=tf.float32),
                tf.constant(-1e9, dtype=tf.float32), tf.constant(0, dtype=tf.float32),
                tf.constant(1, dtype=tf.float32), 
                tf.constant(0, dtype=tf.int64)
            ),
            padded_shapes=(
                tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([None, None]),
                tf.TensorShape([None, None]), tf.TensorShape([None, None]), 
                tf.TensorShape([None, None]), tf.TensorShape([None, None]),
                tf.TensorShape([None, None]), 
                tf.TensorShape([1])
            )
        ).prefetch(20)
                
        return self.dataset1, self.dataset2, self.dataset3

    def numerical_seq(self, smiles_combined, labels):
        x, adjoin_matrix, dist_matrix, atom_features, adjoin_matrix_atom, \
        dist_matrix_atom, atom_match_matrix, sum_atoms = self.merge_molecular_features(
            self._process_drug_smiles(smiles_combined.numpy().decode().strip())
        )
        
        y = np.array([labels]).astype('int64')
        
        return (x, adjoin_matrix, dist_matrix, atom_features, adjoin_matrix_atom,
                dist_matrix_atom, atom_match_matrix, sum_atoms, y)

    def tf_numerical_seq(self, smiles, labels):
        output_types = [
            tf.int64, tf.float32, tf.float32, tf.float32, tf.float32, 
            tf.float32, tf.float32, tf.float32, tf.int64
        ]
        
        outputs = tf.py_function(
            self.numerical_seq, 
            [smiles, labels],
            output_types
        )
        
        outputs[0].set_shape([None])
        outputs[1].set_shape([None, None])
        outputs[2].set_shape([None, None])
        outputs[3].set_shape([None, None])
        outputs[4].set_shape([None, None])
        outputs[5].set_shape([None, None])
        outputs[6].set_shape([None, None])
        outputs[7].set_shape([None, None])
        outputs[8].set_shape([None])
        
        return outputs
    
    def _process_drug_smiles(self, smiles_str):
        smiles_list = [s.strip() for s in smiles_str.split(',')]
        
        mol_features = []
        for smiles in smiles_list:
            if smiles:
                try:
                    features = self.process_single_molecule(smiles)
                    mol_features.append(features)
                except Exception as e:
                    continue
        
        if not mol_features:
            raise ValueError(f"No valid molecules in: {smiles_str}")
            
        return mol_features

