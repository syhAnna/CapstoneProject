import numpy as np
import pandas as pd
from tqdm import tqdm
from annoy import AnnoyIndex
from utils import MODEL_ROOT, loadData, saveData

NUM_TREES = 100


def annoyID(sequences, fname):
    """
    Inplement annoy algorithm
    """
    annoy = AnnoyIndex(sequences.shape[1], 'angular')  # Length of item vector that will be indexed
    for i, row in enumerate(sequences):
        annoy.add_item(i, row)
    annoy.build(NUM_TREES)  # 100 trees
    annoy.save(MODEL_ROOT + fname)


def sim_matrix(sequences, annoy_fname):
    """
    Construct similarity matrix through annoy
    """
    num_texts, max_length = sequences.shape
    sim_matrix = (-1) * np.ones((num_texts, num_texts))
    annoy = AnnoyIndex(max_length, 'angular')
    annoy.load(annoy_fname)

    for i in tqdm(range(num_texts)):
        neighbor_id, dist = annoy.get_nns_by_item(i, num_texts, include_distances=True)
        sim_row = (-1) * np.ones((num_texts,))
        for j in range(num_texts):
            sim_row[neighbor_id[j]] = dist[j]
        sim_matrix[i, :] = sim_row.T
    return sim_matrix


if __name__ == '__main__':
    input_sequences = loadData(MODEL_ROOT + '/input_sequences.pickle')
    target_sequences = loadData(MODEL_ROOT + '/target_sequences.pickle')

    fname1, fname2 = '/my_model_input_annoy.ann', '/my_model_target_annoy.ann'
    annoyID(input_sequences, fname1)
    annoyID(target_sequences, fname2)
    input_sim_matrix = sim_matrix(input_sequences, MODEL_ROOT + fname1)
    target_sim_matrix = sim_matrix(target_sequences, MODEL_ROOT + fname2)
    saveData(input_sim_matrix, MODEL_ROOT + '/input_sim_matrix.pickle')
    saveData(target_sim_matrix, MODEL_ROOT + '/target_sim_matrix.pickle')
    input_sim_df = pd.DataFrame(input_sim_matrix, columns=range(input_sim_matrix.shape[1]))
    target_sim_df = pd.DataFrame(target_sim_matrix, columns=range(target_sim_matrix.shape[1]))
    saveData(input_sim_df, MODEL_ROOT + '/input_sim_matrix.csv')
    saveData(target_sim_df, MODEL_ROOT + '/target_sim_matrix.csv')
    print(input_sim_matrix)
    print(target_sim_matrix)
