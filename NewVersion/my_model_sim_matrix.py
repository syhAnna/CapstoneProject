import numpy as np
from tqdm import tqdm
from annoy import AnnoyIndex
from utils import MODEL_ROOT, loadData, saveData

NUM_TREES = 100
NUM_NN = 10
CONVERSATION_ID = 428  # query: 'do you enjoy movies'; ANS: 'yes i do do you'


def annoyID(sequences, fname):
    """
    Inplement annoy algorithm
    """
    annoy = AnnoyIndex(sequences.shape[1], 'angular')  # Length of item vector that will be indexed
    for i, row in enumerate(sequences):
        annoy.add_item(i, row)
    annoy.build(NUM_TREES)  # 100 trees
    annoy.save(MODEL_ROOT + fname)


def sim_matrix(texts, annoy_fname):
    """
    Construct similarity matrix through annoy
    """
    texts_len, max_input_len = len(texts), max([len(x) for x in texts])
    sim_matrix = (-1) * np.ones((texts_len, texts_len))
    annoy = AnnoyIndex(max_input_len, 'angular')
    annoy.load(annoy_fname)

    for i in tqdm(range(texts_len)):
        neighbor_id, dist = annoy.get_nns_by_item(i, texts_len, include_distances=True)
        sim_row = (-1) * np.ones((texts_len,))
        for j in range(texts_len):
            sim_row[neighbor_id[j]] = dist[j]
        sim_matrix[i, :] = sim_row.T
    return sim_matrix


if __name__ == '__main__':
    input_sequences = loadData(MODEL_ROOT + '/input_sequences.pickle')
    target_sequences = loadData(MODEL_ROOT + '/target_sequences.pickle')
    input_texts = loadData(MODEL_ROOT + '/input_texts.pickle')
    target_texts = loadData(MODEL_ROOT + '/target_texts.pickle')

    fname1, fname2 = '/my_model_input_annoy.ann', '/my_model_target_annoy.ann'
    annoyID(input_sequences, fname1)
    annoyID(target_sequences, fname2)
    input_sim_matrix = sim_matrix(input_texts, MODEL_ROOT + fname1)
    target_sim_matrix = sim_matrix(target_texts, MODEL_ROOT + fname2)
    saveData(input_sim_matrix, MODEL_ROOT + '/input_sim_matrix.pickle')
    saveData(target_sim_matrix, MODEL_ROOT + '/target_sim_matrix.pickle')
    print(input_sim_matrix)
    print(target_sim_matrix)
