"""
Generate Response Sets: DBSCAN
Use the similarity matrix to cluster together all similar sentences in the dataset
and assign labels to each of these clusters.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from collections import Counter
from utils import remove_indicator, loadData, saveData
from utils import MODEL_ROOT

N_NEIGHBOR = 10


def find_eps_nums(sim_matrix):
    """
    Try 50 epsilon number 0~0.03 evenly
    """
    eps_nums = list()
    eps_num_clusters, eps_num_noise = list(), list()
    trial_eps = np.linspace(0, 0.03, num=50)

    for eps_num in trial_eps:
        if eps_num > 0:
            print("Taking EPS as", eps_num)
            # Compute DBSCAN
            db = DBSCAN(eps=eps_num, min_samples=2, metric="precomputed", n_jobs=4).fit(sim_matrix)
            labels = db.labels_
            # get #clusters in labels, ignoring noise
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            # append the values
            eps_nums.append(eps_num)
            eps_num_clusters.append(n_clusters)
            eps_num_noise.append(n_noise)
            print('Number of clusters: %d' % n_clusters)
            print('Number of noise points: %d' % n_noise)
            print("----------------------")
            if n_clusters == 1: # n_clusters > 1
                break

    # plot the image for visualization
    fig, ax1 = plt.subplots()
    color1, color2 = 'r', 'b'
    ax1.set_xlabel('eps')

    ax1.set_ylabel('number of clusters', color=color1)
    ax1.plot(eps_nums, eps_num_clusters, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.set_ylabel('number of noises', color=color2)
    ax2.plot(eps_nums, eps_num_noise, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.show()


def dbscan(eps_num, sim_matrix, texts):
    print("Taking EPS as", eps_num)
    # Compute DBSCAN
    db = DBSCAN(eps=eps_num, min_samples=2, metric="precomputed", n_jobs=4).fit(sim_matrix)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print('Number of clusters: %d' % n_clusters)
    print('Estimated number of noise points: %d' % n_noise)
    for unique_label in set(labels):
        class_member_mask = (labels == unique_label)
        vec_func = np.vectorize(remove_indicator)
        print("In cluster", unique_label, "found", Counter(class_member_mask)[True], "points")
        print("Samples")
        print(vec_func(np.array(texts)[class_member_mask]))
        print("-------------------------------------")
    return db, n_clusters


if __name__ == '__main__':
    # load similarity matrix
    target_sim_matrix = loadData(MODEL_ROOT + '/target_sim_matrix.pickle')
    target_sim_df = loadData(MODEL_ROOT + '/target_sim_matrix.csv')
    target_texts = loadData(MODEL_ROOT + '/target_texts.pickle')
    find_eps_nums(target_sim_df)

    # case1: eps_num = 0.005
    eps_num = 0.005
    db1, n_clusters1 = dbscan(eps_num, target_sim_matrix, target_texts)
    saveData(db1, MODEL_ROOT + '/target_dbscan_0005.pickle')

    # case2: eps_num = 0.008
    eps_num = 0.008
    db2, n_clusters2 = dbscan(eps_num, target_sim_matrix, target_texts)
    saveData(db2, MODEL_ROOT + '/target_dbscan_0008.pickle')

    # case3: eps_num = 0.01
    eps_num = 0.01
    db3, n_clusters3 = dbscan(eps_num, target_sim_matrix, target_texts)
    saveData(db3, MODEL_ROOT + '/target_dbscan_001.pickle')

    # case4: eps_num = 0.015
    eps_num = 0.015
    db4, n_clusters4 = dbscan(eps_num, target_sim_matrix, target_texts)
    saveData(db4, MODEL_ROOT + '/target_dbscan_0015.pickle')

    # case5: eps_num = 0.05
    eps_num = 0.05
    db5, n_clusters5 = dbscan(eps_num, target_sim_matrix, target_texts)
    saveData(db5, MODEL_ROOT + '/target_dbscan_005.pickle')
