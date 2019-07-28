import numpy as np
from scipy.stats import kruskal
from scikit_posthocs import posthoc_conover
from save_tex_table import *


def load_all(folder: str):
    result = []
    for i in range(0, 10):
        filename = folder + "/fold" + str(i) + ".npy"
        fold = np.load(filename)
        result.append(fold)
    return result


def folds_avrg(all_results):
    return list(map(lambda x: np.mean(x, axis=0), all_results))
