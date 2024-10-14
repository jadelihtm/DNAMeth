
# %%
import numpy as np
from sklearn import manifold, datasets
# import matplotlib.pyplot as plt
# import h5py
import os
from scipy.io import savemat,loadmat
# import pandas as pd
# import seaborn as sns
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from pathlib import Path

def ARI(labels_true, labels_pred, beta=1.):
    ari = adjusted_rand_score(labels_true, labels_pred)
    return ari

def NMI_sklearn(predict, label):
    # return metrics.adjusted_mutual_info_score(predict, label)
    return metrics.normalized_mutual_info_score(predict, label)

def SIL_sklearn(x,label):
    # x means dataset; 
    a =metrics.silhouette_score(x,label)
    return a


def eval_cluster(label,pred_label,features):
    ari_ = ARI(np.array(label),pred_label)                                                                                          
    nmi_ = NMI_sklearn(pred_label,label)
    sil_ = SIL_sklearn(features,pred_label)
    return ari_,nmi_,sil_

def eval_cluster_(pred_label,features):
    sil_ = SIL_sklearn(features,pred_label)
    return sil_

if __name__ == "__main__":
    eval_cluster()
