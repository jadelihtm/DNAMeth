from torchvision import models
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch
import numpy as np
from sklearn import manifold, datasets
import matplotlib.pyplot as plt
from scipy.io import savemat,loadmat
import pandas as pd
import seaborn as sns
import umap
import matplotlib.ticker as mticker
img_to_tensor = transforms.ToTensor()
from sklearn.metrics import confusion_matrix
labels=["CT-L6","IT-L23","IT-L4","IT-L5","IT-L6","MGE-Sst","MGE-Pvalb","PT-L5",\
        "L6b","CGE-Lamp5","ODC","ASC","CGE-Vip","MGC","OPC","NP-L6","VLMC-Pia",\
            "PC","VLMC","CLA","PAL-Inh","Unc5c","MSN-D2","EC","AT1","AT2","Basal",\
                "Blood cell","Ciliated","Endothelium","FABP4+ cell","Fibroblast",\
                    "Macrophage","Mucous_secretory","SMC","Unknown"]
# labels=["CT-L6","IT-L23","IT-L4","IT-L5","IT-L6","MGE-Sst","MGE-Pvalb","PT-L5",\
#         "L6b","CGE-Lamp5","ODC","ASC","CGE-Vip","MGC","OPC","NP-L6","VLMC-Pia",\
#             "PC","VLMC","CLA","Unkown"]
# labels=["Aerocytes","AT1","AT2","Basal","Ciliated","Fibroblast","Immune cell","Mucous_secretory",\
#         "SMC","Vascular endo","venous endo","Unknown"]
# region=["MOp","MOs","PIR","ACB","PFC"]
region=["Isocortex","Striatum","Olfactory"]
def display_label(interlabel,X_norm,plot_suffix,path,cmap=['red','blue','green','orange','magenta'],cate="all",alpha=[0.5,0.5,0.5,0.5,0.5]):
    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []
    type3_x = []
    type3_y = []
    type4_x = []
    type4_y = []
    type5_x = []
    type5_y = []
    for i in range(len(interlabel)):
        if interlabel[i] == 0:
            type1_x.append(X_norm[i,0])
            type1_y.append(X_norm[i,1])
        if interlabel[i] == 1:
            type2_x.append(X_norm[i,0])
            type2_y.append(X_norm[i,1])
        if interlabel[i] == 2:
            type3_x.append(X_norm[i,0])
            type3_y.append(X_norm[i,1])
        if interlabel[i] == 3:
            type4_x.append(X_norm[i,0])
            type4_y.append(X_norm[i,1])
        if interlabel[i] == 4:
            type5_x.append(X_norm[i,0])
            type5_y.append(X_norm[i,1])

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    size = 10
    type1 = ax.scatter(type1_x, type1_y, s=size, c = cmap[0],alpha=alpha[0])
    type2 = ax.scatter(type2_x, type2_y, s=size, c = cmap[1],alpha=alpha[1])
    type3 = ax.scatter(type3_x, type3_y, s=size, c = cmap[2],alpha=alpha[2])
    type4 = ax.scatter(type4_x, type4_y, s=size, c = cmap[3],alpha=alpha[3])
    type5 = ax.scatter(type5_x, type5_y, s=size, c = cmap[4],alpha=alpha[4])
    plt.title('t-SNE')
    ax.legend((type1, type2,type3,type4,type5), ('CT-L6','IT-L23','IT-L4','IT-L5','IT-L6'), ncol=2,bbox_to_anchor=(1,0),loc=3,borderaxespad=0)
    # ax.legend((type1, type2,type3,type4), ('2C','3C','4B','5D'), ncol=2,bbox_to_anchor=(1,0),loc=3,borderaxespad=0)
    plt.savefig(path+'/'+plot_suffix+cate+ '.png',bbox_inches='tight')
def interlabel2label(interlabel):
    interlabel_new = []
    for i in interlabel:
        interlabel_new.append([i,labels[i]])
    return interlabel_new

def dim_pca(features,interlabel,path,epoch,labelname=None):

    '''t-SNE'''
    perplexity = 30
    lr = 10
    niter = 2000
    filename = path+'/per_'+ str(perplexity) + 'lr' + str(lr) + 'it' + str(niter)+"_ep"+str(epoch) + '.mat'
    plot_suffix = 'per' + str(perplexity) + 'lr' + str(lr) + 'it' + str(niter)+"_ep"+str(epoch)
    '''
    UMAP(a=None, angular_rp_forest=False, b=None,
     force_approximation_algorithm=False, init='spectral', learning_rate=1.0,
     local_connectivity=1.0, low_memory=False, metric='euclidean',
     metric_kwds=None, min_dist=0.1, n_components=2, n_epochs=None,
     n_neighbors=15, negative_sample_rate=5, output_metric='euclidean',
     output_metric_kwds=None, random_state=42, repulsion_strength=1.0,
     set_op_mix_ratio=1.0, spread=1.0, target_metric='categorical',
     target_metric_kwds=None, target_n_neighbors=-1, target_weight=0.5,
     transform_queue_size=4.0, transform_seed=42, unique=False, verbose=False)
    '''
    tsne = umap.UMAP(n_neighbors=perplexity,n_components=2,min_dist=0.1,learning_rate=lr)
    # tsne = manifold.TSNE(n_components=2,perplexity=perplexity,n_iter=niter,learning_rate=lr)  ## 参数需要进行调整 reducer = umap.UMAP()
    X_tsne = tsne.fit_transform(features)
    print('t-SNE is already...')
    # class_num = len(np.unique(interlabel))

    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    savemat(filename,mdict={'X_norm':X_norm})
    plt.figure(figsize=(10, 10))
    fig, ax = plt.subplots()
    cmap1 = plt.get_cmap("tab20b")
    cmap2 = plt.get_cmap("tab20c")
    from matplotlib.colors import ListedColormap
    new_cmap = ListedColormap(cmap1.colors+ cmap2.colors)
    uni_label = np.unique(interlabel)
    scatter = plt.scatter(X_norm[:, 0], X_norm[:, 1], c=interlabel-np.min(uni_label), cmap=new_cmap, s=2)
    import matplotlib.ticker as mticker
    if labelname is not None:
        fig.colorbar(scatter,boundaries=np.arange(len(uni_label)+1)-0.5,ticks=np.array(range(-np.min(uni_label)+np.max(uni_label)+1)),format=mticker.FixedFormatter(np.array(labelname)[uni_label]))
    
    cate = "celltype"
    plt.savefig(path+'/'+plot_suffix+cate+ '.png',bbox_inches='tight')

    # plt.figure(figsize=(10, 10))
    # fig, ax = plt.subplots()
    # scatter = plt.scatter(X_norm[:, 0], X_norm[:, 1], c=regionlabel, cmap=new_cmap, s=2)
    # fig.colorbar(scatter,boundaries=np.arange(len(region)+1)-0.5,ticks=np.array(range(len(region))),format=mticker.FixedFormatter(region))
    # cate = "region"
    # plt.savefig(path+'/'+plot_suffix+cate+ '.png',bbox_inches='tight')
    
    # plt.figure(figsize=(10, 10))
    # idx = np.where(interlabel==35)[0]
    # fig, ax = plt.subplots()
    # scatter = plt.scatter(X_norm[idx, 0], X_norm[idx, 1], c=interlabel[idx], cmap=new_cmap, s=2)
    # cate = "UnknownCelltype"
    # plt.savefig(path+'/'+plot_suffix+cate+ '.png',bbox_inches='tight')



if __name__ == '__main__':
    data = loadmat("/workspace/algorithm/lijia/liyu/DNAMeth/swavDNA/check_100kbself_rmxy_linear_lr0.1_nopretrianed_3/Fold0/features/test/EP30_fea10kb2048-resnet50.mat")
    features = data["fea"]
    label = data["label"]
    path = "/workspace/algorithm/lijia/liyu/DNAMeth/swavDNA/check_100kbself_rmxy_linear_lr0.1_nopretrianed_3/Fold0/umap/test/"
    dim_pca(features, label,path,epoch=30)