# %%
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.io import savemat
basedir = "/workspace/algorithm/lijia/liyu/DNAMeth/swavDNA/check_100kbself_rmxy_linear_lr0.1_nopretrianed_3/Fold0/cam/test_PredR/"
storedir = "/workspace/algorithm/lijia/liyu/DNAMeth/swavDNA/check_100kbself_rmxy_linear_lr0.1_nopretrianed_3/Fold0/cam/test_mask/"
def get_protypes():
    ct_l6_img = None
    for i in index1:
        img = cv2.imread(os.path.join(basedir,"CT-L6",file_ctl6[i]))
        if ct_l6_img is None:
            ct_l6_img = img[:,:,0]
        else:
            ct_l6_img = ct_l6_img + img[:,:,0]
    ct_l6_img = ct_l6_img/len(index1)
    return ct_l6_img
# for cate in os.listdir(basedir):
#     img = None
#     num = 0
#     for ifile in os.listdir(os.path.join(basedir,cate)):
#         camimg = cv2.imread(os.path.join(basedir,cate,ifile))
#         if img is None:
#             img = camimg[:,:,0]
#         else:
#             img = img + camimg[:,:,0]
#         num = num + 1
#     img = img/num
#     cv2.imwrite(os.path.join(storedir,cate+".png"),img)
#     gray = img/255
#     gray[gray>=0.7] = 255. 
#     gray[gray<0.7] = 0.
#     cv2.imwrite(os.path.join(storedir,cate+"_bin.png"),gray)
thres = 0.7
if not os.path.exists(storedir):
    os.makedirs(storedir)
storedir = os.path.join(storedir,"bin_thr0.7")
if not os.path.exists(storedir):
    os.makedirs(storedir)
# %%
if not os.path.exists(os.path.join(storedir,"Merge_thr")):
    os.makedirs(os.path.join(storedir,"Merge_thr"))
cams = None
for cate in os.listdir(basedir):
    img = None
    num = 0
    if not os.path.exists(os.path.join(storedir,cate)):
        os.makedirs(os.path.join(storedir,cate))
    for ifile in os.listdir(os.path.join(basedir,cate)):
        camimg = cv2.imread(os.path.join(basedir,cate,ifile))
        gray = camimg/255
        gray = (gray-np.min(gray))/(np.max(gray)-np.min(gray))
        # gray = gray/np.max(gray)
        gray[gray>=thres] = 1. 
        gray[gray<thres] = 0.
        # cv2.imwrite(os.path.join(storedir,cate,ifile),gray)
        if img is None:
            img = gray[:,:,0]
        else:
            img = img + gray[:,:,0]
        num = num + 1
    img = img/num
    img = (img-np.min(img))/(np.max(img)-np.min(img))
    img = cv2.applyColorMap(np.uint8(img*255), cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(storedir,"Merge_thr",cate+"_bin_thr0.7.png"),img)

# %% merge density
if not os.path.exists(os.path.join(storedir,"Merge_density")):
    os.makedirs(os.path.join(storedir,"Merge_density"))
for cate in os.listdir(basedir):
    img = None
    num = 0
    if not os.path.exists(os.path.join(storedir,cate)):
        os.makedirs(os.path.join(storedir,cate))
    for ifile in os.listdir(os.path.join(basedir,cate)):
        camimg = cv2.imread(os.path.join(basedir,cate,ifile))
        gray = camimg/255
        gray = (gray-np.min(gray))/(np.max(gray)-np.min(gray))
        if img is None:
            img = gray[:,:,0]
        else:
            img = img + gray[:,:,0]
        num = num + 1
    img = img/num
    img = (img-np.min(img))/(np.max(img)-np.min(img))
    img = np.uint8(img*255)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(cv2.cvtColor(img/255,cv2.COLOR_R,cmap='jet')
    fig.colorbar(cax)
    plt.show()
    
    # img = cv2.applyColorMap(np.uint8(img*255), cv2.COLORMAP_JET)
    # cv2.imwrite(os.path.join(storedir,"Merge_density",cate+"_bin_density_gray.png"),img)
    # img = img/np.max(img)
    # img[img>=0.9] = 1. 
    # img[img<0.9] = 0.
    # # img = img*255/np.max(img)
    # if cams is None:
    #     cams = img
    # else:
    #     cams = cams + img
# cams = np.clip(cams,0,1)
# cams = cams/len(os.listdir(basedir))
# cams = (cams-np.min(cams))/(np.max(cams)-np.min(cams))
# cams = cv2.applyColorMap(np.uint8(cams*255), cv2.COLORMAP_JET)
# cv2.imwrite(os.path.join(storedir,"Sum_bin_norm_density.png"),cams)
# %%    

# %%
# cell numbers influence
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.io import savemat
nums = np.concatenate(([5,10,20,30,40],np.array(range(50,304,20))))
basedir = "/workspace/algorithm/lijia/liyu/DNAMeth/swavDNA/check_100kbself_rmxy_linear_lr0.1_nopretrianed_3/Fold0/cam/test_PredR/"
storedir = "/workspace/algorithm/lijia/liyu/DNAMeth/swavDNA/check_100kbself_rmxy_linear_lr0.1_nopretrianed_3/Fold0/cam/test_mask/"

thres = 0.7
if not os.path.exists(storedir):
    os.makedirs(storedir)
storedir = os.path.join(storedir,"bin_thr0.7")
if not os.path.exists(storedir):
    os.makedirs(storedir)
if not os.path.exists(os.path.join(storedir,"Merge_thr_cellnumgrad2")):
    os.makedirs(os.path.join(storedir,"Merge_thr_cellnumgrad2"))
cams = None
for i in nums:
    for cate in os.listdir(basedir):
        if not cate == "CT-L6":
            continue
        img = None
        num = 0
        if not os.path.exists(os.path.join(storedir,cate)):
            os.makedirs(os.path.join(storedir,cate))
        for ifile in os.listdir(os.path.join(basedir,cate)):
            camimg = cv2.imread(os.path.join(basedir,cate,ifile))
            gray = camimg/255
            gray = (gray-np.min(gray))/(np.max(gray)-np.min(gray))
            # gray = gray/np.max(gray)
            gray[gray>=thres] = 1. 
            gray[gray<thres] = 0.
            # cv2.imwrite(os.path.join(storedir,cate,ifile),gray)
            if img is None:
                img = gray[:,:,0]
            else:
                img = img + gray[:,:,0]
            num = num + 1
            if num == i:
                break
        img = img/num
        img = (img-np.min(img))/(np.max(img)-np.min(img))
        img = cv2.applyColorMap(np.uint8(img*255), cv2.COLORMAP_JET)
        plt.figure()
        plt.imshow(img)
        plt.show()
        # cv2.imwrite(os.path.join(storedir,"Merge_thr_cellnumgrad2",cate+"_"+str(num)+"_bin_thr0.7.png"),img)

# %%
import pandas as pd
refchom = ["chr1","chr2","chr3","chr4","chr5","chr6","chr7","chr8","chr9","chr10","chr11","chr12","chr13","chr14","chr15","chr16","chr17","chr18","chr19"]
bedfile = pd.read_csv("/workspace/algorithm/lijia/liyu/DNAMeth/swavDNA/check_100kbself_rmxy_linear_lr0.1_nopretrianed_3/Fold0/cam/ref_100kb.bed",sep="\t",header=None)
chom = bedfile.iloc[:,0]
table = []
for i,ichom in enumerate(chom):
    if ichom in refchom:
        table.append(bedfile.iloc[i,:])
table = pd.DataFrame(data=table)
table.to_csv("/workspace/algorithm/lijia/liyu/DNAMeth/swavDNA/check_100kbself_rmxy_linear_lr0.1_nopretrianed_3/Fold0/cam/ref_100kb2.bed",header=None,sep="\t",index=False)    
# %% 
# transfer merge to 1d
import pandas as pd
from PIL import Image
import copy
import cv2
refchom = ["chr1","chr2","chr3","chr4","chr5","chr6","chr7","chr8","chr9","chr10","chr11","chr12","chr13","chr14","chr15","chr16","chr17","chr18","chr19"]
bedfile = pd.read_csv("/workspace/algorithm/lijia/liyu/DNAMeth/swavDNA/check_100kbself_rmxy_linear_lr0.1_nopretrianed_3/Fold0/cam/ref_100kb2.bed",sep="\t",header=None)
for i in os.listdir(os.path.join(storedir,"Merge_density")):
    if not i.endswith("gray.png"):
        continue
    img = cv2.imread(os.path.join(storedir,"Merge_density",i))[:,:,0]
    print(img.size)
    ref = copy.deepcopy(bedfile.iloc[:24639,:])
    a = (img.flatten()/255)[:24639]
    a[a>=0.7] = 1
    a[a<0.7] = 0
    ref.iloc[:,3] = a.astype(np.uint8)
    ref.to_csv(os.path.join(storedir,"Merge_density",i.split(".png")[0]+"_thr0.7.bed"),header=None,sep="\t",index=False)
# %%
# bedfile = pd.read_csv("/workspace/algorithm/lijia/liyu/DNAMeth/swavDNA/check_100kbself_rmxy_linear_lr0.1_nopretrianed_3/Fold0/cam/ref_100kb.bed",sep="\t",header=None)
bedfile = pd.read_csv("/workspace/algorithm/lijia/liyu/DNAMeth/swavDNA/check_100kbself_rmxy_linear_lr0.1_nopretrianed_3/Fold0/cam/test_mask/bin_thr0.7/Merge_density/L6b_bin_density.bed",sep="\t",header=None)
chom = bedfile.iloc[:,0]
a = 0
for i in chom:
    if i in refchom:
        a = a+1
print(a)

# %%
# cates = []
# filenames = []
# for i in os.listdir(basedir):
#     for j in os.listdir(os.path.join(basedir,i)):
#         icate = i
#         cates.append(icate)
#         filenames.append(j)
# num = len(set(list(cates)))
# celltypes = list(set(list(cates)))
# grays = []
# c = 0
# for i,ifile in enumerate(filenames):
#     for a,icate in enumerate(celltypes):
#         if cates[i] == icate:
#             img = cv2.imread(os.path.join(basedir,icate,ifile))
#             gray = img[:,:,0]/255
#             # gray[gray>=0.75] = 255. 
#             # gray[gray<0.75] = 0.
#             # if not os.path.exists(os.path.join(storedir,icate)):
#             #     os.mkdir(os.path.join(storedir,icate))
#             # cv2.imwrite(os.path.join(storedir,icate,ifile),np.array(gray))
#             plt.figure()
#             plt.hist(gray.ravel(),bins=100)
#             plt.savefig(os.path.join(storedir,str(a)+"_"+ifile))
#             if c==0:
#                 grays = gray.ravel()
#             else:
#                 grays = np.concatenate((grays,gray.ravel()),axis=0)
#             c+=1
# plt.figure()
# plt.hist(grays.ravel()[1:],bins=100)
# plt.savefig(os.path.join(storedir,"sum.png"))

#%%
file_ctl6 = []
file_itl23 = []
# file_itl4 = []
# file_itl5 = []
file_itl6 = []
nums = []
for i in os.listdir(basedir):
    if i == "CT-L6" or i == "IT-L4" or i == "IT-L5":
        for j in os.listdir(os.path.join(basedir,i)):
            if i == "CT-L6":
                file_ctl6.append(j)
            elif i == "IT-L4":
                file_itl6.append(j)
            elif i == "IT-L5":
                file_itl23.append(j)
ctl6 = []
itl23 = []
itl6 = []
group1 = []
group2 = []
group3 = []
def getimgs(files,icate,index):
    ctl6s = []
    psnr_per = []
    for i in index:
        img = cv2.imread(os.path.join(basedir,icate,files[i]))
        ctl6s.append(img[:,:,0])
    for i in range(len(index)-1):
        for j in range(1,len(index)):
            img1 = ctl6s[i]
            img2 = ctl6s[j]
            tmp = psnr(img1,img2,data_range=255)
            if np.isinf(tmp):
                psnr_per.append(0)
            else:
                psnr_per.append(tmp)
            # print(icate,i,np.mean(np.array(psnr_per)))
    return np.array(psnr_per)


def getimgs_group(files1,icate1,index1,files2,icate2,index2):
    ctl6s1 = []
    ctl6s2 = []
    psnr_group = []
    for i in index1:
        img = cv2.imread(os.path.join(basedir,icate1,files1[i]))
        ctl6s1.append(img[:,:,0])
    for i in index2:
        img = cv2.imread(os.path.join(basedir,icate2,files2[i]))
        ctl6s2.append(img[:,:,0])

    for i in range(len(index1)):
        for j in range(len(index2)):
            img1 = ctl6s1[i]
            img2 = ctl6s2[j]
            tmp = psnr(img1,img2,data_range=255)
            if np.isinf(tmp):
                psnr_group.append(0)
            else:
                psnr_group.append(tmp)
            # print(icate,i,np.mean(np.array(psnr_per)))
    return np.array(psnr_group)

# def get_protypes(index1,index2,index3):
#     ct_l6_img = None
#     it_l23_img = None
#     it_l6_img = None
#     for i in index1:
#         img = cv2.imread(os.path.join(basedir,"CT-L6",file_ctl6[i]))
#         if ct_l6_img is None:
#             ct_l6_img = img[:,:,0]
#         else:
#             ct_l6_img = ct_l6_img + img[:,:,0]
#     ct_l6_img = ct_l6_img/len(index1)
#     for i in index3:
#         img = cv2.imread(os.path.join(basedir,"IT-L23",file_itl23[i]))
#         if it_l23_img is None:
#             it_l23_img = img[:,:,0]
#         else:
#             it_l23_img = it_l23_img + img[:,:,0]
#     it_l23_img = it_l23_img/len(index3)
#     for i in index2:
#         img = cv2.imread(os.path.join(basedir,"IT-L6",file_itl6[i]))
#         if  it_l6_img is None:
#             it_l6_img = img[:,:,0]
#         else:
#             it_l6_img = it_l6_img + img[:,:,0]
#     it_l6_img = it_l6_img/len(index2)
#     return ct_l6_img,it_l23_img,it_l6_img


for i in range(300):
    print("================================= Time %d ================================="%i)
    index1 = np.random.choice(len(file_ctl6),100,replace=False)
    index2 = np.random.choice(len(file_itl6),100,replace=False)
    index3 = np.random.choice(len(file_itl23),100,replace=False)
    # ct_l6_img,it_l23_img,it_l6_img = get_protypes(index1,index2,index3)
    # a1 = psnr(ct_l6_img,it_l6_img,data_range=255)
    a1 = getimgs_group(file_ctl6,"CT-L6",index1,file_itl6,"IT-L4",index2)
    group1.append(np.mean(a1))
    a2 = getimgs_group(file_ctl6,"CT-L6",index1,file_itl23,"IT-L5",index3)
    group2.append(np.mean(a2))
    a3 = getimgs_group(file_itl6,"IT-L4",index2,file_itl23,"IT-L5",index3)
    group3.append(np.mean(a3))

    # if np.isinf(a1):
    #     group1.append(0)
    # else:
    #     group1.append(a1)
    # a2 = psnr(ct_l6_img,it_l23_img,data_range=255)
    # if np.isinf(a2):
    #     group2.append(0)
    # else:
    #     group2.append(a2)
    # a3 = psnr(it_l23_img,it_l6_img,data_range=255)
    # if np.isinf(a3):
    #     group3.append(0)
    # else:
    #     group3.append(a3)
    # print(i,np.mean(np.array(group1)),np.std(np.array(group1)),\
    #       np.mean(np.array(group2)),np.std(np.array(group2)),\
    #         np.mean(np.array(group3)),np.std(np.array(group3)))
    ctl6.append(getimgs(file_ctl6,"CT-L6",index1))
    itl6.append(getimgs(file_itl6,"IT-L4",index2))
    itl23.append(getimgs(file_itl23,"IT-L5",index3))
    print(i,np.mean(np.array(group1)),np.std(np.array(group1)),\
          np.mean(np.array(group2)),np.std(np.array(group2)),\
            np.mean(np.array(group3)),np.std(np.array(group3)))
    print(i,np.mean(np.array(ctl6)),np.std(np.array(ctl6)),\
          np.mean(np.array(itl6)),np.std(np.array(itl6)),\
            np.mean(np.array(itl23)),np.std(np.array(itl23)))
    if i%50 == 0:
        savemat("/workspace/algorithm/lijia/liyu/DNAMeth/swavDNA/check_100kbself_rmxy_linear_lr0.1_nopretrianed_3/Fold0/cam/group_predR.mat",\
                {"group1":group1,"group2":group2,"group3":group3,"group":{"group1":"CT-L6-IT-L4","group2":"CT-L6-IT-L5","group3":"IT-L4-IT-L5"}})
        savemat("/workspace/algorithm/lijia/liyu/DNAMeth/swavDNA/check_100kbself_rmxy_linear_lr0.1_nopretrianed_3/Fold0/cam/sample_predR.mat",\
                {"ctl6":ctl6,"itl4":itl6,"itl5":itl23})


# %%

            