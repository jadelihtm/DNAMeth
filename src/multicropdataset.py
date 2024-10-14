# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
from logging import getLogger
from PIL import Image
from PIL import ImageFilter
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# labels={"CT-L6":0,"IT-L23":1,"IT-L4":2,"IT-L5":3,"IT-L6":4,"MGE-Sst":5,"MGE-Pvalb":6,"PT-L5":7,\
#         "L6b":8,"CGE-Lamp5":9,"ODC":10,"ASC":11,"CGE-Vip":12,"MGC":13,"OPC":14,"NP-L6":15,"VLMC-Pia":16,\
#             "PC":17,"VLMC":18,"CLA":19,"PAL-Inh":20,"Unc5c":21,"MSN-D2":22,"EC":23,"AT1":24,"AT2":25,\
#                 "Basal":26,"Blood cell":27,"Ciliated":28,"Endothelium":29,"FABP4+ cell":30,"Fibroblast":31,\
#         "Macrophage":32,"Mucous_secretory":33,"SMC":34,"Unknown":35}
# labels={"Aerocytes":0,"AT1":1,"AT2":2,"Basal":3,"Ciliated":4,"Fibroblast":5,"Immune cell":6,"Mucous_secretory":7,\
#         "SMC":8,"Vascular endo":9,"venous endo":10,"Unknown":11}
# labels={"CT-L6":0,"IT-L23":1,"IT-L5":2,"IT-L6":3,"MGE-Sst":4,"MGE-Pvalb":5,"PT-L5":6,\
#         "L6b":7,"CGE-Lamp5":8,"ODC":9,"ASC":10,"MGC":11,"NP-L6":12,\
#             "PC":13,"CLA":14}
# labels={"CLP":0,"CMP":1,"GMP":2,"HSC":3,"HSCfl":4,"HSCbm":5,"MEP":6,"MK":7,\
#         "MLP0":8,"MLP1":9,"MLP2":10,"MLP3":11,"MPP":12,"Mono":13,"MPPbm":14}
# silce={"2C":"MOp","3C":"MOp","4B":"MOp","5D":"MOp",'1A':"MOs",'3E':"PIR",'4F':"PIR",'2B':"MOs",'4E':"ACB",'3A':"PFC",'2D':'PIR'}
# region={"MOp":0,"MOs":1,"PIR":2,"ACB":3,"PFC":4,"Unknown":5}

mus2human={"ASC":"ASC","CT-L6":"L6-CT","EC":"EC","IT-L23":"L23-IT","IT-L4":"L4-IT","IT-L5":"L5-IT","IT-L6":"L6-IT","L6b":"L6b",\
            "MGC":"MGC","MSN-D2":"MSN-D2","NP-L6":"L56-NP","ODC":"ODC","OPC":"OPC","PC":"PC","VLMC":"VLMC"}
silce={"2C":"Isocortex","3C":"Isocortex","4B":"Isocortex","5D":"Isocortex",'1A':"Isocortex",'2B':"Isocortex",\
       '4E':"Striatum",'3A':"Isocortex",'2D':'Olfactory','3E':"Olfactory",'4F':"Olfactory"}
region={"Isocortex":0,"Striatum":1,"Olfactory":2,"Unknown":3}
logger = getLogger()

class MultiCropDataset(datasets.ImageFolder):
    def __init__(
        self,
        data_path,
        size_crops,
        nmb_crops,
        min_scale_crops,
        max_scale_crops,
        size_dataset=-1,
        return_index=False,
        fresize = True,
        dataset="ratio",
    ):
        super(MultiCropDataset, self).__init__(data_path)
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        if size_dataset >= 0:
            self.samples = self.samples[:size_dataset]
        self.return_index = return_index
        self.size_crops = size_crops
        self.dataset = dataset

        if dataset == "MethSCAn":
            mean = [0.7582,0.7582,0.7582]
            std= [0.2469,0.2469,0.2469]
        elif dataset =="ratio":
            mean = [0.7576,0.7576,0.7576]
            std= [0.2468,0.2468,0.2468]
        elif dataset == "Merge":
            mean = [[0.7576,0.7576,0.7576],[0.7582,0.7582,0.7582]]
            std= [[0.2468,0.2468,0.2468],[0.2469,0.2469,0.2469]]

        trans = []
        for i in range(len(size_crops)):
            # randomresizedcrop = transforms.RandomResizedCrop(
            #     size_crops[i],
            #     scale=(min_scale_crops[i], max_scale_crops[i]),
            # )
            if fresize:
                randomresizedcrop = transforms.Resize(size_crops[i])
            else:
                randomresizedcrop = transforms.RandomCrop(size_crops[i])
            # 
            trans.extend([transforms.Compose([
                randomresizedcrop,
                transforms.RandomHorizontalFlip(p=0.5),
                # transforms.Compose(color_transform),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
            ] * nmb_crops[i])
        self.nmb_crops = nmb_crops
        self.trans = trans

    def __getitem__(self, index):
        path, _ = self.samples[index]
        labelname = path.split('/')[-2]
        relabel = labels[labelname]
        if relabel ==24:
            relabel = random.randint(24,34)
        if self.dataset == "Merge":
            img = np.array(self.loader(path))[:,:,(0,1)]
            img1 = np.repeat(img[:,:,0].reshape(img.size[0],img.size[0],-1),3,axis=2)
            img2 = np.repeat(img[:,:,0].reshape(img.size[0],img.size[0],-1),3,axis=2)
            image1 = Image.fromarray((img1).astype(np.uint8))
            image2 = Image.fromarray((img2).astype(np.uint8))
            multi_crops1 = list(map(lambda trans: trans(image1), self.trans))
            multi_crops2 = list(map(lambda trans: trans(image2), self.trans))
            multi_crops = [multi_crops1,multi_crops2]
        else:
            img = np.array(self.loader(path))
            image = Image.fromarray((img).astype(np.uint8))
            tmp = map(lambda trans: trans(image), self.trans)
            multi_crops = list(tmp)
        if self.return_index:
            return index,relabel,multi_crops
        return multi_crops,relabel


class MultiCropDataset_linear(datasets.ImageFolder):
    def __init__(
        self,
        data_path,
        size_crops,
        size_dataset=-1,
        return_index=False,
        fresize = True,
        dataset="ratio",
        mean=None,
        std = None,
        labelnames=None,
        target_datasets=False,
    ):
        super(MultiCropDataset_linear, self).__init__(data_path)
        if size_dataset >= 0:
            self.samples = self.samples[:size_dataset]
        self.return_index = return_index
        self.dataset = dataset
        if mean is None:
            if dataset == "MethSCAn":
                mean = [0.7582,0.7582,0.7582]
                std= [0.2469,0.2469,0.2469]
            elif dataset =="ratio":
                mean = [0.7576,0.7576,0.7576]
                std= [0.2468,0.2468,0.2468]
                # mean = [0.7597,0.7597,0.7597]
                # std= [0.2690,0.2690,0.2690]
            elif dataset == "Merge":
                mean = [[0.7576,0.7576,0.7576],[0.7582,0.7582,0.7582]]
                std= [[0.2468,0.2468,0.2468],[0.2469,0.2469,0.2469]]
        else:
            mean = mean
            std = std
        trans = []
        if fresize:
            randomresizedcrop = transforms.Resize(size_crops)
        else:
            randomresizedcrop = transforms.RandomCrop(size_crops)
        # 
        trans.extend([transforms.Compose([
            randomresizedcrop,
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.Compose(color_transform),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        ])
        self.trans = trans
        self.labelsnames = labelnames
        self.target_datasets = target_datasets

    def __getitem__(self, index):
        path, _ = self.samples[index]
        labelname = path.split('/')[-2]
        if self.labelsnames is None:
            relabel = 50
        else:
            if self.target_datasets:
                relabel = self.labelsnames[mus2human[labelname]]
            else:
                relabel = self.labelsnames[labelname]
        if self.dataset == "Merge":
            img = np.array(self.loader(path))[:,:,(0,1)]
            img1 = np.repeat(img[:,:,0].reshape(img.size[0],img.size[0],-1),3,axis=2)
            img2 = np.repeat(img[:,:,0].reshape(img.size[0],img.size[0],-1),3,axis=2)
            image1 = Image.fromarray((img1).astype(np.uint8))
            image2 = Image.fromarray((img2).astype(np.uint8))
            multi_crops1 = list(map(lambda trans: trans(image1), self.trans))
            multi_crops2 = list(map(lambda trans: trans(image2), self.trans))
            multi_crops = [multi_crops1,multi_crops2]
        else:
            img = np.array(self.loader(path))
            image = Image.fromarray((img).astype(np.uint8))
            tmp = map(lambda trans: trans(image), self.trans)
            multi_crops = list(tmp)
        if self.return_index:
            return index,multi_crops,relabel
        return multi_crops,relabel
    

class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort
