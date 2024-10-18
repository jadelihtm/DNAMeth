# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out1 = out + identity
        out = self.relu(out1)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x,domain="source"):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out1 = out+ identity
        out = self.relu(out1)

        return out


class ResNet(nn.Module):
    def __init__(
            self,
            block,
            layers,
            zero_init_residual=False,
            groups=1,
            widen=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=None,
            normalize=False,
            eval_mode=False,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.eval_mode = eval_mode
        self.padding = nn.ConstantPad2d(1, 0.0)

        self.inplanes = width_per_group * widen
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # change padding 3 -> 2 compared to original torchvision code because added a padding layer
        num_out_filters = width_per_group * widen
        self.conv1 = nn.Conv2d(
            3, num_out_filters, kernel_size=7, stride=2, padding=2, bias=False
        )
        self.bn1 = norm_layer(num_out_filters)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, num_out_filters, layers[0])
        num_out_filters *= 2
        self.layer2 = self._make_layer(
            block, num_out_filters, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        num_out_filters *= 2
        self.layer3 = self._make_layer(
            block, num_out_filters, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        num_out_filters *= 2
        self.layer4 = self._make_layer(
            block, num_out_filters, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # normalize output features
        self.l2norm = normalize

        # projection head
        # if output_dim == 0:
        #     self.projection_head = None
        #     self.projection_sechead = None
        # elif hidden_mlp == 0:
        #     self.projection_head = nn.Linear(num_out_filters * block.expansion, output_dim)
        #     if self.cls:
        #         self.projection_sechead = nn.Linear(num_out_filters * block.expansion, output_dim2)
        # else:
        #     self.projection_head = nn.Sequential(
        #         nn.Linear(num_out_filters * block.expansion, hidden_mlp),
        #         nn.BatchNorm1d(hidden_mlp),
        #         nn.ReLU(inplace=False),
        #         # nn.PReLU(),
        #         nn.Linear(hidden_mlp, output_dim),
        #     )
        #     if self.cls:
        #         self.projection_sechead = nn.Sequential(
        #             nn.Linear(num_out_filters * block.expansion, hidden_mlp),
        #             nn.BatchNorm1d(hidden_mlp),
        #             nn.ReLU(inplace=False),
        #             # nn.PReLU(),
        #             nn.Linear(hidden_mlp, output_dim2),
        #         )

        # # prototype layer
        # self.prototypes = None
        # self.prototypes_sec = None
        # if isinstance(nmb_prototypes, list):
        #     self.prototypes = MultiPrototypes(output_dim, nmb_prototypes)
        #     # self.prototypes_sec = MultiPrototypes(output_dim, nmb_prototypes,name="sec")
        # elif nmb_prototypes > 0:
        #     self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)
        #     # self.prototypes_sec = nn.Linear(output_dim, nmb_prototypes, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # self.domain_head = DomainDiscriminator()

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward_backbone(self, x):
        x = self.padding(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.eval_mode:
            return x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def forward_head(self, x):
        if self.projection_head is not None:
            x = self.projection_head(x)

        if self.l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)

        if self.prototypes is not None:
            return x, self.prototypes(x)
        return x
    
    def forward_sechead(self, x):
        domain,a = self.domain_head(x)
        if self.projection_sechead is not None:
            x = self.projection_sechead(x)

        if self.l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)

        if self.prototypes_sec is not None:
            return x, self.prototypes_sec(x)
        return x,domain

    # def forward(self, inputs1,inputs2=None):
    #     if not isinstance(inputs1, list):
    #         inputs1 = [inputs1]
    #         if inputs2 is not None:
    #             inputs2 = [inputs2]
    #     idx_crops1 = torch.cumsum(torch.unique_consecutive(
    #         torch.tensor([inp.shape[-1] for inp in inputs1]),
    #         return_counts=True,
    #     )[1], 0)
    #     if inputs2 is not None:
    #         idx_crops2 = torch.cumsum(torch.unique_consecutive(
    #             torch.tensor([inp.shape[-1] for inp in inputs2]),
    #             return_counts=True,
    #         )[1], 0)
    #     start_idx = 0
    #     for end_idx in idx_crops1:
    #         _out = self.forward_backbone(torch.cat(inputs1[start_idx: end_idx]).cuda(non_blocking=True))
    #         if start_idx == 0:
    #             output1 = _out
    #         else:
    #             output1 = torch.cat((output1, _out))
    #         start_idx = end_idx
    #     if inputs2 is not None:
    #         start_idx = 0
    #         for end_idx in idx_crops2:
    #             _out = self.forward_backbone(torch.cat(inputs2[start_idx: end_idx]).cuda(non_blocking=True))
    #             if start_idx == 0:
    #                 output2 = _out
    #             else:
    #                 output2 = torch.cat((output2, _out))
    #             start_idx = end_idx
    #     emb1 = output1
    #     output1,domain1,a1 = self.forward_head(output1)
    #     # output1 = self.forward_head(output1)
    #     if inputs2 is not None:
    #         emb2 = output2
    #         if self.cls:
    #             output2,domain2 = self.forward_sechead(output2)
    #         else:
    #             output2,domain2,a2 = self.forward_head(output2)
    #             # output2 = self.forward_head(output2)

    #     if inputs2 is not None:
    #         return emb1,output1,domain1,a1,emb2,output2,domain2,a2
    #     else:
    #         return emb1,output1,domain1,a1

    def forward(self, inputs1):
        if not isinstance(inputs1, list):
            inputs1 = [inputs1]
        idx_crops1 = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in inputs1]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        for end_idx in idx_crops1:
            _out = self.forward_backbone(torch.cat(inputs1[start_idx: end_idx]).cuda(non_blocking=True))
            if start_idx == 0:
                output1 = _out
            else:
                output1 = torch.cat((output1, _out))
            start_idx = end_idx
        emb1 = output1
        # output1 = self.forward_head(output1)
        return emb1
    
    # def forward(self, inputs1,en_fea=False):
    #     if not isinstance(inputs1, list):
    #         inputs1 = [inputs1]
    #     idx_crops1 = torch.cumsum(torch.unique_consecutive(
    #         torch.tensor([inp.shape[-1] for inp in inputs1]),
    #         return_counts=True,
    #     )[1], 0)
    #     start_idx = 0
    #     for end_idx in idx_crops1:
    #         _out = self.forward_backbone(torch.cat(inputs1[start_idx: end_idx]).cuda(non_blocking=True))
    #         if start_idx == 0:
    #             output1 = _out
    #         else:
    #             output1 = torch.cat((output1, _out))
    #         start_idx = end_idx
    #     if self.prototypes is not None:
    #         emb1,output1,feas = self.forward_head(output1)
    #         return emb1,output1,feas
    #     else:
    #         output1,emb = self.forward_head(output1)
    #         return emb, output1

    # def forward(self, inputs1):
    #     if not isinstance(inputs1, list):
    #         inputs1 = [inputs1]
    #     idx_crops1 = torch.cumsum(torch.unique_consecutive(
    #         torch.tensor([inp.shape[-1] for inp in inputs1]),
    #         return_counts=True,
    #     )[1], 0)
    #     start_idx = 0
    #     for end_idx in idx_crops1:
    #         _out = self.forward_backbone(torch.cat(inputs1[start_idx: end_idx]).cuda(non_blocking=True))
    #         if start_idx == 0:
    #             output1 = _out
    #         else:
    #             output1 = torch.cat((output1, _out))
    #         start_idx = end_idx
    #     if self.prototypes is not None:
    #         emb1,output1,feas = self.forward_head(output1)
    #         return emb1,output1,feas
    #     else:
    #         output1 = self.forward_head(output1)
    #         return  output1




class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, nmb_prototypes,name=None):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_prototypes)
        self.name = name
        for i, k in enumerate(nmb_prototypes):
            if self.name is None:
                self.add_module("prototypes" + str(i), nn.Linear(output_dim, k, bias=False))
            else:
                self.add_module("prototypes_"+self.name + str(i), nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            if self.name is None:
                out.append(getattr(self, "prototypes" + str(i))(x))
            else:
                out.append(getattr(self, "prototypes_"+self.name + str(i))(x))    
        return out

# source_domain_preds = domain_discriminator(grad_reverse(source_features, alpha=1.0))
# target_domain_preds = domain_discriminator(grad_reverse(target_features, alpha=1.0))
class classifier(nn.Module):
    def __init__(self,output_dim=0,hidden_mlp=0,normalize=False):
        super(classifier,self).__init__()
        # projection head
        if output_dim == 0:
            self.projection_head = None
            self.projection_sechead = None
        elif hidden_mlp == 0:
            self.projection_head = nn.Linear(2048, output_dim)
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(2048, hidden_mlp),
                nn.BatchNorm1d(hidden_mlp),
                nn.ReLU(inplace=False),
                # nn.PReLU(),
                nn.Linear(hidden_mlp, output_dim),
            )
        self.l2norm = normalize
    def forward(self, x):
        if self.projection_head is not None:
            x = self.projection_head(x)

        if self.l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)

        return x

class DomainDiscriminator(nn.Module):
    def __init__(self):
        super(DomainDiscriminator, self).__init__()
        dim =100
        self.fc1 = nn.Linear(2048, dim)
        self.bn1 = nn.BatchNorm1d(dim, momentum=0.01)
        self.relu = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(dim, 2)  # 2表示源域和目标域
        self.logit = nn.LogSoftmax(dim=1)
    def forward(self, x,domain="source"):
        # x=x
        x = self.fc1(x)
        # print(f"Source features version before: {x._version}")
        x = self.bn1(x)
        
        # print(f"Source features version before: {x._version}")
        out = x.detach()  # 保存一个副本以避免潜在的梯度问题
        x = self.relu(x)
        x = self.fc2(x)
        return self.logit(x),out

def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet50w2(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], widen=2, **kwargs)


def resnet50w4(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], widen=4, **kwargs)


def resnet50w5(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], widen=5, **kwargs)
