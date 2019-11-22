from models.bases.base import Base
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import torch

from .nl import NONLocalBlock1D
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(1,3,3), stride=(1, stride,stride),padding=(0,1,1), groups=1, bias=False, dilation=1)
        self.bn1 = nn.BatchNorm3d(planes, eps=0.001, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1,3,3), stride=(1,1,1),padding=(0,1,1), groups=1, bias=False, dilation=1)
        self.bn2 = nn.BatchNorm3d(planes, eps=0.001, momentum=0.01)
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

        out += identity
        out = self.relu(out)

        return out

class ResNet3D(nn.Module):
    def __init__(self, block, layers, num_classes=400):
        super(ResNet3D, self).__init__()
        self.global_pooling = True#Original Repo is False, revised by dongzhuoyao
        self.inplanes = 64
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1,2,2),
                               padding=(0, 3, 3), bias=False)
        #self.bn1 = nn.BatchNorm3d(64)
        self.bn1 = nn.BatchNorm3d(64, eps=0.001, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3),
                                    stride=(1,2,2), padding=(0, 1, 1))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.maxpool2 = nn.MaxPool3d(kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.dropout = nn.Dropout(.5)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        #self.cls_mem = torch.zeros([num_classes, 512 * block.expansion], requires_grad=False).cuda()
        self.cls_mem = nn.Parameter(torch.zeros(num_classes, 512 * block.expansion),requires_grad=False)
        self.cls_nl = NONLocalBlock1D(in_channels=512 * block.expansion, inter_channels=512 * block.expansion, sub_sample=False, bn_layer=True)
        self.rank_nl = NONLocalBlock1D(in_channels=512 * block.expansion, inter_channels=512 * block.expansion, sub_sample=False,
                                      bn_layer=True)
        self.nled_fc = nn.Linear(512 * block.expansion, num_classes)
        self.num_classes = num_classes

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=(1, stride, stride),
                          bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, mv=0.9):
        # x is of the form b x n x h x w x c
        # model expects b x c x n x h x w
        #import ipdb;ipdb.set_trace()  # BREAKPOINT
        x,y,temperature = x
        x = x.permute(0, 4, 1, 2, 3)#torch.Size([8, 3, 128, 112, 112])

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        rank_embed = x.view(x.size(0), x.size(1),x.size(2), -1).mean(dim=3)#B,C,T,torch.Size([12, 512, 128])
        cls_embed =  x.view(x.size(0), x.size(1), -1).mean(dim=2)#B,C

        normalized_cls_embed = F.normalize(cls_embed, p=2, dim=-1)#v3, normalize embed at the very beginning.
        #original feature nl interacts with normalized memory

        #embed#[B,512],[200, 512]->[B,200]
        batch_size = normalized_cls_embed.size(0)
        reg_logits = torch.ones([batch_size, self.num_classes]).cuda()
        for b in range(batch_size):
            tmp = -torch.norm(normalized_cls_embed[b] - self.cls_mem, p=2,dim=1)/temperature
            reg_logits[b] = tmp

        if self.training:
            with torch.no_grad():
                for ii, _y in enumerate(y):
                    old = self.cls_mem.data[_y]
                    tmp = mv*old + (1-mv)*normalized_cls_embed[ii]
                    self.cls_mem.data[_y] = F.normalize(tmp, p=2, dim=0)#https://discuss.pytorch.org/t/leaf-variable-was-used-in-an-inplace-operation/308/4

        def wrap_rankembed(_rank_embed):
            c_num = _rank_embed.size(1)
            t_num = _rank_embed.size(2)
            forinput = _rank_embed.permute(0,2,1).reshape(-1, c_num)#->[B,T,C]
            nled_embed = self.rank_nl(x_support=forinput, query=self.cls_mem)
            nled_embed = nled_embed.reshape(-1, t_num, c_num).permute(0, 2, 1)  # [B,C,T]
            return nled_embed


        logits = self.fc(self.dropout(cls_embed))  # torch.Size([8, 200, 15, 1, 1])
        #nled_logits = self.nled_fc(self.cls_nl(x_support=cls_embed, query=self.cls_mem))
        #nled_embed = wrap_rankembed(_rank_embed=rank_embed)

        nled_logits = self.nled_fc(cls_embed)
        nled_embed = rank_embed

        return logits, rank_embed, nled_logits, nled_embed, reg_logits, self.cls_mem

    def load_2d(self, model2d):
        print('inflating 2d resnet parameters')
        sd = self.state_dict()
        sd2d = model2d.state_dict()
        sd = OrderedDict([(x.replace('module.', ''), y) for x, y in sd.items()])
        sd2d = OrderedDict([(x.replace('module.', ''), y) for x, y in sd2d.items()])

        #for ii,_ in sd2d.items():
        #    print("name:{}, 2d: {}, 3d: {}".format(ii, sd2d[ii].shape,sd[ii].shape))

        for k, v in sd2d.items():
            if k not in sd:
                print('ignoring state key for loading: {}'.format(k))
                continue
            if 'conv' in k or 'downsample.0' in k:
                s = sd[k].shape#torch.Size([64, 3, 5, 7, 7])
                t = s[2]
                sd[k].copy_(v.unsqueeze(2).expand(*s) / t)#v:torch.Size([64, 3, 7, 7])
            elif 'bn' in k or 'downsample.1' in k:
                sd[k].copy_(v)
            else:
                print('skipping: {}'.format(k))

    def replace_logits(self, num_classes):
        pass
        #self.fc = nn.Conv3d(self.fc.in_channels, num_classes, kernel_size=1)


class ResNet183DF2FMemFixv3RegTempVisBank(Base):
    @classmethod
    def get(cls, args):
        model = ResNet3D(BasicBlock, [2, 2, 2, 2],num_classes=args.nclass)  # 50
        if args.pretrained:
            print("loading pretrained weight.!")
            from torchvision.models.resnet import resnet18
            model2d = resnet18(pretrained=True)
            model.load_2d(model2d)
        return model


if __name__ == "__main__":
    import torch
    batch_size = 8
    num_frames = 32
    img_feature_dim = 224
    input_var = torch.randn(batch_size, num_frames, img_feature_dim, img_feature_dim, 3).cuda()
    model = ResNet503D.get(None)
    model = model.cuda()
    output = model(input_var)
    print(output.shape)
