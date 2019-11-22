from models.bases.base import Base
import torch.nn as nn
from collections import OrderedDict


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

        class TemporalModule(nn.Module):
            def __init__(self,_dim=512 * block.expansion):
                """
                In the constructor we instantiate two nn.Linear modules and assign them as
                member variables.
                """
                super(TemporalModule, self).__init__()
                self.temporal_conv1 = nn.Conv1d(_dim, _dim, kernel_size=3, stride=1,
                                                padding=1, bias=False)
                self.temporal_relu1 = nn.ReLU(inplace=True)
                self.temporal_conv2 = nn.Conv1d(_dim, _dim, kernel_size=3, stride=1,
                                                dilation=2, padding=1, bias=False)
                self.temporal_relu2 = nn.ReLU(inplace=True)
                self.temporal_conv3 = nn.Conv1d(_dim, _dim, kernel_size=3, stride=1,
                                                dilation=4, padding=1, bias=False)
                self.temporal_relu3 = nn.ReLU(inplace=True)
                self.dropout = nn.Dropout(.5)
                self.fc = nn.Linear(_dim, num_classes, bias=False)

            def forward(self, rank_embed):
                _init_feat = rank_embed
                rank_embed = self.temporal_relu1(self.temporal_conv1(rank_embed))
                rank_embed = self.temporal_relu2(self.temporal_conv2(rank_embed))
                rank_embed = self.temporal_relu3(self.temporal_conv3(rank_embed))

                embed = rank_embed.view(rank_embed.size(0), rank_embed.size(1), -1).mean(dim=-1)  # B,C
                logits = self.fc(self.dropout(embed))  # torch.Size([8, 200, 15, 1, 1])
                return logits, _init_feat

        self.temporal = TemporalModule(_dim=512 * block.expansion)


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

    def forward(self, x):
        # x is of the form b x n x h x w x c
        # model expects b x c x n x h x w
        #import ipdb;ipdb.set_trace()  # BREAKPOINT
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
        logits, rank_embed = self.temporal(rank_embed)


        return logits, rank_embed

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


class ResNet183DF2FLate1DMT(Base):
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
