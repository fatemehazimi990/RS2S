import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from copy import deepcopy


class Initializer(nn.Module):
    def __init__(self):
        super(Initializer, self).__init__()

        trained_model = models.vgg16(pretrained=True)
        vgg_conv_weights = [layer for layer in trained_model.features if isinstance(layer, nn.Conv2d)]
        self.vgg_backbone = VGGUNet()
        self.vgg_backbone.initialize_vgg(vgg_conv_weights)

        conv1_weights = trained_model.features[0].weight.data.clone()
        conv1_bias = trained_model.features[0].bias.data.clone()
        self.vgg_backbone.encoder[0][0] = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        nn.init.xavier_uniform_(self.vgg_backbone.encoder[0][0].weight)
        self.vgg_backbone.encoder[0][0].weight.data[:, 0:3, :, :].copy_(conv1_weights)
        self.vgg_backbone.encoder[0][0].bias.data.copy_(conv1_bias)

        fc6 = nn.Conv2d(512, 4096, 7, padding=7 // 2)
        fc6.weight.data.copy_(trained_model.classifier[0].weight.data.view(fc6.weight.size()))
        fc6.bias.data.copy_(trained_model.classifier[0].bias.data.view(fc6.bias.size()))

        fc7 = nn.Conv2d(4096, 4096, 1)
        fc7.weight.data.copy_(trained_model.classifier[3].weight.data.view(fc7.weight.size()))
        fc7.bias.data.copy_(trained_model.classifier[3].bias.data.view(fc7.bias.size()))
        self.mid_layer = nn.Sequential(fc6, nn.ReLU(True), fc7, nn.ReLU(True))

        self.h_1 = nn.Conv2d(4096, 512, 1)
        self.c_1 = nn.Conv2d(4096, 512, 1)

        self.reduce_conv = nn.Conv2d(4096, 512, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_conv = nn.Conv2d(512, 512, 3, padding=1)
        self.cat_conv = nn.Conv2d(512*2, 512, 1)

        self.h_2 = nn.Conv2d(512, 512, 1)
        self.c_2 = nn.Conv2d(512, 512, 1)
        # for next layer
        self.upsample_conv_2 = nn.Conv2d(512, 256, 3, padding=1)
        self.cat_conv_2 = nn.Conv2d(256*2, 256, 1)
        self.h_3 = nn.Conv2d(256, 256, 1)
        self.c_3 = nn.Conv2d(256, 256, 1)

        init_list = [self.h_1, self.c_1, self.reduce_conv, self.upsample_conv, self.cat_conv, self.h_2, self.c_2, 
                self.upsample_conv_2, self.cat_conv_2, self.h_3, self.c_3]

        for m in init_list:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.zero_()

    def forward(self, x):
        x, mid_features = self.vgg_backbone(x)
        x = self.mid_layer(x)

        h_1 = F.relu(self.h_1(x))
        c_1 = F.relu(self.c_1(x))

        x = F.relu(self.reduce_conv(x))
        x_up = self.upsample(x)
        x_up = F.relu(self.upsample_conv(x_up))
        x_up = F.relu(self.cat_conv(torch.cat([x_up, mid_features[-2]], dim=1)))
        h_2 = F.relu(self.h_2(x_up))
        c_2 = F.relu(self.c_2(x_up))

        x_upup = self.upsample(x_up)
        x_upup = F.relu(self.upsample_conv_2(x_upup))
        x_upup = F.relu(self.cat_conv_2(torch.cat([x_upup, mid_features[-3]], dim=1)))
        h_3 = F.relu(self.h_3(x_upup))
        c_3 = F.relu(self.c_3(x_upup))

        return [(h_1, c_1,), (h_2, c_2,), (h_3, c_3,)]


class ConvLSTMCell(nn.Module):
    """
    https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
    """
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias=True):

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        # g = torch.tanh(cc_g)
        g = torch.sigmoid(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * F.relu(c_next)
        # changed from tanh to relu
        return h_next, c_next


class VGGUNet(nn.Module):
    """
    https://github.com/kmaninis/OSVOS-PyTorch/blob/master/layers/osvos_layers.py
    """
    def __init__(self):
        super(VGGUNet, self).__init__()
        blocks = [[64, 64, 'M'],
                  [128, 128, 'M'],
                  [256, 256, 256, 'M'],
                  [512, 512, 512, 'M'],
                  [512, 512, 512, 'M']]
        in_channels = [3, 64, 128, 256, 512]

        encoder = nn.ModuleList()
        for i in range(len(blocks)):
            encoder.append(self.make_layers(blocks[i], in_channels[i]))

        self.encoder = encoder

    def forward(self, x):
        mid_feats = []
        for layer in self.encoder:
            x = layer(x)
            mid_feats.append(x)

        return x, mid_feats

    def initialize_vgg(self, vgg_conv_weights):
        num_conv = 0
        for e in self.encoder:
            for ee in e:
                if isinstance(ee, nn.Conv2d):
                    ee.weight = deepcopy(vgg_conv_weights[num_conv].weight)
                    ee.bias = deepcopy(vgg_conv_weights[num_conv].bias)
                    num_conv += 1

    @staticmethod
    def make_layers(cfg, in_channel):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channel, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channel = v
        return nn.Sequential(*layers)


class EncoderVGGUnet(nn.Module):
    def __init__(self):
        super(EncoderVGGUnet, self).__init__()

        trained_model = models.vgg16(pretrained=True).cuda()
        vgg_conv_weights = [layer for layer in trained_model.features if isinstance(layer, nn.Conv2d)]
        self.vgg_backbone = VGGUNet().cuda()
        self.vgg_backbone.initialize_vgg(vgg_conv_weights)

        fc_conv = nn.Conv2d(512, 4096 * 49, kernel_size=(1, 1))
        fc_conv.weight.data.copy_(trained_model.classifier[0].weight.data.view_as(fc_conv.weight.data).clone())

        out_conv = nn.Conv2d(4096 * 49, 512, 1)
        torch.nn.init.xavier_uniform_(out_conv.weight)

        self.out = nn.Sequential(fc_conv,
                                 nn.ReLU(True),
                                 out_conv,
                                 nn.ReLU(True))

    def forward(self, x):
        x, mid_feats = self.vgg_backbone(x)
        out = self.out(x)
        return out, mid_feats


class DecoderSkip(nn.Module):
    def __init__(self, num_classes):
        super(DecoderSkip, self).__init__()
        self.conv_1 = nn.Conv2d(1024, 512, 1)
        self.conv_11 = nn.Conv2d(512, 512, 5, padding=2)

        self.conv_2 = nn.Conv2d(3*512, 512, 1)
        self.conv_22 = nn.Conv2d(512, 256, 5, padding=2)

        self.conv_3_ = nn.Conv2d(512+256, 256, 1)
        self.conv_33 = nn.Conv2d(256, 128, 5, padding=2)

        self.conv_4 = nn.Conv2d(256, 128, 1)
        self.conv_44 = nn.Conv2d(128, 64, 5, padding=2)

        self.conv_5 = nn.Conv2d(128, 64, 1)
        self.conv_55 = nn.Conv2d(64, 64, 5, padding=2)

        self.distance_classifier_ = nn.Conv2d(64, num_classes, 3, padding=1)
        self.segmentation_branch_ = nn.Conv2d(64, num_classes, 3, padding=1)

        self.merge_ = nn.Conv2d(2 * num_classes, 1, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x, mid_features, h_middle, h_3):
        x = torch.cat((x, mid_features[4]), dim=1)
        x = F.relu(self.conv_1(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.conv_11(x))

        x = torch.cat((x, mid_features[3], h_middle), dim=1)
        x = F.relu(self.conv_2(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.conv_22(x))

        x = torch.cat((x, mid_features[2], h_3), dim=1)
        x = F.relu(self.conv_3_(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.conv_33(x))

        x = torch.cat((x, mid_features[1]), dim=1)
        x = F.relu(self.conv_4(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.conv_44(x))

        x = torch.cat((x, mid_features[0]), dim=1)
        x = F.relu(self.conv_5(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.conv_55(x))

        class_scores = self.distance_classifier_(x)
        seg_branch = self.segmentation_branch_(x)

        pred_mask = self.merge_(F.relu(torch.cat([class_scores, seg_branch], dim=1)))

        return pred_mask, class_scores


EPS = 1e-15
class SegmentationLossWithJaccardIndexLoss(nn.BCEWithLogitsLoss):

    def __init__(self, pos_weight, jacc_weight=0.3):
        super(SegmentationLossWithJaccardIndexLoss, self).__init__()
        self.jacc_weight = jacc_weight
        self.pos_weight = pos_weight
    def forward(self, output, target):
        bce = F.binary_cross_entropy_with_logits(output, target.float(), pos_weight=self.pos_weight)

        jaccard_target = (target == 1).float()
        jaccard_output = torch.sigmoid(output)
        intersection = (jaccard_output * jaccard_target).sum()
        union = jaccard_output.sum() + jaccard_target.sum()
        jacc = torch.log((intersection + EPS) / (union - intersection + EPS))
        return (1 - self.jacc_weight) * bce - jacc * self.jacc_weight

class JaccardIndexLoss(nn.BCEWithLogitsLoss):

    def __init__(self):
        super(JaccardIndexLoss, self).__init__()
    def forward(self, output, target):

        jaccard_target = (target == 1).float()
        jaccard_output = torch.sigmoid(output)
        intersection = (jaccard_output * jaccard_target).sum()
        union = jaccard_output.sum() + jaccard_target.sum()
        jacc = torch.log((intersection + EPS) / (union - intersection + EPS))
        return -jacc


def class_balanced_cross_entropy_loss(output, label, size_average=True, batch_average=True):
    """Define the class balanced cross entropy loss to train the network
    Args:
    output: Output of the network
    label: Ground truth label
    Returns:
    Tensor that evaluates the loss
    https://github.com/kmaninis/OSVOS-PyTorch/blob/master/layers/osvos_layers.py
    """

    labels = torch.ge(label, 0.5).float()

    num_labels_pos = torch.sum(labels)
    num_labels_neg = torch.sum(1.0 - labels)
    num_total = num_labels_pos + num_labels_neg

    output_gt_zero = torch.ge(output, 0).float()
    loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
        1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

    loss_pos = torch.sum(-torch.mul(labels, loss_val))
    loss_neg = torch.sum(-torch.mul(1.0 - labels, loss_val))

    final_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg

    if size_average:
        final_loss /= np.prod(label.size())
    elif batch_average:
        final_loss /= label.size()[0]

    return final_loss


