import torch.nn as nn
from src.models import pvt_v2_b2_li
import torch


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)

        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )

        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), dim=1))
        x = self.relu(x_cat + self.conv_res(x))
        return x


class EXP1Model(nn.Module):
    def __init__(self, ):
        super(EXP1Model, self).__init__()

        self.rgb_encoder = pvt_v2_b2_li(pretrained=True)
        self.depth_encoder = pvt_v2_b2_li(pretrained=True)
        self.rgb_rfb = RFB_modified(512, 512)
        self.depth_rfb = RFB_modified(512, 512)
        self.decoder_stage1 = nn.Sequential(
            nn.Conv2d(512, 320, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(320),
            nn.ReLU(True),
            nn.Conv2d(320, 320, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(320),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True), )
        self.decoder_stage2 = nn.Sequential(
            nn.Conv2d(320, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True), )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True), )
        self.dropout = nn.Dropout2d(0.5)
        self.seg_head = nn.Conv2d(64, 1, kernel_size=1)

    def forward_train(self, img, depth):
        rgb_feats = self.rgb_encoder(img)
        depth_feats = self.depth_encoder(depth)
        rgb_feat = self.rgb_rfb(rgb_feats[-1])
        depth_feat = self.depth_rfb(depth_feats[-1])
        x = self.decoder_stage1(rgb_feat + depth_feat + rgb_feats[-1])
        x = self.decoder_stage2(x+rgb_feats[-2])
        x = self.decoder_stage3(x+rgb_feats[-3])
        x = self.decoder_stage4(x+rgb_feats[-4])
        # x = self.dropout(x)
        out = self.seg_head(x)
        return out

    def forward(self, img, depth):
        return self.forward_train(img, depth)


if __name__ == '__main__':
    x = torch.rand(1, 3, 352, 352)
    model = EXP1Model()
    out = model(x, x)
    print(out.shape)
    pass

