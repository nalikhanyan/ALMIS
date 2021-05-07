import torch
from torch import hub
from torch.nn import functional as F


def forward(self, x):
    enc1 = self.encoder1(x)
    enc2 = self.encoder2(self.pool1(enc1))
    enc3 = self.encoder3(self.pool2(enc2))
    enc4 = self.encoder4(self.pool3(enc3))

    F.dropout(x, p=0.5, training=self.training)

    bottleneck = self.bottleneck(self.pool4(enc4))

    dec4 = self.upconv4(bottleneck)
    dec4 = torch.cat((dec4, enc4), dim=1)
    dec4 = self.decoder4(dec4)
    dec3 = self.upconv3(dec4)
    dec3 = torch.cat((dec3, enc3), dim=1)
    dec3 = self.decoder3(dec3)
    dec2 = self.upconv2(dec3)
    dec2 = torch.cat((dec2, enc2), dim=1)
    dec2 = self.decoder2(dec2)
    dec1 = self.upconv1(dec2)
    dec1 = torch.cat((dec1, enc1), dim=1)
    dec1 = self.decoder1(dec1)
    return torch.sigmoid(self.conv(dec1))


def get_drp_unet(in_channels, pretrained):
    model = hub.load('mateuszbuda/brain-segmentation-pytorch',
                    'unet',
                    in_channels=in_channels,
                    pretrained=pretrained)

    model.__class__.forward = forward
    return model
