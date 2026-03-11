import torch
import torch.nn as nn
import torch.nn.functional as F

#================================================
#3) MODEL (Single Definition Only)
#================================================
import torch.nn as nn

def make_up_block(in_c, out_c, k=5, use_bn=True):
    pad = k // 2
    layers = [
        nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
        nn.Conv1d(in_c, out_c, k, padding=pad),
    ]
    if use_bn:
        layers.append(nn.BatchNorm1d(out_c))
    layers.append(nn.GELU())
    return nn.Sequential(*layers)

class BetterClassifier(nn.Module):
    def __init__(self, cls_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(cls_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, num_classes)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.3)
        self.norm = nn.LayerNorm(cls_dim)

    def forward(self, z):
        z = self.norm(z)
        h = self.act(self.fc1(z))
        h = h + self.dropout(self.act(self.fc2(h)))   # residual block
        return self.out(h)

class ChannelAttention(nn.Module):
    def __init__(self, c, reduction=16):
        super().__init__()
        hidden = max(1, c // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(c, hidden, 1),
            nn.GELU(),
            nn.Conv1d(hidden, c, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.net(x)

class ResBlock1D(nn.Module):
    def __init__(self, in_c, out_c, k=5, stride=1, use_bn=True, use_attn=True):
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv1d(in_c, out_c, k, stride=stride, padding=pad)
        self.bn1   = nn.BatchNorm1d(out_c) if use_bn else nn.Identity()
        self.conv2 = nn.Conv1d(out_c, out_c, k, padding=pad)
        self.bn2   = nn.BatchNorm1d(out_c) if use_bn else nn.Identity()
        self.skip  = nn.Conv1d(in_c, out_c, 1, stride=stride) \
            if (in_c != out_c or stride != 1) else nn.Identity()
        self.act   = nn.GELU()
        self.attn  = ChannelAttention(out_c) if use_attn else nn.Identity()

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y = self.attn(y)
        return self.act(y + self.skip(x))

    

class DeepConvAutoencoderClassifier(nn.Module):
    def __init__(
        self,
        input_length=2048,
        latent_dim=64,      
        cls_dim=64,         
        num_classes=7,
        use_bn=True,
        use_attn=True,
        use_projection_head=True,
    ):
        super().__init__()

        # ----------------------------
        # Shared encoder
        # ----------------------------
        self.encoder = nn.Sequential(
            ResBlock1D(1,   32, k=15, stride=2, use_bn=use_bn, use_attn=use_attn),
            ResBlock1D(32,  64, k=11, stride=2, use_bn=use_bn, use_attn=use_attn),
            ResBlock1D(64, 128, k=9,  stride=2, use_bn=use_bn, use_attn=use_attn),
        )

        # compute encoder output size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
            enc_out = self.encoder(dummy)
            self.enc_C = enc_out.size(1)
            self.enc_L = enc_out.size(2)
            self.encoder_output_dim = enc_out.numel()

        # ----------------------------
        # Separate latents
        # ----------------------------

        # Latent for reconstruction
        self.fc_latent_rec = nn.Linear(self.encoder_output_dim, latent_dim)
        self.latent_norm_rec = nn.LayerNorm(latent_dim)

        # Latent for classification / contrastive
        self.fc_latent_cls = nn.Linear(self.encoder_output_dim, cls_dim)
        self.latent_norm_cls = nn.LayerNorm(cls_dim)

        # ----------------------------
        # Projection head (contrastive on z_cls only)
        # ----------------------------
        self.projection_head = None
        if use_projection_head:
            self.projection_head = nn.Sequential(
                nn.Linear(cls_dim, 128),
                nn.GELU(),
                nn.Linear(128, 64)
            )

        # ----------------------------
        # Decoder (only uses z_rec)
        # ----------------------------
        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim, self.encoder_output_dim),
            nn.GELU()
        )
        self.decoder = nn.Sequential(
            make_up_block(self.enc_C, 64,  k=9,  use_bn=use_bn),
            make_up_block(64,        32,  k=11, use_bn=use_bn),
            make_up_block(32,        16,  k=15, use_bn=use_bn),
            nn.Conv1d(16, 1, kernel_size=7, padding=3),
        )

        # ----------------------------
        # Classifier (only uses z_cls)
        # ----------------------------
        self.classifier = BetterClassifier(cls_dim, num_classes)


    # ----------------------------
    # Helpers
    # ----------------------------

    def encode(self, x):
        """Return both reconstruction and classification latents."""
        if x.ndim == 2:
            x = x.unsqueeze(1)
        h = self.encoder(x).flatten(1)
        z_rec = self.latent_norm_rec(self.fc_latent_rec(h))
        z_cls = self.latent_norm_cls(self.fc_latent_cls(h))
        return z_rec, z_cls

    def project(self, z_cls):
        if self.projection_head is None:
            return z_cls
        return self.projection_head(z_cls)

    def decode(self, z_rec):
        B = z_rec.size(0)
        h = self.fc_decoder(z_rec)
        h = h.view(B, self.enc_C, self.enc_L)
        return self.decoder(h).squeeze(1)

    def forward(self, x, return_latent=False, return_proj=False):
        z_rec, z_cls = self.encode(x)
        x_hat = self.decode(z_rec)
        logits = self.classifier(z_cls)

        z_proj = self.project(z_cls) if return_proj and self.projection_head is not None else None

        if return_latent or return_proj:
            # we return z_cls as the "latent" used for classification/contrastive
            return x_hat, z_cls, logits, z_proj

        return x_hat, None, logits, None

    @staticmethod
    def gradient_sparsity_loss(x_hat):
        grad = x_hat[:, 1:] - x_hat[:, :-1]
        return grad.abs().mean()