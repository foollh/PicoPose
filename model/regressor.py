import torch
from torch import nn
import torch.nn.functional as F


class Regressor(nn.Module):
    """
    A simple MLP to regress scale and rotation from DINOv2 features
    """

    def __init__(
        self,
        cfg, 
        use_tanh_act=True,
        normalize_output=True,
    ):
        super(Regressor, self).__init__()
        self.in_channel = cfg.in_channel
        self.hidden_dim = cfg.hidden_dim
        self.normalize_output = normalize_output
        self.use_tanh_act = use_tanh_act
        
        num_layers = 1
        self.feat_size = 8
        num_gn_groups = 32
        self.features = nn.ModuleList()
        for i in range(num_layers):
            _in_channels = self.in_channel if i == 0 else self.hidden_dim
            self.features.append(nn.Conv2d(_in_channels, self.hidden_dim, kernel_size=1))
            self.features.append(nn.GroupNorm(num_gn_groups, self.hidden_dim))
            self.features.append(nn.ReLU(inplace=True))
            self.features.append(nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=2, padding=1, bias=False))
            self.features.append(nn.GroupNorm(num_gn_groups, self.hidden_dim))
            self.features.append(nn.ReLU(inplace=True))

        self.fc1 = nn.Linear(self.hidden_dim * self.feat_size * self.feat_size, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.act = nn.LeakyReLU(0.1, inplace=True)

        self.translation_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 2),
        )

        self.scale_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 1),
        )

        self.inplane_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 2),
            nn.Tanh() if self.use_tanh_act else nn.Identity(),
        )
        
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        for _i, layer in enumerate(self.features):
            x = layer(x)

        x = x.flatten(1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))

        translation = self.translation_predictor(x)
        scale = self.scale_predictor(x)
        inplane = self.inplane_predictor(x)
        inplane = F.normalize(inplane, dim=1)
        return translation, scale.squeeze(1), inplane



