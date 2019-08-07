import torch.nn.functional as F
import torch.optim as optim

from . import config as cfg
from models import SampleNet


model_name = 'SampleNet'
model = SampleNet().to(cfg.device)
criterion = F.cross_entropy
optimizer = optim.Adam(model.parameters())
