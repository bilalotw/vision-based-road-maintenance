import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

class SeverityNet(nn.Module):
    def __init__(self, num_classes=3, geometry_dim=11, use_image=True, use_geometry=True,
                 pretrained=True, dropout=0.35):
        super().__init__()
        if not (use_image or use_geometry): raise ValueError("At least one input modality required")
        self.use_image=use_image; self.use_geometry=use_geometry
        if use_image:
            backbone=resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
            self.image_encoder=nn.Sequential(*list(backbone.children())[:-1]); image_dim=512
        else: image_dim=0
        if use_geometry:
            self.geometry_encoder=nn.Sequential(nn.Linear(geometry_dim,64),nn.ReLU(),
                nn.BatchNorm1d(64),nn.Dropout(dropout),nn.Linear(64,64),nn.ReLU())
            geom_dim=64
        else: geom_dim=0
        self.head=nn.Sequential(nn.Linear(image_dim+geom_dim,128),nn.ReLU(),
                                nn.Dropout(dropout),nn.Linear(128,num_classes))
    def forward(self,image,geometry):
        parts=[]
        if self.use_image: parts.append(self.image_encoder(image).flatten(1))
        if self.use_geometry: parts.append(self.geometry_encoder(geometry))
        return self.head(torch.cat(parts,dim=1))

def mc_dropout_mode(model):
    model.eval()
    for m in model.modules():
        if isinstance(m,nn.Dropout): m.train()

@torch.no_grad()
def mc_predict(model,image,geometry,passes=30):
    mc_dropout_mode(model); samples=[]
    for _ in range(passes): samples.append(torch.softmax(model(image,geometry),dim=1))
    p=torch.stack(samples); mean=p.mean(0)
    entropy=-(mean*mean.clamp_min(1e-8).log()).sum(1)
    expected=-(p*p.clamp_min(1e-8).log()).sum(2).mean(0)
    return {"probabilities":mean,"predictive_entropy":entropy,
            "mutual_information":(entropy-expected).clamp_min(0),"variance":p.var(0)}
