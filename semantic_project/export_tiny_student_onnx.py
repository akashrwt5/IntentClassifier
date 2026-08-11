
import os
import json
import torch
from torch import nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "tiny_semantic_student_v1")

class TinySemanticStudent(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_classes,
        embed_dim=64,
        heads=4,
        layers=2,
        ff_dim=128,
        max_len=24,
        dropout=0.1
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        mask = x.eq(0)

        pos = torch.arange(
            x.size(1), device=x.device
        ).unsqueeze(0)

        h = self.embedding(x) + self.position(pos)

        h = self.encoder(
            h,
            src_key_padding_mask=mask
        )

        valid = (~mask).unsqueeze(-1).float()

        pooled = (
            h * valid
        ).sum(dim=1) / valid.sum(
            dim=1
        ).clamp(min=1.0)

        pooled = self.norm(pooled)

        return self.classifier(pooled)

with open(os.path.join(MODEL_DIR, "config.json"), "r") as f:
    config = json.load(f)

model = TinySemanticStudent(
    vocab_size=config["vocab_size"],
    num_classes=config["num_classes"],
    embed_dim=config["embed_dim"],
    heads=config["num_heads"],
    layers=config["num_layers"],
    ff_dim=config["ff_dim"],
    max_len=config["max_len"],
    dropout=0.0
)

state = torch.load(
    os.path.join(MODEL_DIR, "student_fp32.pt"),
    map_location="cpu"
)

model.load_state_dict(state)
model.eval()

dummy = torch.ones(
    (1, config["max_len"]),
    dtype=torch.long
)

out_path = os.path.join(
    SCRIPT_DIR,
    "tiny_semantic_student_v1.onnx"
)

torch.onnx.export(
    model,
    (dummy,),
    out_path,
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch"},
        "logits": {0: "batch"}
    },
    opset_version=17,
    dynamo=False
)

size_mb = os.path.getsize(out_path) / (1024 * 1024)

print("\nONNX export successful")
print("File:", out_path)
print("Size:", round(size_mb, 3), "MB")
print("\nInput: input_ids [batch, 24]")
print("Output: logits [batch, 11]")
print("\nNext step: INT8 quantization.")
