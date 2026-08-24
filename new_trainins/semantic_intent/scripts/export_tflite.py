import os
import argparse
import torch
import litert_torch as ai_edge_torch
from transformers import AutoTokenizer
from pipeline import IntentModel


def export_tflite(model_dir: str, output_path: str):
    print("Loading model from models/final...")
    model = IntentModel.load(model_dir)
    # tokenizer = AutoTokenizer.from_pretrained(model_dir)

    from export_onnx import FusedIntentNet

    whitening = (
        model.ood.L_
        if getattr(model, "ood", None) is not None and model.ood.method == "mahalanobis"
        else None
    )
    fused = FusedIntentNet(model.encoder, model.clf, model.temperature, whitening)
    fused_model = fused.net
    fused_model.eval()

    print("Preparing dummy inputs...")
    # Mobile typically uses static shapes
    seq_len = 64
    batch_size = 1
    dummy_input = torch.zeros((batch_size, seq_len), dtype=torch.long)
    dummy_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

    print("Converting to TFLite with litert-torch...")

    # Wrap model forward to accept exactly the args we want
    class WrappedModel(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask)

    wrapped = WrappedModel(fused_model)
    wrapped.eval()

    # Convert using AI Edge Torch (now litert-torch)
    edge_model = ai_edge_torch.convert(wrapped, (dummy_input, dummy_mask))

    print("Saving TFLite model...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    edge_model.export(output_path)

    print(f"Export successful! TFLite model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="models/final")
    parser.add_argument("--output", type=str, default="models/final/tflite/intent.tflite")
    args = parser.parse_args()
    export_tflite(args.model_dir, args.output)
