import os
import argparse
import torch
import coremltools as ct
from transformers import AutoTokenizer

from pipeline import IntentModel
from export_onnx import FusedIntentNet


def export_coreml(model_dir: str, output_path: str):
    """
    Exports the trained IntentModel to Core ML format, optimizing for Apple Neural Engine (ANE).
    """
    print(f"Loading model from {model_dir}...")
    # Load model and tokenizer
    model = IntentModel.load(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Create the fused PyTorch model
    fused = FusedIntentNet(model)
    fused.eval()

    # Define dummy inputs for sequence length of 64 and batch size 1 (required for ANE)
    seq_len = 64
    dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (1, seq_len), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, seq_len), dtype=torch.long)

    print("Exporting model with torch.export...")
    # Export using PyTorch 2.x torch.export and run decompositions
    exported_model = torch.export.export(fused.net, (dummy_input_ids, dummy_attention_mask))
    exported_model = exported_model.run_decompositions()
    print("Model successfully exported via torch.export!")

    print("Converting to Core ML...")
    # Convert to Core ML
    mlmodel = ct.convert(
        exported_model,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, seq_len), dtype=torch.int32),
            ct.TensorType(name="attention_mask", shape=(1, seq_len), dtype=torch.int32),
        ],
        outputs=[ct.TensorType(name="intent_logits", dtype=torch.float32)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,  # Allow running on ANE, GPU, or CPU
    )

    print(f"Saving Core ML model to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mlmodel.save(output_path)
    print("Conversion successful!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export IntentModel to Core ML")
    parser.add_argument(
        "--model-dir", type=str, default="models/final", help="Directory of the trained model"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/final/coreml/intent.mlpackage",
        help="Output path for the .mlpackage",
    )
    args = parser.parse_args()

    export_coreml(args.model_dir, args.output)
