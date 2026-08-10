import onnx
import sys

model = onnx.load(sys.argv[1])
matmuls = [n.name for n in model.graph.initializer if "MatMul" in n.name and "quantized" in n.name]
print(f"Total MatMul quantized weights: {len(matmuls)}")
for m in sorted(matmuls)[:15]:
    print(m)
