import onnx
import sys
from collections import defaultdict

model = onnx.load(sys.argv[1])
sizes = defaultdict(int)
types = defaultdict(int)

for init in model.graph.initializer:
    # size in bytes = number of elements * bytes per element
    if init.data_type == onnx.TensorProto.FLOAT:
        bytes_per_elem = 4
    elif init.data_type == onnx.TensorProto.FLOAT16:
        bytes_per_elem = 2
    elif init.data_type in [onnx.TensorProto.INT8, onnx.TensorProto.UINT8]:
        bytes_per_elem = 1
    else:
        bytes_per_elem = 4  # fallback

    num_elems = 1
    for dim in init.dims:
        num_elems *= dim

    size_bytes = num_elems * bytes_per_elem
    sizes[init.name] = size_bytes
    types[init.data_type] += size_bytes

print("Total weight size (MB):", sum(sizes.values()) / 1024 / 1024)
for k, v in sorted(sizes.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"{k}: {v / 1024 / 1024:.2f} MB")
print("Types (bytes):", dict(types))
