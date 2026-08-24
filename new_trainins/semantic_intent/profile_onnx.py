import onnx
import sys
from collections import defaultdict


def profile(path):
    model = onnx.load(path)
    print(f"Profiling {path}")

    total_size = 0
    init_sizes = {}
    op_sizes = defaultdict(int)

    for init in model.graph.initializer:
        size = len(init.raw_data)
        init_sizes[init.name] = size
        total_size += size

    print("Top 15 Largest Weights (Initializers):")
    for k, v in sorted(init_sizes.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {k}: {v / 1024:.2f} KB")

    print(f"\nTotal initializer size: {total_size / (1024*1024):.2f} MB")


if __name__ == "__main__":
    profile(sys.argv[1])
