import os
from pathlib import Path
import urllib.request

url = "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"
filename = "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"

targets = [
    Path.home() / ".keras" / "models" / filename,
    Path(__file__).resolve().parent / ".cache" / "keras" / "models" / filename,
]

print(f"Downloading MobileNetV2 ImageNet weights from {url}...")
for target in targets:
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(target))
        print(f"Successfully downloaded {target.stat().st_size} bytes to {target}")
    except Exception as e:
        print(f"Could not write to {target}: {e}")

print("Pretrained weights setup complete.")
