import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

with open("params.yaml") as f:
    params = yaml.safe_load(f)["preprocess"]

raw = np.load("data/raw/fashion_mnist.npz")

x_train = (raw["x_train"].astype("float32") / 255.0 - 0.286) / 0.353
y_train = raw["y_train"]
x_test = (raw["x_test"].astype("float32") / 255.0 - 0.286) / 0.353
y_test = raw["y_test"]

x_tr, x_val, y_tr, y_val = train_test_split(
    x_train,
    y_train,
    test_size=params["test_size"],
    random_state=params["seed"],
    stratify=y_train,
)

os.makedirs("data/processed", exist_ok=True)
np.savez("data/processed/train.npz", x=x_tr, y=y_tr)
np.savez("data/processed/val.npz", x=x_val, y=y_val)
np.savez("data/processed/test.npz", x=x_test, y=y_test)

print("Saved processed data:", x_tr.shape, x_val.shape, x_test.shape)

print("Preprocessing finished successfully")