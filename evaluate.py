import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow import keras

model = keras.models.load_model("models/model.h5")
test = np.load("data/processed/test.npz")

loss, acc = model.evaluate(test["x"], test["y"], verbose=0)

with open("metrics.json", "w") as f:
    json.dump({"test_loss": float(loss), "test_accuracy": float(acc)}, f, indent=2)

pred = np.argmax(model.predict(test["x"], verbose=0), axis=1)
cm = confusion_matrix(test["y"], pred)

fig, ax = plt.subplots(figsize=(8, 8))
ConfusionMatrixDisplay(cm).plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Fashion-MNIST Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=120, bbox_inches="tight")

print(f"Test loss: {loss:.4f} | Test accuracy: {acc:.4f}")
print("Saved metrics.json and confusion_matrix.png")