import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

from model import MLP, set_model_params

def plot_training_curves(log_csv, outdir):
    os.makedirs(outdir, exist_ok=True)
    log = pd.read_csv(log_csv)

    # Accuracy curve
    if "test_acc" in log.columns:
        plt.plot(log["round"], log["test_acc"], marker="o")
        plt.xlabel("Round")
        plt.ylabel("Global Test Accuracy")
        plt.title("Accuracy over FL Rounds")
        plt.grid(True)
        plt.savefig(os.path.join(outdir, "accuracy_curve.png"))
        plt.close()

    # Loss curve
    if "test_loss" in log.columns:
        plt.plot(log["round"], log["test_loss"], marker="o", color="red")
        plt.xlabel("Round")
        plt.ylabel("Global Test Loss")
        plt.title("Loss over FL Rounds")
        plt.grid(True)
        plt.savefig(os.path.join(outdir, "loss_curve.png"))
        plt.close()


def plot_model_eval(model_path, X_test, y_test, outdir):
    os.makedirs(outdir, exist_ok=True)
    device = torch.device("cpu")

    model = MLP(d_in=X_test.shape[1])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
        preds = torch.argmax(logits, dim=1).numpy()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Intrusion Detection")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(outdir, "roc_curve.png"))
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Intrusion Detection")
    plt.savefig(os.path.join(outdir, "confusion_matrix.png"))
    plt.close()
