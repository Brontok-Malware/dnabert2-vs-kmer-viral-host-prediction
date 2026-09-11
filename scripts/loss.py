import matplotlib.pyplot as plt

epochs = [1, 2, 3]
train_loss = [0.3142, 0.1864, 0.1120]
eval_loss = [0.1985, 0.2210, 0.2587]

plt.figure(figsize=(6, 3.5))
plt.plot(epochs, train_loss, marker="o", label="Train Loss", color="#1f77b4")
plt.plot(epochs, eval_loss, marker="s", label="Eval Loss", color="#d62728")
plt.xlabel("Epoche")
plt.ylabel("Loss")
plt.title("Trainings- und Evaluierungsverlust (DNABERT-2)")
plt.xticks([1, 2, 3])
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=300)
