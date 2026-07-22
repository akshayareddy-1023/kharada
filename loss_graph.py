import matplotlib.pyplot as plt

# ---------------------------
# Transformer Loss
# ---------------------------

epochs = list(range(1,31))

loss = [
4.52,
4.31,
4.18,
4.06,
3.95,
3.88,
3.80,
3.73,
3.69,
3.64,
3.60,
3.55,
3.52,
3.48,
3.45,
3.42,
3.39,
3.36,
3.34,
3.31,
3.29,
3.27,
3.25,
3.23,
3.21,
3.20,
3.19,
3.18,
3.17,
3.15
]

plt.figure(figsize=(10,6))

plt.plot(
    epochs,
    loss,
    marker='o',
    linewidth=2,
    label="Transformer"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Transformer Training Loss")

plt.grid(True)

plt.legend()

plt.savefig("graphs/transformer_loss.png",dpi=300)

plt.show()

print("Graph Saved Successfully")