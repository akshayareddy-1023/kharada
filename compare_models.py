import os
import matplotlib.pyplot as plt

# Create graphs folder automatically
os.makedirs("graphs", exist_ok=True)

# ------------------------------
# Transformer Metrics
# ------------------------------

transformer = {
    "BLEU": 4.04,
    "Accuracy": 0.82,
    "Precision": 0.81,
    "Recall": 0.80,
    "F1": 0.80
}

# ------------------------------
# LSTM Metrics
# ------------------------------

lstm = {
    "BLEU": 2.80,
    "Accuracy": 0.50,
    "Precision": 0.52,
    "Recall": 0.49,
    "F1": 0.50
}

metrics = list(transformer.keys())

t_values = list(transformer.values())
l_values = list(lstm.values())

x = range(len(metrics))
width = 0.35

plt.figure(figsize=(10,6))

bars1 = plt.bar(
    [i-width/2 for i in x],
    t_values,
    width,
    label="Transformer"
)

bars2 = plt.bar(
    [i+width/2 for i in x],
    l_values,
    width,
    label="LSTM"
)

plt.xticks(x, metrics, fontsize=11)
plt.ylabel("Score", fontsize=12)
plt.xlabel("Evaluation Metrics", fontsize=12)
plt.title("Performance Comparison of Transformer and LSTM Models", fontsize=14)

plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Show values above bars
for bar in bars1:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.03,
        f"{height:.2f}",
        ha='center',
        fontsize=10
    )

for bar in bars2:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.03,
        f"{height:.2f}",
        ha='center',
        fontsize=10
    )

plt.tight_layout()

plt.savefig("graphs/model_comparison.png", dpi=300)

plt.show()

print("Graph saved successfully in graphs/model_comparison.png")