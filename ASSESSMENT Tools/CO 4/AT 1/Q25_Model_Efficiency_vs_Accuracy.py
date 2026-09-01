import matplotlib.pyplot as plt
models = ["Model A", "Model B", "Model C"]
accuracy = [90, 85, 88]
computation = ["High", "Low", "Moderate"]
computation_score = {
    "Low": 1,
    "Moderate": 2,
    "High": 3
}
print("=" * 65)
print("Q25 - MODEL EFFICIENCY VS ACCURACY")
print("=" * 65)
print(f"{'Model':<12}{'Accuracy':<12}{'Computation':<15}")
print("-" * 65)
for model, acc, comp in zip(models, accuracy, computation):
    print(f"{model:<12}{acc:<12}%{comp:<15}")
efficiency_score = []
for acc, comp in zip(accuracy, computation):
    computation_efficiency = {
        "Low": 100,
        "Moderate": 70,
        "High": 40
    }[comp]

    score = (0.70 * acc) + (0.30 * computation_efficiency)
    efficiency_score.append(score)
best_index = efficiency_score.index(max(efficiency_score))

print("\nEdge Device Evaluation")
print("-" * 65)

for model, score in zip(models, efficiency_score):
    print(f"{model:<12} Edge Suitability Score: {score:.2f}")

print("\nConclusion")
print(f"{models[best_index]} is the most suitable model for edge devices")
print("because it provides a good balance between accuracy and computation.")

plt.figure(figsize=(8, 5))
plt.scatter(computation, accuracy, s=120)

for i, model in enumerate(models):
    plt.annotate(
        model,
        (computation[i], accuracy[i]),
        xytext=(5, 5),
        textcoords="offset points"
    )

plt.title("Model Efficiency vs Accuracy")
plt.xlabel("Computational Requirement")
plt.ylabel("Accuracy (%)")
plt.ylim(75, 100)
plt.grid(True)
plt.tight_layout()
plt.show()
