import matplotlib.pyplot as plt
models = ["Model A", "Model B", "Model C"]
accuracy = [88, 92, 95]
speed = ["Fast", "Moderate", "Slow"]
robustness = ["Low", "High", "High"]
speed_score = {
    "Fast": 100,
    "Moderate": 70,
    "Slow": 40
}
robustness_score = {
    "High": 100,
    "Low": 50
}
overall_scores = []
print("=" * 65)
print("Q23 - MULTI-MODEL COMPARISON")
print("=" * 65)
print(f"{'Model':<12}{'Accuracy':<12}{'Speed':<12}{'Robustness':<15}{'Score'}")
print("-" * 65)
for model, acc, spd, rob in zip(models, accuracy, speed, robustness):
    score = (0.40 * acc) + (0.30 * speed_score[spd]) + (0.30 * robustness_score[rob])
    overall_scores.append(score)
    print(f"{model:<12}{acc:<12}%{spd:<12}{rob:<15}{score:.2f}")
best_index = overall_scores.index(max(overall_scores))
print("\nPerformance Evaluation")
print("-" * 65)
print(f"Best Overall Model : {models[best_index]}")
print(f"Overall Score      : {overall_scores[best_index]:.2f}")
print(f"Highest Accuracy   : {models[accuracy.index(max(accuracy))]} ({max(accuracy)}%)")
print("\nConclusion")
print(f"{models[best_index]} provides the best balance of accuracy, speed,")
print("and robustness based on the selected evaluation weights.")
plt.figure(figsize=(8, 5))
plt.bar(models, overall_scores)
plt.title("Overall Model Performance")
plt.xlabel("Model")
plt.ylabel("Overall Score")
plt.ylim(0, 100)
plt.grid(axis="y")
plt.tight_layout()
plt.show()
