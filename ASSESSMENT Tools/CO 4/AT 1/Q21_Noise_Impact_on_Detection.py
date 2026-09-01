
import matplotlib.pyplot as plt

noise_levels = ["Low", "Medium", "High"]
accuracy = [92, 80, 60]

print("=" * 55)
print("Q21 - NOISE IMPACT ON DETECTION")
print("=" * 55)

for level, value in zip(noise_levels, accuracy):
    print(f"{level:<15} Noise Accuracy: {value}%")

best_index = accuracy.index(max(accuracy))
worst_index = accuracy.index(min(accuracy))
accuracy_drop = max(accuracy) - min(accuracy)

print("\nPerformance Evaluation")
print("-" * 55)
print(f"Best Performance  : {noise_levels[best_index]} Noise ({accuracy[best_index]}%)")
print(f"Worst Performance : {noise_levels[worst_index]} Noise ({accuracy[worst_index]}%)")
print(f"Accuracy Drop     : {accuracy_drop} percentage points")

if accuracy_drop <= 15:
    robustness = "High"
elif accuracy_drop <= 30:
    robustness = "Moderate"
else:
    robustness = "Low"

print(f"System Robustness : {robustness}")

plt.figure(figsize=(8, 5))
plt.plot(noise_levels, accuracy, marker="o", linewidth=2)
plt.title("Noise Level vs Detection Accuracy")
plt.xlabel("Noise Level")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.grid(True)
plt.tight_layout()
plt.show()
