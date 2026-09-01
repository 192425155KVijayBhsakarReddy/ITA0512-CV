import matplotlib.pyplot as plt
methods = ["Optical Flow", "Motion Models"]
sparse_scores = [3, 2]
dense_scores = [1, 3]
rating_names = {
    3: "High",
    2: "Moderate",
    1: "Low"
}
print("=" * 65)
print("Q24 - MOTION DETECTION UNDER CROWD DENSITY")
print("=" * 65)
print(f"{'Method':<20}{'Sparse Crowd':<20}{'Dense Crowd'}")
print("-" * 65)
for i, method in enumerate(methods):
    print(
        f"{method:<20}"
        f"{rating_names[sparse_scores[i]]:<20}"
        f"{rating_names[dense_scores[i]]}"
    )
best_sparse_index = sparse_scores.index(max(sparse_scores))
best_dense_index = dense_scores.index(max(dense_scores))
print("\nPerformance Evaluation")
print("-" * 65)
print(f"Best for Sparse Crowd : {methods[best_sparse_index]}")
print(f"Best for Dense Crowd  : {methods[best_dense_index]}")
print("\nConclusion")
print("Optical Flow performs best in sparse crowds because individual")
print("movements are easier to distinguish.")
print("Motion Models perform best in dense crowds because they handle")
print("complex and overlapping movement patterns more effectively.")
x = range(len(methods))
width = 0.35
plt.figure(figsize=(9, 5))
plt.bar([i - width / 2 for i in x], sparse_scores, width, label="Sparse Crowd")
plt.bar([i + width / 2 for i in x], dense_scores, width, label="Dense Crowd")
plt.xticks(list(x), methods)
plt.yticks([1, 2, 3], ["Low", "Moderate", "High"])
plt.title("Motion Detection Performance vs Crowd Density")
plt.xlabel("Method")
plt.ylabel("Performance")
plt.legend()
plt.grid(axis="y")
plt.tight_layout()
plt.show()
