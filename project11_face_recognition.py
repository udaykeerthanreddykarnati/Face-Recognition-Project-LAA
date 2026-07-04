import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

FACES_FOLDER = "faces"
PERSONS = ["person1", "person2", "person3", "person4"]
IMAGES_PER_PERSON = 4
TEST_FILE = "test.jpg"
TARGET_SIZE = (92, 112)

def load_image(path):
    img = Image.open(path).convert("L")
    img = img.resize(TARGET_SIZE, Image.LANCZOS)
    return np.array(img, dtype=np.float64)

# ================= STEP 1 =================
print("="*60)
print("STEP 1: MATRIX REPRESENTATION")
print("="*60)

w, h = TARGET_SIZE
mn = w * h

P_cols = []
labels = []
training_images = []

for person_id, person in enumerate(PERSONS):
    for i in range(1, IMAGES_PER_PERSON + 1):
        path = os.path.join(FACES_FOLDER, person, f"{i}.jpg")
        img = load_image(path)
        P_cols.append(img.reshape(mn,1))
        labels.append(person_id)
        training_images.append(img)

P = np.hstack(P_cols)
labels = np.array(labels)

print("P shape:", P.shape)
print("Each column = one image vector")
print("Matrix represents dataset in vector space")
print("Matrix acts as linear transformation mapping images to vector space")

# 🔹 SHOW TRAINING IMAGES
fig, axes = plt.subplots(4, 4, figsize=(8,8))
for i, ax in enumerate(axes.flat):
    ax.imshow(training_images[i], cmap='gray')
    ax.set_title(f"P{labels[i]+1}")
    ax.axis('off')
plt.show()

# ================= STEP 2 =================
print("\n" + "="*60)
print("STEP 2: MEAN + CENTERING")
print("="*60)

mean_face = np.mean(P, axis=1)
mean_col = mean_face.reshape(-1,1)
P_centered = P - mean_col @ np.ones((1,P.shape[1]))

print("Data centered → removes bias")
print("Why centering? → removes lighting and intensity variations")

# 🔹 SHOW MEAN FACE
plt.imshow(mean_face.reshape(h, w), cmap='gray')
plt.title("Mean Face")
plt.axis('off')
plt.show()

# ================= STEP 2.5 =================
print("\nSTEP 2.5: MATRIX SIMPLIFICATION (RREF)")

def rref(A):
    A = A.astype(float)
    rows, cols = A.shape
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        pivot = np.argmax(abs(A[r:, c])) + r
        if A[pivot, c] == 0:
            continue
        A[[r, pivot]] = A[[pivot, r]]
        A[r] = A[r] / A[r, c]
        for i in range(rows):
            if i != r:
                A[i] -= A[i, c] * A[r]
        r += 1
    return A

print(rref(P_centered[:10,:]))
print("RREF → identifies independent columns")
print("Why RREF? → detect linear dependence")

# ================= STEP 3 =================
print("\n" + "="*60)
print("STEP 3: STRUCTURE OF SPACE")
print("="*60)

rank = np.linalg.matrix_rank(P)
nullity = P.shape[1] - rank

print("Rank:", rank)
print("Nullity:", nullity)
print("Column space (C(A)) → span of face vectors")
print("Subspace → all possible face combinations")
print("Null space (N(A)) → vectors mapped to zero (redundant features)")

# ================= STEP 4 =================
print("\n" + "="*60)
print("STEP 4: REMOVE REDUNDANCY")
print("="*60)

print("Independent vectors form BASIS")
print("Basis → set of independent vectors spanning column space")
print("Basis dimension =", rank)

# ================= STEP 5 =================
print("\n" + "="*60)
print("STEP 5: PCA (EIGEN / PATTERN DISCOVERY)")
print("="*60)

PTP = P_centered.T @ P_centered
Values, Vectors = np.linalg.eig(PTP)

order = np.argsort(Values.real)[::-1]
Values = Values.real[order]
Vectors = Vectors[:, order]

EigenVectors = P_centered @ Vectors
EigenVectors = EigenVectors / np.linalg.norm(EigenVectors, axis=0)

print("Eigenvalues:\n", Values)
print("Large eigenvalues = dominant facial patterns")
print("Why eigen? → extract important features")

# 🔹 SHOW EIGENFACES
fig, axes = plt.subplots(1, 4, figsize=(10,3))
for i in range(4):
    ef = EigenVectors[:, i].reshape(h, w)
    axes[i].imshow(ef, cmap='gray')
    axes[i].set_title(f"Eigen {i+1}")
    axes[i].axis('off')
plt.show()

# ================= STEP 6 =================
print("\n" + "="*60)
print("STEP 6: ORTHOGONALIZATION")
print("="*60)

print("E^T E:\n", EigenVectors.T @ EigenVectors)
print("Eigenvectors form orthogonal basis (Gram-Schmidt equivalent)")
print("Orthogonal basis ensures independent feature directions")

# ================= STEP 7 =================
print("\n" + "="*60)
print("STEP 7: PROJECTION")
print("="*60)

test_img = load_image(os.path.join(FACES_FOLDER, TEST_FILE))
U = test_img.reshape(mn,1)

U_centered = U - mean_col
W_test = EigenVectors.T @ U_centered

print("Projection coefficients:", W_test.flatten())
print("Projection → maps image into eigen space")
print("Projection onto subspace spanned by eigenfaces")
print("Why projection? → compare faces in reduced space")

# ================= STEP 8 =================
print("\n" + "="*60)
print("STEP 8: LEAST SQUARES")
print("="*60)

print("Projection = Least Squares Solution")
print("Formula: x = (A^T A)^(-1) A^T b")
print("Least squares used because system may be inconsistent")
print("Least squares → best approximation of test image")

# ================= STEP 9 =================
print("\n" + "="*60)
print("STEP 9: SYSTEM SIMPLIFICATION")
print("="*60)

D = np.diag(Values)
print("Diagonal matrix:\n", D)
print("Diagonalization reduces complexity")

# ================= STEP CONNECTION =================
print("\n" + "="*60)
print("STEP CONNECTION FLOW")
print("="*60)

print("RREF → independence")
print("Rank → basis size")
print("Eigen → orthogonal basis")
print("Projection → coordinates in new space")

# ================= MATCHING =================
print("\n" + "="*60)
print("STEP 10: CLASSIFICATION")
print("="*60)

W_train = EigenVectors.T @ P_centered
dists = [np.linalg.norm(W_test.flatten() - W_train[:,i]) for i in range(P.shape[1])]

best_idx = np.argmin(dists)
predicted_person = labels[best_idx]

print("Matched Image Index:", best_idx)
print("Predicted Person:", predicted_person + 1)

# ================= FINAL =================
print("\n" + "="*60)
print("FINAL OUTPUT")
print("="*60)

print("Recognized person:", predicted_person + 1)
print("Dimensionality reduced from", mn, "to", P.shape[1])
print("Patterns extracted using eigenvalues")
print("Application Type: Pattern Recognition (Face Recognition)")
print("Prediction: Identifying closest matching person")
print("Pipeline: Raw images → mathematical model → recognized person")
print("Complete Linear Algebra Pipeline Implemented")

# 🔹 FINAL VISUAL OUTPUT
recon = EigenVectors @ W_test + mean_col
recon_img = np.clip(recon.reshape(h, w), 0, 255)

matched_img = training_images[best_idx]

fig, axes = plt.subplots(1, 3, figsize=(10,4))

axes[0].imshow(test_img, cmap='gray')
axes[0].set_title("Test Image")
axes[0].axis('off')

axes[1].imshow(recon_img, cmap='gray')
axes[1].set_title("Reconstructed")
axes[1].axis('off')

axes[2].imshow(matched_img, cmap='gray')
axes[2].set_title(f"Match P{predicted_person+1}")
axes[2].axis('off')

plt.show()