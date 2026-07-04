# LAA Face Recognition Project

A face recognition system built from scratch using **pure Linear Algebra** — no ML libraries. Implements PCA (Eigenfaces method) step by step, covering matrix representation, mean centering, eigendecomposition, projection, and nearest-neighbor classification.

---

## How It Works

The pipeline maps face images into a mathematical space and identifies the closest match to a test image:

```
Raw Images → Matrix → Mean Center → PCA (Eigenfaces) → Project → Match
```

| Step | What it does |
|------|-------------|
| 1 | Load images as column vectors, build matrix P |
| 2 | Compute mean face, center the data |
| 2.5 | RREF to detect linear dependence |
| 3 | Compute rank and nullity |
| 4 | Identify basis (independent vectors) |
| 5 | PCA via eigendecomposition → Eigenfaces |
| 6 | Verify orthogonality of eigenvectors |
| 7 | Project test image into eigenspace |
| 8 | Least squares approximation |
| 9 | Diagonalization for complexity reduction |
| 10 | Nearest-neighbor classification → match |

---

## Folder Structure

```
project/
│
├── faces/
│   ├── person1/
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   ├── 3.jpg
│   │   └── 4.jpg
│   ├── person2/
│   ├── person3/
│   ├── person4/
│   └── test.jpg          ← the image to recognize
│
└── project11_face_recognition.py
```

---

## Requirements

```bash
pip install numpy pillow matplotlib
```

No OpenCV, no sklearn, no deep learning — just NumPy.

---

## Setup

### 1. Prepare your training images

- Create a `faces/` folder in the same directory as the script
- Inside it, create folders: `person1`, `person2`, `person3`, `person4`
- Add **4 grayscale face images** per person, named `1.jpg`, `2.jpg`, `3.jpg`, `4.jpg`
- Images can be any size — they are automatically resized to `92×112`

### 2. Add a test image

- Place the image you want to recognize as `faces/test.jpg`

### 3. Run

```bash
python project11_face_recognition.py
```

---

## Output

The script prints each step's result to the console and shows these plots:

- **Training images grid** — all 16 training images
- **Mean face** — average of all training images
- **Eigenfaces** — top 4 principal components
- **Final result** — test image, reconstructed image, and matched person side by side

Console output example:
```
Matched Image Index: 3
Predicted Person: 2
Dimensionality reduced from 10304 to 16
```

---

## Customization

| Variable | Default | Description |
|----------|---------|-------------|
| `PERSONS` | `["person1"..."person4"]` | Folder names for each person |
| `IMAGES_PER_PERSON` | `4` | Training images per person |
| `TARGET_SIZE` | `(92, 112)` | Resize all images to this |
| `TEST_FILE` | `"test.jpg"` | Test image filename |

To add more people, add a folder under `faces/`, add the name to `PERSONS`, and place 4 images inside.

---

## Concepts Covered

- Vector spaces and subspaces
- Matrix rank and nullity
- RREF and linear independence
- PCA via eigendecomposition
- Gram-Schmidt orthogonalization
- Orthogonal projection
- Least squares approximation
- Nearest-neighbor classification

---

## Notes

- Works best with consistent lighting and frontal face images
- All images are converted to grayscale automatically
- Accuracy improves with more training images per person
