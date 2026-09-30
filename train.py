from pathlib import Path

import joblib

from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# =========================================================
# PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "model"

MODEL_DIR.mkdir(
    exist_ok=True
)

MODEL_PATH = MODEL_DIR / "iris_svm.pkl"


# =========================================================
# LOAD DATA
# =========================================================

iris = load_iris()

X = iris.data
y = iris.target


# =========================================================
# MODEL
# =========================================================

model = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),

        (
            "svm",
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=True,
                random_state=42,
            ),
        ),
    ]
)


# =========================================================
# TRAIN
# =========================================================

model.fit(
    X,
    y
)


# =========================================================
# SAVE
# =========================================================

joblib.dump(
    model,
    MODEL_PATH
)


print("=" * 60)
print("HUẤN LUYỆN MÔ HÌNH SVM HOÀN TẤT")
print("=" * 60)

print(
    "Số mẫu:",
    len(X)
)

print(
    "Số đặc trưng:",
    X.shape[1]
)

print(
    "Các lớp:",
    list(iris.target_names)
)

print(
    "Kernel:",
    "RBF"
)

print(
    "Model:",
    MODEL_PATH
)

print("=" * 60)