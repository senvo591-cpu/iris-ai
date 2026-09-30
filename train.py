from sklearn import datasets
from sklearn.svm import SVC
import joblib

# ==============================
# 1. Load Iris Dataset
# ==============================

iris = datasets.load_iris()

X = iris.data
y = iris.target

# ==============================
# 2. Create SVM model
# ==============================

model = SVC(
    kernel="linear",
    probability=True
)

# ==============================
# 3. Train model
# ==============================

model.fit(X, y)

# ==============================
# 4. Save model
# ==============================

joblib.dump(model, "svm_model.pkl")

print("================================")
print("SVM MODEL TRAINED SUCCESSFULLY")
print("================================")
print("Model saved as svm_model.pkl")