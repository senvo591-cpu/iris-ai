from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# =========================================================
# 1. PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
MODEL_DIR = BASE_DIR / "model"

TEMPLATES_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
(STATIC_DIR / "css").mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "js").mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


# =========================================================
# 2. FASTAPI
# =========================================================

app = FastAPI(
    title="IRIS AI",
    description="Ứng dụng phân loại hoa Iris bằng mô hình SVM",
    version="2.0.0",
)


# =========================================================
# 3. STATIC + TEMPLATES
# =========================================================

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)

templates = Jinja2Templates(
    directory=str(TEMPLATES_DIR)
)


# =========================================================
# 4. DỮ LIỆU IRIS
# =========================================================

iris = load_iris()

X = iris.data
y = iris.target

CLASS_NAMES = list(iris.target_names)

FEATURE_NAMES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]


# =========================================================
# 5. LOAD MODEL
# =========================================================

MODEL_PATH = MODEL_DIR / "iris_svm.pkl"


def create_model():
    """
    Tạo pipeline:
    StandardScaler + SVM RBF
    """

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

    return model


def load_or_train_model():

    if MODEL_PATH.exists():

        try:
            model = joblib.load(MODEL_PATH)
            return model

        except Exception:
            pass

    model = create_model()

    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)

    return model


model = load_or_train_model()


# =========================================================
# 6. THÔNG TIN HOA
# =========================================================

FLOWER_INFO = {

    "setosa": {
        "name": "Iris Setosa",
        "name_vi": "Hoa Iris Setosa",
        "description": (
            "Iris Setosa thường có hoa nhỏ, cánh hoa nổi bật "
            "và là một trong ba loài thuộc bộ dữ liệu Iris."
        ),
        "color": "Tím xanh / trắng",
        "origin": "Bắc Mỹ và Đông Á",
        "characteristic": (
            "Thường có cánh hoa nhỏ và chiều dài cánh hoa "
            "ngắn hơn rõ rệt so với hai loài còn lại."
        ),
        "image": (
            "https://commons.wikimedia.org/wiki/"
            "Special:Redirect/file/Iris_setosa.JPG"
        ),
    },

    "versicolor": {
        "name": "Iris Versicolor",
        "name_vi": "Hoa Iris Versicolor",
        "description": (
            "Iris Versicolor còn được gọi là Blue Flag Iris, "
            "thường có hoa màu xanh tím."
        ),
        "color": "Xanh tím",
        "origin": "Bắc Mỹ",
        "characteristic": (
            "Có kích thước trung gian. Các giá trị "
            "petal length và petal width thường nằm giữa Setosa "
            "và Virginica."
        ),
        "image": (
            "https://commons.wikimedia.org/wiki/"
            "Special:Redirect/file/Iris_versicolor.jpg"
        ),
    },

    "virginica": {
        "name": "Iris Virginica",
        "name_vi": "Hoa Iris Virginica",
        "description": (
            "Iris Virginica là loài Iris có kích thước thường "
            "lớn hơn trong bộ dữ liệu Iris."
        ),
        "color": "Tím xanh",
        "origin": "Đông Bắc và Đông Nam Hoa Kỳ",
        "characteristic": (
            "Thường có chiều dài và chiều rộng cánh hoa "
            "lớn hơn hai loài còn lại."
        ),
        "image": (
            "https://commons.wikimedia.org/wiki/"
            "Special:Redirect/file/IMG_7911-Iris_virginica.jpg"
        ),
    },
}


# =========================================================
# 7. HÀM DỰ ĐOÁN
# =========================================================

def predict_flower(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
) -> Dict[str, Any]:

    values = np.array(
        [
            [
                sepal_length,
                sepal_width,
                petal_length,
                petal_width,
            ]
        ]
    )

    prediction = int(model.predict(values)[0])

    probabilities = model.predict_proba(values)[0]

    confidence = float(
        probabilities[prediction] * 100
    )

    class_name = CLASS_NAMES[prediction]

    info = FLOWER_INFO[class_name]

    return {
        "success": True,
        "prediction": prediction,
        "class": class_name,
        "name": info["name"],
        "name_vi": info["name_vi"],
        "description": info["description"],
        "characteristic": info["characteristic"],
        "image": info["image"],
        "color": info["color"],
        "origin": info["origin"],
        "confidence": round(confidence, 2),
        "probabilities": {
            CLASS_NAMES[i]: round(
                float(probabilities[i] * 100),
                2
            )
            for i in range(len(CLASS_NAMES))
        },
        "input": {
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width,
        },
    }


# =========================================================
# 8. HOME
# =========================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "title": "IRIS AI | Trang chủ"
        },
    )


# =========================================================
# 9. TRANG DỰ ĐOÁN
# =========================================================

@app.get(
    "/predict",
    response_class=HTMLResponse
)
async def predict_page(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="predict.html",
        context={
            "title": "IRIS AI | Nhận diện hoa"
        },
    )


# =========================================================
# 10. API DỰ ĐOÁN
# =========================================================

@app.post("/predict")
async def predict_api(data: Dict[str, Any]):

    try:

        sepal_length = float(data["sepal_length"])
        sepal_width = float(data["sepal_width"])
        petal_length = float(data["petal_length"])
        petal_width = float(data["petal_width"])

    except Exception:

        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": (
                    "Dữ liệu đầu vào không hợp lệ."
                ),
            },
        )

    values = [
        sepal_length,
        sepal_width,
        petal_length,
        petal_width,
    ]

    if any(v <= 0 for v in values):

        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": (
                    "Các thông số phải lớn hơn 0."
                ),
            },
        )

    try:

        result = predict_flower(
            sepal_length,
            sepal_width,
            petal_length,
            petal_width,
        )

        return JSONResponse(
            content=result
        )

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": str(e),
            },
        )


# =========================================================
# 11. VISUALIZATION
# =========================================================

@app.get(
    "/visualization",
    response_class=HTMLResponse
)
async def visualization(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="visualization.html",
        context={
            "title": "IRIS AI | Trực quan dữ liệu"
        },
    )


# =========================================================
# 12. ANALYTICS
# =========================================================

@app.get(
    "/analytics",
    response_class=HTMLResponse
)
async def analytics(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="analytics.html",
        context={
            "title": "IRIS AI | Đánh giá mô hình"
        },
    )


# =========================================================
# 13. SPECIES
# =========================================================

@app.get(
    "/species",
    response_class=HTMLResponse
)
async def species(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="species.html",
        context={
            "title": "IRIS AI | Các loài hoa"
        },
    )


# =========================================================
# 14. API THÔNG TIN HOA
# =========================================================

@app.get("/api/species")
async def api_species():

    return FLOWER_INFO


# =========================================================
# 15. API PHÂN TÍCH MÔ HÌNH
# =========================================================

@app.get("/api/analytics")
async def api_analytics():

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    evaluation_model = create_model()

    evaluation_model.fit(
        X_train,
        y_train
    )

    y_pred = evaluation_model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    report = classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),

        "confusion_matrix": cm.tolist(),

        "classification_report": {
            name: {
                "precision": round(
                    values["precision"] * 100,
                    2
                ),
                "recall": round(
                    values["recall"] * 100,
                    2
                ),
                "f1-score": round(
                    values["f1-score"] * 100,
                    2
                ),
            }
            for name, values in report.items()
            if name in CLASS_NAMES
        },

        "features": FEATURE_NAMES,

        "classes": CLASS_NAMES,
    }


# =========================================================
# 16. HEALTH CHECK
# =========================================================

@app.get("/health")
async def health():

    return {
        "status": "online",
        "model_loaded": model is not None,
        "classes": CLASS_NAMES,
    }


# =========================================================
# 17. ROOT TEST
# =========================================================

@app.get("/api")
async def api_root():

    return {
        "message": "IRIS AI API is running",
        "version": "2.0.0",
        "endpoints": [
            "/",
            "/predict",
            "/visualization",
            "/analytics",
            "/species",
            "/health",
            "/docs",
        ],
    }