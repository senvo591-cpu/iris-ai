from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import numpy as np


# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load("svm_model.pkl")


# ==========================================
# FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="IRIS AI",
    description="Iris Flower Classification using SVM",
    version="2.0.0"
)


# ==========================================
# STATIC FILES
# ==========================================

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


templates = Jinja2Templates(
    directory="templates"
)


# ==========================================
# SPECIES
# ==========================================

species = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}


species_info = {
    "setosa": {
        "name": "Iris Setosa",
        "description": "Iris Setosa có cánh hoa nhỏ, thường dễ nhận biết bởi chiều dài cánh hoa rất ngắn.",
        "color": "Lavender"
    },

    "versicolor": {
        "name": "Iris Versicolor",
        "description": "Iris Versicolor có kích thước trung gian giữa Setosa và Virginica.",
        "color": "Purple"
    },

    "virginica": {
        "name": "Iris Virginica",
        "description": "Iris Virginica thường có kích thước lớn nhất trong ba nhóm Iris.",
        "color": "Deep Purple"
    }
}


# ==========================================
# INPUT MODEL
# ==========================================

class IrisInput(BaseModel):

    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# ==========================================
# HOME
# ==========================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request
        }
    )


# ==========================================
# PREDICT PAGE
# ==========================================

@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):

    return templates.TemplateResponse(
        "predict.html",
        {
            "request": request
        }
    )


# ==========================================
# ANALYTICS PAGE
# ==========================================

@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request):

    return templates.TemplateResponse(
        "analytics.html",
        {
            "request": request
        }
    )


# ==========================================
# FLOWERS PAGE
# ==========================================

@app.get("/flowers", response_class=HTMLResponse)
async def flowers_page(request: Request):

    return templates.TemplateResponse(
        "flowers.html",
        {
            "request": request,
            "species_info": species_info
        }
    )


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "model": "SVM",
        "dataset": "Iris"
    }


# ==========================================
# PREDICTION API
# ==========================================

@app.post("/api/predict")
async def predict(data: IrisInput):

    features = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])

    prediction = int(model.predict(features)[0])

    probabilities = model.predict_proba(features)[0]

    confidence = float(
        probabilities[prediction] * 100
    )

    result = species[prediction]

    return {
        "class_id": prediction,
        "prediction": result,
        "display_name": species_info[result]["name"],
        "confidence": round(confidence, 2)
    }


# ==========================================
# OLD API COMPATIBILITY
# ==========================================

@app.post("/predict")
async def predict_old(data: IrisInput):

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = int(
        model.predict(features)[0]
    )

    return {
        "class_id": prediction,
        "prediction": species[prediction]
    }