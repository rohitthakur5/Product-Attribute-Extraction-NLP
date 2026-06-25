from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Product Attribute Extraction API")

# =========================
# INPUT SCHEMA (IMPORTANT FIX)
# =========================

class ProductInput(BaseModel):
    product_description: str


# =========================
# LOAD MODELS
# =========================

fabric_model = joblib.load("../models/fabric.pkl")
neck_model = joblib.load("../models/neck.pkl")
sleeve_model = joblib.load("../models/sleeve.pkl")
color_model = joblib.load("../models/color.pkl")
cat_model = joblib.load("../models/category.pkl")
sil_model = joblib.load("../models/silhouette.pkl")
len_model = joblib.load("../models/length.pkl")
emb_model = joblib.load("../models/embellishment.pkl")

vectorizer = joblib.load("../models/vectorizer.pkl")


# =========================
# ROOT CHECK (optional but good)
# =========================

@app.get("/")
def home():
    return {"message": "API is running 🚀"}


# =========================
# MAIN API ENDPOINT
# =========================

@app.post("/extract")
def extract(data: ProductInput):

    # convert input text
    text = [data.product_description]
    X_vec = vectorizer.transform(text)

    # predictions
    result = {
        "Silhouette": sil_model.predict(X_vec)[0],
        "Fabric": fabric_model.predict(X_vec)[0],
        "Neckline": neck_model.predict(X_vec)[0],
        "Sleeve": sleeve_model.predict(X_vec)[0],
        "Length": len_model.predict(X_vec)[0],
        "Embellishment": emb_model.predict(X_vec)[0],
        "Color": color_model.predict(X_vec)[0],
        "Category": cat_model.predict(X_vec)[0]
    }

    return {
        "input": data.product_description,
        "extracted_attributes": result
    }