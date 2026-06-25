# import pandas as pd
# import joblib

# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, f1_score

# # =========================
# # LOAD DATA
# # =========================
# df = pd.read_csv("../data/data.csv", sep="\t")

# df["Embellishment"] = df["Embellishment"].fillna("None")

# # =========================
# # TF-IDF (SINGLE VECTOR SPACE)
# # =========================
# vectorizer = TfidfVectorizer(
#     ngram_range=(1,2),
#     lowercase=True
# )

# X_vec = vectorizer.fit_transform(df["product_description"])

# # =========================
# # TRAIN FUNCTION
# # =========================
# def train_model(X, y, name):
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y,
#         test_size=0.2,
#         random_state=42
#     )

#     model = LogisticRegression(
#         max_iter=2000,
#         class_weight="balanced"
#     )

#     model.fit(X_train, y_train)
#     pred = model.predict(X_test)

#     print(f"\n===== {name} =====")
#     print("Accuracy:", accuracy_score(y_test, pred))
#     print("F1 Score:", f1_score(y_test, pred, average="weighted"))

#     return model

# # =========================
# # TRAIN ALL MODELS
# # =========================
# fabric_model = train_model(X_vec, df["Fabric"], "Fabric")
# neck_model = train_model(X_vec, df["Neckline"], "Neckline")
# sleeve_model = train_model(X_vec, df["Sleeve"], "Sleeve")
# color_model = train_model(X_vec, df["Color"], "Color")
# category_model = train_model(X_vec, df["Category"], "Category")
# silhouette_model = train_model(X_vec, df["Silhouette"], "Silhouette")
# length_model = train_model(X_vec, df["Length"], "Length")
# embellishment_model = train_model(X_vec, df["Embellishment"], "Embellishment")

# # =========================
# # SAVE MODELS
# # =========================
# joblib.dump(vectorizer, "vectorizer.pkl")

# joblib.dump(fabric_model, "fabric.pkl")
# joblib.dump(neck_model, "neck.pkl")
# joblib.dump(sleeve_model, "sleeve.pkl")
# joblib.dump(color_model, "color.pkl")
# joblib.dump(category_model, "category.pkl")
# joblib.dump(silhouette_model, "silhouette.pkl")
# joblib.dump(length_model, "length.pkl")
# joblib.dump(embellishment_model, "embellishment.pkl")

# print("\n✅ ALL MODELS TRAINED + SAVED SUCCESSFULLY")


import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("../data/data.csv", sep="\t")

df["Embellishment"] = df["Embellishment"].fillna("None")

# =========================
# SINGLE VECTORIZER (IMPORTANT)
# =========================
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X_vec = vectorizer.fit_transform(df["product_description"])

# =========================
# SAFE TRAIN FUNCTION
# =========================
def train_model(X, y, name):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print(f"\n===== {name} =====")
    print("Accuracy:", accuracy_score(y_test, pred))
    print("F1 Score:", f1_score(y_test, pred, average="weighted"))

    return model

# =========================
# TRAIN ALL MODELS
# =========================
fabric_model = train_model(X_vec, df["Fabric"], "Fabric")
neck_model = train_model(X_vec, df["Neckline"], "Neckline")
sleeve_model = train_model(X_vec, df["Sleeve"], "Sleeve")
color_model = train_model(X_vec, df["Color"], "Color")
category_model = train_model(X_vec, df["Category"], "Category")
silhouette_model = train_model(X_vec, df["Silhouette"], "Silhouette")
length_model = train_model(X_vec, df["Length"], "Length")
embellishment_model = train_model(X_vec, df["Embellishment"], "Embellishment")

# =========================
# SAVE MODELS
# =========================
joblib.dump(vectorizer, "vectorizer.pkl")

joblib.dump(fabric_model, "fabric.pkl")
joblib.dump(neck_model, "neck.pkl")
joblib.dump(sleeve_model, "sleeve.pkl")
joblib.dump(color_model, "color.pkl")
joblib.dump(category_model, "category.pkl")
joblib.dump(silhouette_model, "silhouette.pkl")
joblib.dump(length_model, "length.pkl")
joblib.dump(embellishment_model, "embellishment.pkl")

print("\n✅ TRAINING COMPLETE (SAFE VERSION)")