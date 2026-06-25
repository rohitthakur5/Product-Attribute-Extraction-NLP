import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
df = pd.read_csv("../data/data.csv", sep='\t')

df.head()

print(df.info())


df["Embellishment"] = df["Embellishment"].fillna("None")

X = df["product_description"]

y_fabric = df["Fabric"]



# 5. TRAIN TEST SPLIT
# =========================
from sklearn.model_selection import train_test_split

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X,
    y_fabric,
    test_size=0.3,
    random_state=42
)


print(df["Fabric"].value_counts())
# 6. TEXT TO NUMBER (TF-IDF)
# =========================


from sklearn.feature_extraction.text import TfidfVectorizer




vectorizer = TfidfVectorizer(ngram_range=(1,2))

X_vec = vectorizer.fit_transform(df["product_description"])

# 7. TRAIN MODEL
# =========================

from sklearn.linear_model import LogisticRegression

fabric_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)

fabric_model.fit(
    X_train_vec,
    y_train
)


# 8. PREDICTION
# =========================

y_pred = fabric_model.predict(X_test_vec)

print(y_test.values)
print(y_pred)

# 9. EVALUATION
# =========================
from sklearn.metrics import accuracy_score, f1_score

print("Accuracy:", accuracy_score(y_test,y_pred))

print("F1 Score:", f1_score(y_test,y_pred,average="weighted"))

# =========================
# NECKLINE MODEL
# =========================

y_neck = df["Neckline"]


# train test split

X_train_text, X_test_text, y_train_neck, y_test_neck = train_test_split(
    X,
    y_neck,
    test_size=0.3,
    random_state=42
)


# TF-IDF

X_train_vec = vectorizer.fit_transform(X_train_text)

X_test_vec = vectorizer.transform(X_test_text)


# Model

neck_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)


neck_model.fit(
    X_train_vec,
    y_train_neck
)


# Prediction

neck_pred = neck_model.predict(X_test_vec)


# Evaluation

print("Neck Accuracy:",
      accuracy_score(y_test_neck, neck_pred))

print("Neck F1:",
      f1_score(y_test_neck, neck_pred, average="weighted"))


# =========================
# SLEEVE MODEL
# =========================

y_sleeve = df["Sleeve"]


X_train_text, X_test_text, y_train_sleeve, y_test_sleeve = train_test_split(
    X,
    y_sleeve,
    test_size=0.3,
    random_state=42
)


X_train_vec = vectorizer.fit_transform(X_train_text)

X_test_vec = vectorizer.transform(X_test_text)


sleeve_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)


sleeve_model.fit(
    X_train_vec,
    y_train_sleeve
)


sleeve_pred = sleeve_model.predict(X_test_vec)


print("Sleeve Accuracy:",
      accuracy_score(y_test_sleeve, sleeve_pred))

print("Sleeve F1:",
      f1_score(y_test_sleeve, sleeve_pred, average="weighted"))


# =========================
# COLOR MODEL (FINAL)
# =========================

X = df["product_description"]
y_color = df["Color"]


# split
from sklearn.model_selection import train_test_split

X_train_text, X_test_text, y_train_color, y_test_color = train_test_split(
    X,
    y_color,
    test_size=0.2,
    random_state=42
)


# TF-IDF (word based)
from sklearn.feature_extraction.text import TfidfVectorizer

color_vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    lowercase=True
)


X_train_vec_color = color_vectorizer.fit_transform(X_train_text)
X_test_vec_color = color_vectorizer.transform(X_test_text)


# Model
from sklearn.linear_model import LogisticRegression

color_model = LogisticRegression(
    max_iter=5000,
    class_weight="balanced",
    solver="lbfgs"
)


color_model.fit(X_train_vec_color, y_train_color)


# Prediction
color_pred = color_model.predict(X_test_vec_color)


# Evaluation
from sklearn.metrics import accuracy_score, f1_score

print("Color Accuracy:", accuracy_score(y_test_color, color_pred))
print("Color F1:", f1_score(y_test_color, color_pred, average="weighted"))

df["Color"].value_counts()
X = df["product_description"]
y_category = df["Category"]


# train-test split
X_train_text, X_test_text, y_train_cat, y_test_cat = train_test_split(
    X,
    y_category,
    test_size=0.2,
    random_state=42
)


# TF-IDF
cat_vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    lowercase=True
)

X_train_vec_cat = cat_vectorizer.fit_transform(X_train_text)
X_test_vec_cat = cat_vectorizer.transform(X_test_text)


# model
cat_model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced"
)

cat_model.fit(X_train_vec_cat, y_train_cat)


# prediction
y_pred_cat = cat_model.predict(X_test_vec_cat)


# evaluation
print("Category Accuracy:", accuracy_score(y_test_cat, y_pred_cat))
print("Category F1:", f1_score(y_test_cat, y_pred_cat, average="weighted"))


# sample check
print("\nActual:")
print(y_test_cat.values[:10])

print("\nPredicted:")
print(y_pred_cat[:10])

# =========================
# SILHOUETTE MODEL
# =========================

X = df["product_description"]
y_silhouette = df["Silhouette"]


X_train_text, X_test_text, y_train_sil, y_test_sil = train_test_split(
    X,
    y_silhouette,
    test_size=0.2,
    random_state=42
)


sil_vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    lowercase=True
)


X_train_vec_sil = sil_vectorizer.fit_transform(X_train_text)

X_test_vec_sil = sil_vectorizer.transform(X_test_text)


sil_model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced"
)


sil_model.fit(
    X_train_vec_sil,
    y_train_sil
)


sil_pred = sil_model.predict(
    X_test_vec_sil
)


print("Silhouette Accuracy:",
      accuracy_score(y_test_sil, sil_pred))


print("Silhouette F1:",
      f1_score(y_test_sil, sil_pred, average="weighted"))

# =========================
# LENGTH MODEL
# =========================

X = df["product_description"]
y_length = df["Length"]


X_train_text, X_test_text, y_train_len, y_test_len = train_test_split(
    X,
    y_length,
    test_size=0.2,
    random_state=42
)


len_vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    lowercase=True
)

X_train_vec_len = len_vectorizer.fit_transform(X_train_text)
X_test_vec_len = len_vectorizer.transform(X_test_text)


len_model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced"
)

len_model.fit(X_train_vec_len, y_train_len)


len_pred = len_model.predict(X_test_vec_len)


print("Length Accuracy:", accuracy_score(y_test_len, len_pred))
print("Length F1:", f1_score(y_test_len, len_pred, average="weighted"))

# =========================
# EMBELLISHMENT MODEL
# =========================

X = df["product_description"]
y_emb = df["Embellishment"]


X_train_text, X_test_text, y_train_emb, y_test_emb = train_test_split(
    X,
    y_emb,
    test_size=0.2,
    random_state=42
)


emb_vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    lowercase=True
)


X_train_vec_emb = emb_vectorizer.fit_transform(X_train_text)
X_test_vec_emb = emb_vectorizer.transform(X_test_text)


emb_model = LogisticRegression(
    max_iter=4000,
    class_weight="balanced"
)


emb_model.fit(X_train_vec_emb, y_train_emb)


emb_pred = emb_model.predict(X_test_vec_emb)


print("Embellishment Accuracy:", accuracy_score(y_test_emb, emb_pred))
print("Embellishment F1:", f1_score(y_test_emb, emb_pred, average="weighted"))
import joblib

joblib.dump(fabric_model, "fabric.pkl")
joblib.dump(neck_model, "neck.pkl")
joblib.dump(sleeve_model, "sleeve.pkl")
joblib.dump(color_model, "color.pkl")
joblib.dump(cat_model, "category.pkl")
joblib.dump(sil_model, "silhouette.pkl")
joblib.dump(len_model, "length.pkl")
joblib.dump(emb_model, "embellishment.pkl")

joblib.dump(vectorizer, "vectorizer.pkl")