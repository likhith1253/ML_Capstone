"""Data loading, model registry and caching for the capstone dashboard.

Every pipeline mirrors the corresponding notebook (same cleaning, split and seeds).
New models are added by putting one line in CLASSIFIERS or REGRESSORS.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, calinski_harabasz_score, davies_bouldin_score, f1_score,
                             mean_absolute_error, mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = Path(__file__).resolve().parent / ".cache"
CACHE.mkdir(exist_ok=True)
SEED = 42


def cached(name, build):
    """Build once, keep on disk so later launches are instant."""
    path = CACHE / f"{name}.joblib"
    if path.exists():
        return joblib.load(path)
    out = build()
    joblib.dump(out, path)
    return out


# ------------------------------------------------------------------ classification
# name -> (factory, needs scaled features)
CLASSIFIERS = {
    "Logistic Regression": (lambda: LogisticRegression(C=10, max_iter=1000, random_state=SEED), True),
    "K-Nearest Neighbors": (lambda: KNeighborsClassifier(n_neighbors=9), True),
    "Decision Tree": (lambda: DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=SEED), False),
    "Naive Bayes": (lambda: GaussianNB(), False),
    "SVM (RBF)": (lambda: SVC(C=10, kernel="rbf", probability=True, random_state=SEED), True),
    "Random Forest": (lambda: RandomForestClassifier(n_estimators=200, max_depth=16, min_samples_split=2,
                                                     random_state=SEED, n_jobs=-1), False),
    "AdaBoost": (lambda: AdaBoostClassifier(n_estimators=150, learning_rate=0.5, algorithm="SAMME", random_state=SEED), False),
    "Gradient Boosting": (lambda: GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=4,
                                                             random_state=SEED), False),
}
PENDING_CLASSIFIERS = ["Bagging Classifier", "MLP Classifier"]


def load_beans():
    df = pd.read_csv(DATA / "Dry_Bean_Dataset.csv").drop_duplicates()
    cols = df.select_dtypes("number").columns

    def trim(g):
        q1, q3 = g[cols].quantile(0.25), g[cols].quantile(0.75)
        iqr = q3 - q1
        keep = ~((g[cols] < q1 - 1.5 * iqr) | (g[cols] > q3 + 1.5 * iqr)).any(axis=1)
        return g[keep]

    df = pd.concat([trim(g) for _, g in df.groupby("Class")], ignore_index=True)
    df["Area_Perimeter_Ratio"] = df["Area"] / df["Perimeter"]
    X = df.drop(columns="Class")
    le = LabelEncoder()
    y = le.fit_transform(df["Class"])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    return dict(df=df, X=X, X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te, scaler=scaler, classes=list(le.classes_))


def _train_classifiers():
    d = load_beans()
    Xs_tr, Xs_te = d["scaler"].transform(d["X_tr"]), d["scaler"].transform(d["X_te"])
    models, preds, probs, rows = {}, {}, {}, []
    for name, (make, scaled) in CLASSIFIERS.items():
        t0 = time.time()
        m = make().fit(Xs_tr if scaled else d["X_tr"], d["y_tr"])
        X_te = Xs_te if scaled else d["X_te"]
        p, pr = m.predict(X_te), m.predict_proba(X_te)
        y = d["y_te"]
        rows.append({"Model": name, "Accuracy": accuracy_score(y, p),
                     "Precision": precision_score(y, p, average="weighted", zero_division=0),
                     "Recall": recall_score(y, p, average="weighted", zero_division=0),
                     "F1 (weighted)": f1_score(y, p, average="weighted"),
                     "F1 (macro)": f1_score(y, p, average="macro"),
                     "ROC-AUC": roc_auc_score(y, pr, multi_class="ovr", average="weighted"),
                     "Fit time (s)": time.time() - t0})
        models[name], preds[name], probs[name] = m, p, pr
    return dict(models=models, preds=preds, probs=probs, metrics=pd.DataFrame(rows))


def classification_bundle():
    return cached("classification", _train_classifiers)


# ------------------------------------------------------------------ regression
LEAK_PREFIX = ("active.", "idle.")
RATE_COLS = ["fwd_pkts_per_sec", "bwd_pkts_per_sec", "flow_pkts_per_sec", "payload_bytes_per_second"]
POLY_FEATURES = ["fwd_pkts_tot", "bwd_pkts_tot", "fwd_header_size_tot", "bwd_header_size_tot"]


def load_flows():
    df = pd.read_csv(DATA / "RT_IOT2022_cleaned.csv")
    split = json.load(open(ROOT / "regression" / "regression_split_indices.json", encoding="utf-8"))
    drop = [c for c in df.columns if "iat" in c.lower() or c.startswith(LEAK_PREFIX)]
    drop += [c for c in RATE_COLS if c in df.columns] + ["Attack_type"]
    d = df.drop(columns=drop)
    d["total_packets"] = d["fwd_pkts_tot"] + d["bwd_pkts_tot"]
    X, y = d.drop(columns="flow_duration"), d["flow_duration"]
    tr, te = split["train_indices"], split["test_indices"]
    return dict(X_tr=X.loc[tr], X_te=X.loc[te], y_tr=y.loc[tr], y_te=y.loc[te])


def _regressors(cat, num):
    ohe = lambda: OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    scaled = lambda: ColumnTransformer([("cat", ohe(), cat), ("num", StandardScaler(), num)])
    trees = lambda: ColumnTransformer([("cat", ohe(), cat), ("num", "passthrough", num)])
    poly = lambda: ColumnTransformer([("cat", ohe(), cat), ("num", Pipeline([
        ("s", StandardScaler()), ("p", PolynomialFeatures(degree=2, include_bias=False))]), POLY_FEATURES)])
    # name -> (preprocessor factory, estimator factory)
    return {
        "Linear Regression": (scaled, lambda: LinearRegression()),
        "Ridge": (scaled, lambda: Ridge(alpha=0.1, random_state=SEED)),
        "Lasso": (scaled, lambda: Lasso(alpha=1.0, max_iter=2500, random_state=SEED)),
        "ElasticNet": (scaled, lambda: ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=2500, random_state=SEED)),
        "Polynomial (deg 2)": (poly, lambda: LinearRegression()),
        "Decision Tree": (trees, lambda: DecisionTreeRegressor(random_state=SEED)),
        "Random Forest": (trees, lambda: RandomForestRegressor(n_estimators=100, max_depth=25, random_state=SEED, n_jobs=-1)),
        "Gradient Boosting": (trees, lambda: GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=SEED)),
        "SVR (linear)": (scaled, lambda: SVR(kernel="linear", C=1.0, max_iter=2500)),
        "KNN (k=5)": (scaled, lambda: KNeighborsRegressor(n_neighbors=5, n_jobs=-1)),
    }


def _train_regressors():
    d = load_flows()
    cat = d["X_tr"].select_dtypes("object").columns.tolist()
    num = d["X_tr"].select_dtypes("number").columns.tolist()
    models, preds, rows = {}, {}, []
    for name, (prep, est) in _regressors(cat, num).items():
        t0 = time.time()
        pipe = Pipeline([("prep", prep()), ("reg", est())]).fit(d["X_tr"], d["y_tr"])
        p = pipe.predict(d["X_te"])
        rows.append({"Model": name, "R2": r2_score(d["y_te"], p),
                     "RMSE": float(np.sqrt(mean_squared_error(d["y_te"], p))),
                     "MAE": mean_absolute_error(d["y_te"], p), "Fit time (s)": time.time() - t0})
        models[name], preds[name] = pipe, p
    names = pipe.named_steps["prep"].get_feature_names_out()
    importances = {}
    for n in ("Decision Tree", "Random Forest", "Gradient Boosting"):
        feats = [s.split("__", 1)[1] for s in models[n].named_steps["prep"].get_feature_names_out()]
        importances[n] = pd.Series(models[n].named_steps["reg"].feature_importances_, index=feats)
    template = {c: d["X_tr"][c].median() for c in num}
    template.update({c: d["X_tr"][c].mode()[0] for c in cat})
    return dict(models=models, preds=preds, metrics=pd.DataFrame(rows), importances=importances,
                template=template, cat=cat, num=num,
                cat_values={c: sorted(d["X_tr"][c].unique()) for c in cat},
                ranges={c: (float(d["X_tr"][c].min()), float(d["X_tr"][c].quantile(0.99)), float(d["X_tr"][c].max())) for c in num})


def regression_bundle():
    return cached("regression", _train_regressors)


def regression_data():
    return load_flows()


# ------------------------------------------------------------------ clustering
HOTEL_NUM = ["lead_time", "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies",
             "is_repeated_guest", "previous_cancellations", "previous_bookings_not_canceled", "booking_changes",
             "days_in_waiting_list", "adr", "required_car_parking_spaces", "total_of_special_requests"]
HOTEL_CAT = ["hotel", "meal", "market_segment", "distribution_channel", "deposit_type", "customer_type",
             "reserved_room_type"]
HOTEL_LOG = ["lead_time", "days_in_waiting_list", "previous_cancellations", "previous_bookings_not_canceled",
             "booking_changes", "adr"]


def _hotel_matrix(frame, fit=None):
    X = frame[HOTEL_NUM + HOTEL_CAT].copy()
    X["children"] = X["children"].fillna(0)
    for c in HOTEL_LOG:
        X[c] = np.log1p(X[c])
    if fit is None:
        fit = ColumnTransformer([("num", StandardScaler(), HOTEL_NUM),
                                 ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), HOTEL_CAT)]).fit(X)
    return fit, fit.transform(X)


def _build_clustering():
    df = pd.read_csv(DATA / "hotel_bookings.csv").drop_duplicates()
    df = df[(df["adults"] + df["children"].fillna(0) + df["babies"]) > 0]
    df = df[df["adr"] >= 0]
    sample = df.sample(n=15000, random_state=SEED).reset_index(drop=True)
    prep, Z = _hotel_matrix(sample)
    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]

    sweep, km_labels = [], {}
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
        km_labels[k] = km.labels_
        sweep.append({"k": k, "Inertia": km.inertia_, "Silhouette": silhouette_score(Z, km.labels_),
                      "Davies-Bouldin": davies_bouldin_score(Z, km.labels_),
                      "Calinski-Harabasz": calinski_harabasz_score(Z, km.labels_)})
    km4 = KMeans(n_clusters=4, n_init=10, random_state=SEED).fit(Z)
    agg = AgglomerativeClustering(n_clusters=4, linkage="ward").fit_predict(Z)

    pca = PCA(n_components=2, random_state=SEED).fit(Z)
    coords = pca.transform(Z)
    rs = np.random.RandomState(SEED)
    tsne_idx = rs.choice(len(Z), size=5000, replace=False)
    tsne = TSNE(n_components=2, perplexity=30, random_state=SEED).fit_transform(Z[tsne_idx])
    dendro_idx = rs.choice(len(Z), size=1000, replace=False)
    return dict(sample=sample, prep=prep, Z=Z, names=names, sweep=pd.DataFrame(sweep), km_labels=km_labels,
                km4=km4, agg=agg, coords=coords, var=pca.explained_variance_ratio_, tsne=tsne, tsne_idx=tsne_idx,
                linkage=linkage(Z[dendro_idx], method="ward", metric="euclidean"),
                metrics={"K-Means": _scores(Z, km4.labels_), "Agglomerative": _scores(Z, agg)})


def _scores(Z, labels):
    return {"Silhouette": silhouette_score(Z, labels), "Davies-Bouldin": davies_bouldin_score(Z, labels),
            "Calinski-Harabasz": calinski_harabasz_score(Z, labels)}


def clustering_bundle():
    return cached("clustering", _build_clustering)


def assign_booking(bundle, row):
    """Nearest K-Means (k=4) cluster for one hypothetical booking."""
    frame = pd.DataFrame([row])
    _, z = _hotel_matrix(frame, fit=bundle["prep"])
    return int(bundle["km4"].predict(z)[0])
