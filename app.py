import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(page_title="Decision Tree Classifier", page_icon="🌳", layout="wide")
st.title("Decision Tree Classification System")
st.write("Upload a CSV dataset, choose the target column, and compare ID3 and C4.5-style decision trees.")

# ---------- Decision tree implementation ----------
def entropy(y):
    counts = pd.Series(y).value_counts(normalize=True)
    return float(-(counts * np.log2(counts + 1e-12)).sum())

def split_score(data, feature, target, method):
    parent_entropy = entropy(data[target])
    weighted_entropy = 0.0
    split_info = 0.0
    for _, group in data.groupby(feature, dropna=False):
        p = len(group) / len(data)
        weighted_entropy += p * entropy(group[target])
        split_info -= p * np.log2(p + 1e-12)
    gain = parent_entropy - weighted_entropy
    if method == "ID3":
        return gain
    return gain / split_info if split_info > 0 else 0.0

def majority_class(y):
    return pd.Series(y).mode().iloc[0]

def build_tree(data, features, target, method, depth=0, max_depth=8):
    labels = data[target]
    majority = majority_class(labels)
    if labels.nunique() == 1:
        return {"leaf": labels.iloc[0], "majority": majority}
    if not features or depth >= max_depth or len(data) < 2:
        return {"leaf": majority, "majority": majority}
    scores = {f: split_score(data, f, target, method) for f in features}
    best = max(scores, key=scores.get)
    # Stop when no useful split remains.
    if scores[best] <= 1e-10:
        return {"leaf": majority, "majority": majority}
    node = {"feature": best, "majority": majority, "children": {}}
    remaining = [f for f in features if f != best]
    for value, group in data.groupby(best, dropna=False):
        node["children"][str(value)] = build_tree(group.drop(columns=[best]), remaining, target, method, depth+1, max_depth)
    return node

def predict_one(tree, row):
    while "leaf" not in tree:
        feature = tree["feature"]
        value = str(row.get(feature, ""))
        if value not in tree["children"]:
            return tree["majority"]
        tree = tree["children"][value]
    return tree["leaf"]

def tree_to_dot(tree, name):
    lines = ["digraph Tree {", 'graph [rankdir=TB];', 'node [shape=box, style="rounded,filled", fillcolor="#EAF2F8", color="#496A81", fontname="Arial"];', 'edge [fontname="Arial"];']
    counter = [0]
    def walk(node):
        current = str(counter[0]); counter[0] += 1
        if "leaf" in node:
            label = f"Predict: {node['leaf']}"
            lines.append(f'{current} [label="{label}", fillcolor="#E8F5E9"];')
        else:
            lines.append(f'{current} [label="{node["feature"]}"];')
            for val, child in node["children"].items():
                child_id = walk(child)
                safe_val = val.replace('"', "'")
                lines.append(f'{current} -> {child_id} [label="{safe_val}"];')
        return current
    walk(tree)
    lines.append("}")
    return "\n".join(lines)

# ---------- Sidebar and data ----------
uploaded = st.file_uploader("Upload classification dataset (CSV)", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV file to begin. You can try the sample dataset included with this project.")
    st.stop()

try:
    raw = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()

if raw.empty or len(raw.columns) < 2:
    st.error("The CSV must contain at least two columns and one data row.")
    st.stop()

st.subheader("1. Dataset preview")
st.write(f"Rows: {raw.shape[0]} | Columns: {raw.shape[1]}")
st.dataframe(raw.head(10), use_container_width=True)

target = st.selectbox("Select target attribute (the column to predict)", raw.columns, index=len(raw.columns)-1)
test_size = st.slider("Test data percentage", min_value=10, max_value=40, value=20, step=5)
max_depth = st.slider("Maximum tree depth", min_value=2, max_value=12, value=8)

if st.button("Train and evaluate models", type="primary"):
    data = raw.copy()
    data = data.dropna(subset=[target])
    if data[target].nunique() < 2:
        st.error("Target column must contain at least two classes.")
        st.stop()

    # Fill missing values and convert all features to categorical values.
    X = data.drop(columns=[target]).copy()
    y = data[target].astype(str)
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].median())
            # Bin numeric columns so both algorithms can use categorical splits.
            unique_count = X[col].nunique()
            bins = min(3, unique_count)
            if bins >= 2:
                try:
                    X[col] = pd.qcut(X[col], q=bins, duplicates="drop").astype(str)
                except Exception:
                    X[col] = X[col].astype(str)
            else:
                X[col] = X[col].astype(str)
        else:
            X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else "Unknown").astype(str)
    X = X.astype(str)

    if X.shape[1] == 0:
        st.error("No feature columns available.")
        st.stop()

    # Stratify when every class has enough samples; otherwise use a regular split.
    stratify = y if y.value_counts().min() >= 2 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=42, stratify=stratify
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=42
        )

    train_data = X_train.copy()
    train_data[target] = y_train.values
    features = list(X.columns)

    results = {}
    trees = {}
    for algorithm in ["ID3", "C4.5"]:
        tree = build_tree(train_data, features.copy(), target, algorithm, max_depth=max_depth)
        predictions = [predict_one(tree, row) for _, row in X_test.iterrows()]
        results[algorithm] = {
            "predictions": predictions,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
            "recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
            "f1": f1_score(y_test, predictions, average="weighted", zero_division=0),
            "cm": confusion_matrix(y_test, predictions, labels=sorted(y.unique()))
        }
        trees[algorithm] = tree

    st.subheader("2. Model performance comparison")
    comparison = pd.DataFrame({
        "Metric": ["Accuracy", "Precision (weighted)", "Recall (weighted)", "F1-score (weighted)"],
        "ID3": [results["ID3"]["accuracy"], results["ID3"]["precision"], results["ID3"]["recall"], results["ID3"]["f1"]],
        "C4.5": [results["C4.5"]["accuracy"], results["C4.5"]["precision"], results["C4.5"]["recall"], results["C4.5"]["f1"]]
    })
    st.dataframe(comparison.style.format({"ID3": "{:.4f}", "C4.5": "{:.4f}"}), use_container_width=True)
    best = max(results, key=lambda a: results[a]["f1"])
    st.success(f"Best model by weighted F1-score: {best}")

    st.subheader("3. Confusion matrices")
    c1, c2 = st.columns(2)
    labels = sorted(y.unique())
    for col, algorithm in [(c1, "ID3"), (c2, "C4.5")]:
        with col:
            st.markdown(f"**{algorithm}**")
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=results[algorithm]["cm"], display_labels=labels)
            disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
            st.pyplot(fig)
            plt.close(fig)

    st.subheader("4. Constructed decision trees")
    c1, c2 = st.columns(2)
    for col, algorithm in [(c1, "ID3"), (c2, "C4.5")]:
        with col:
            st.markdown(f"**{algorithm} tree**")
            st.graphviz_chart(tree_to_dot(trees[algorithm], algorithm), use_container_width=True)

    st.subheader("5. Predictions on test data")
    prediction_table = X_test.copy()
    prediction_table["Actual"] = y_test.values
    prediction_table["ID3 Prediction"] = results["ID3"]["predictions"]
    prediction_table["C4.5 Prediction"] = results["C4.5"]["predictions"]
    st.dataframe(prediction_table, use_container_width=True)
    st.download_button(
        "Download predictions as CSV",
        prediction_table.to_csv(index=False).encode("utf-8"),
        file_name="decision_tree_predictions.csv",
        mime="text/csv"
    )

    st.caption("Implementation note: ID3 selects splits using Information Gain. C4.5-style selects splits using Gain Ratio. Numeric features are converted into up to three quantile-based categories before training. This is an educational categorical implementation, not a full production C4.5 implementation with continuous-threshold pruning.")
