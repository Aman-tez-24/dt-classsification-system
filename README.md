# Decision Tree Classification System

A Streamlit application that accepts a CSV classification dataset, lets the user choose the target column, preprocesses the data, trains ID3 and C4.5-style decision trees, visualizes the trees, predicts test rows, and compares evaluation metrics.

## Features
- CSV upload and dataset preview
- Target attribute selection
- Missing-value handling
- Numeric feature binning and categorical conversion
- ID3 split selection using Information Gain
- C4.5-style split selection using Gain Ratio
- Decision-tree visualization using Graphviz
- Predictions, confusion matrices, accuracy, precision, recall, and weighted F1-score
- Downloadable prediction CSV

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## GitHub
```bash
git init
git add .
git commit -m "Decision Tree Classification System"
git branch -M main
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

## Algorithm note
This is an educational implementation. ID3 uses Information Gain. The C4.5-style implementation uses Gain Ratio. Numeric features are discretized into up to three quantile bins before training. It does not implement every detail of the original C4.5 algorithm, such as its full continuous-threshold selection and pruning procedure.
