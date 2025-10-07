from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# 1. Load text data (we’ll simulate spam/ham using newsgroups data)
categories = ['sci.electronics', 'talk.politics.misc']
data = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

X = data.data
y = data.target  # 0 = sci.electronics (ham), 1 = talk.politics.misc (spam-like)

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])

# 4. Hyperparameter grid
param_grid = {
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'tfidf__max_df': [0.75, 0.9],
    'clf__C': [0.01, 0.1, 1, 10],  # Regularization strength (lower = stronger reg.)
    'clf__penalty': ['l2']
}

# 5. Grid Search
grid = GridSearchCV(pipeline, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)

# 6. Evaluate
y_pred = grid.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()

# 7. Show table output
print("Best Parameters:")
print(grid.best_params_)

print("\nPerformance Report (on test data):")
print(df_report.round(3))
