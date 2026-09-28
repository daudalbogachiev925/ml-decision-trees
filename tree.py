"""Деревья решений с визуализацией и подбором гиперпараметров."""
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# === 1. Данные ===
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 2. Базовое дерево ===
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
print("Базовая точность:", accuracy_score(y_test, clf.predict(X_test)))
print(export_text(clf, feature_names=load_iris().feature_names))

# === 3. Подбор гиперпараметров ===
param_grid = {
    'max_depth': [2, 3, 4, 5, None],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}
grid = GridSearchCV(DecisionTreeClassifier(random_state=42),
                    param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print("Лучшие параметры:", grid.best_params_)
print("Лучшая точность CV:", grid.best_score_)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

# === 4. Важность признаков ===
importances = best_model.feature_importances_
features = load_iris().feature_names
for f, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
    print(f"{f}: {imp:.4f}")
