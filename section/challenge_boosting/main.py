from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# 1. Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Train AdaBoost
ada_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=0.8,
    random_state=42
)
ada_model.fit(X_train, y_train)

# 4. Train Gradient Boosting
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb_model.fit(X_train, y_train)

# 5. Evaluate models
ada_pred = ada_model.predict(X_test)
gb_pred = gb_model.predict(X_test)

ada_acc = accuracy_score(y_test, ada_pred)
gb_acc = accuracy_score(y_test, gb_pred)

print(f"AdaBoost Accuracy: {ada_acc:.3f}")
print(f"Gradient Boosting Accuracy: {gb_acc:.3f}")