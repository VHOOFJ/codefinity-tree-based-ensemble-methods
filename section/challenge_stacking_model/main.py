from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score

# 1. Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Define base models
base_estimators = [
    ('tree', DecisionTreeClassifier(max_depth=3, random_state=42)),
    ('svc', SVC(probability=True, random_state=42))
]

# 4. Define final estimator
final_estimator = LogisticRegression(random_state=42)

# 5. Create and train stacking model
stack_model = StackingClassifier(
    estimators=base_estimators,
    final_estimator=final_estimator
)
stack_model.fit(X_train, y_train)

# 6. Make predictions
y_pred = stack_model.predict(X_test)

# 7. Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Stacking Model Accuracy: {accuracy:.3f}")