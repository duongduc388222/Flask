from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle
iris = load_iris()
X,y = iris.data, iris.target

rf_classifier = RandomForestClassifier()
rf_classifier.fit(X,y)

with open("model.pkl", "wb") as f:
    pickle.dump(rf_classifier, f)