This C++ with DSA is completed with the help of College Wallah, I have used their free resources which is available in Youtube where they
have taught entire c++ with dsa from basic to advanced, I have uploaded all the code in every lecture they have taught.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn import datasets
import matplotlib.pyplot as plt

# ---------------------------
# 1. Play Tennis Dataset
# ---------------------------
data = {
    'outlook': ['sunny','sunny','overcast','rainy','rainy','rainy','overcast','sunny','sunny',
                'rainy','sunny','overcast','overcast','rainy','rainy','overcast','overcast','sunny','overcast','overcast'],
    'temperature': ['hot','hot','hot','mild','cool','cool','cool','mild','cool',
                    'mild','mild','mild','hot','mild','mild','hot','hot','mild','mild','mild'],
    'humidity': ['high','high','high','high','normal','normal','normal','high','normal',
                 'normal','normal','high','normal','high','high','high','high','high','high','high'],
    'windy': [False, True, False, False, False, True, True, False, False,
              False, True, True, False, True, True, False, False, False, False, False],
    'play': ['no','no','yes','yes','yes','no','yes','no','yes',
             'yes','yes','yes','yes','no','no','no','yes','yes','no','yes']
}

df = pd.DataFrame(data)

# Encode categorical features
df_encoded = pd.get_dummies(df.drop('play', axis=1))
y = df['play']

# ---------- CASE A: 75/25 ----------
X_train, X_test, y_train, y_test = train_test_split(df_encoded, y, test_size=0.25, random_state=42)
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Play Tennis Dataset (75/25 split) Accuracy:", accuracy_score(y_test, y_pred))

plt.figure(figsize=(10,6))
plot_tree(clf, feature_names=df_encoded.columns, class_names=clf.classes_, filled=True)
plt.show()

# ---------- CASE B: 66.6/33.3 ----------
X_train, X_test, y_train, y_test = train_test_split(df_encoded, y, test_size=0.333, random_state=42)
clf2 = DecisionTreeClassifier(criterion='entropy')
clf2.fit(X_train, y_train)
y_pred2 = clf2.predict(X_test)
print("Play Tennis Dataset (66.6/33.3 split) Accuracy:", accuracy_score(y_test, y_pred2))

plt.figure(figsize=(10,6))
plot_tree(clf2, feature_names=df_encoded.columns, class_names=clf2.classes_, filled=True)
plt.show()


# ---------------------------
# 2. IRIS Dataset
# ---------------------------
iris = datasets.load_iris()
X, y = iris.data, iris.target

# ---------- CASE A: 75/25 ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
clf3 = DecisionTreeClassifier(criterion='entropy')
clf3.fit(X_train, y_train)
y_pred3 = clf3.predict(X_test)
print("Iris Dataset (75/25 split) Accuracy:", accuracy_score(y_test, y_pred3))

plt.figure(figsize=(12,8))
plot_tree(clf3, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()

# ---------- CASE B: 66.6/33.3 ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333, random_state=42)
clf4 = DecisionTreeClassifier(criterion='entropy')
clf4.fit(X_train, y_train)
y_pred4 = clf4.predict(X_test)
print("Iris Dataset (66.6/33.3 split) Accuracy:", accuracy_score(y_test, y_pred4))

plt.figure(figsize=(12,8))
plot_tree(clf4, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()