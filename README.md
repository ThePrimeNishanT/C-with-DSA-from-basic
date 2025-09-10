This C++ with DSA is completed with the help of College Wallah, I have used their free resources which is available in Youtube where they
have taught entire c++ with dsa from basic to advanced, I have uploaded all the code in every lecture they have taught.
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load Iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Train Decision Tree
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X, y)

# Plot the tree
plt.figure(figsize=(12,8))
plot_tree(
    clf, 
    feature_names=iris.feature_names, 
    class_names=iris.target_names, 
    filled=True, 
    rounded=True
)
plt.show()