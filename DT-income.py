import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "decision_trees"

def image_path(fig_id):
    return os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id)

def save_fig(fig_id, tight_layout=True):
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(image_path(fig_id) + ".png", format='png', dpi=300)


dirt = "../../ML/D3/"
adult = pd.read_csv(dirt + 'adult.csv')

adult.head()

adult.describe()


adult.info()


adult.dtypes

adult.dropna(inplace=True)
adult['workclass']=adult['workclass'].astype('category')
adult['education']=adult['education'].astype('category')
adult['marital-status']=adult['marital-status'].astype('category')
adult['occupation']=adult['occupation'].astype('category')
adult['relationship']=adult['relationship'].astype('category')
adult['race']=adult['race'].astype('category')
adult['sex']=adult['sex'].astype('category')
adult['native-country']=adult['native-country'].astype('category')
adult['income']=adult['income'].astype('category')

adult.dtypes

adult.describe()


adult.shape
Xcols=['workclass','education','marital-status','occupation','relationship','race','sex','native-country','age',
      'fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
Ycol = ['income']
Xcols=adult[Xcols]
Xcols=pd.get_dummies(Xcols)
Xcols.dtypes
Y=adult[Ycol]

X = pd.concat([Xcols], axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

feature_names=X.columns
print (feature_names)


from sklearn import metrics
adult_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4, max_leaf_nodes=5)
adult_tree.fit(X_train, Y_train)
predictions = adult_tree.predict(X_test)
print(metrics.confusion_matrix(Y_test, predictions))
print("Accuracy Score =", metrics.accuracy_score(Y_test, predictions))

###### 10-fold cross-validation ###################
from sklearn.model_selection import cross_val_score
cross_val_score(adult_tree, X_train, Y_train, cv=5, scoring = "accuracy")

from sklearn.tree import export_graphviz

## Decision Tree Visualization
export_graphviz(
   adult_tree,
   out_file=dirt+"tree_income.dot",
   feature_names=feature_names,
   class_names=('0','1','2','3','4','5'),
   rounded=True,
   filled=True)

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
importances = adult_tree.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances 
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

