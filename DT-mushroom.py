
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report


# In[2]:


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


# In[11]:


dirt = "C:/Users/Xing/Desktop/ML/D4/"
mush = pd.read_csv(dirt + 'mush.csv')


# In[12]:


mush.head()


# In[14]:


mush.describe()


# In[15]:


mush.info()


# In[16]:


mush.dtypes


# In[21]:


mush.dropna(inplace=True)
mush['edible']=mush['edible'].astype('category')
mush['cap-shape']=mush['cap-shape'].astype('category')
mush['cap-surface']=mush['cap-surface'].astype('category')
mush['cap-color']=mush['cap-color'].astype('category')
mush['bruises']=mush['bruises'].astype('category')
mush['odor']=mush['odor'].astype('category')

mush['gill-attachment']=mush['gill-attachment'].astype('category')
mush['gill-spacing']=mush['gill-spacing'].astype('category')
mush['gill-size']=mush['gill-size'].astype('category')
mush['gill-color']=mush['gill-color'].astype('category')
mush['stalk-shape']=mush['stalk-shape'].astype('category')
mush['stalk-root']=mush['stalk-root'].astype('category')
mush['stalk-surface-above-ring']=mush['stalk-surface-above-ring'].astype('category')
mush['stalk-surface-below-ring']=mush['stalk-surface-below-ring'].astype('category')
mush['stalk-color-above-ring']=mush['stalk-color-above-ring'].astype('category')
mush['stalk-color-below-ring']=mush['stalk-color-below-ring'].astype('category')
mush['veil-type']=mush['veil-type'].astype('category')
mush['veil-color']=mush['veil-color'].astype('category')
mush['ring-number']=mush['ring-number'].astype('category')

mush['ring-type']=mush['ring-type'].astype('category')
mush['spore-print-color']=mush['spore-print-color'].astype('category')
mush['population']=mush['population'].astype('category')
mush['habitat']=mush['habitat'].astype('category')


# In[22]:


mush.dtypes


# In[25]:


mush.describe()


# In[26]:


mush.shape


# In[27]:


Xcols=['cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color',
       'stalk-shape','stalk-root','stalk-surface-above-ring',
       'stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number',
       'ring-type','spore-print-color','population','habitat']
Ycol = ['edible']
Xcols=mush[Xcols]
Xcols=pd.get_dummies(Xcols)
Xcols.dtypes


# In[28]:


Y=mush[Ycol]


# In[29]:


X = pd.concat([Xcols], axis=1)


# In[30]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y)


# In[31]:


feature_names=X.columns
print (feature_names)


# In[51]:


from sklearn import metrics
mush_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=6, max_leaf_nodes=9)
mush_tree.fit(X_train, Y_train)
predictions = mush_tree.predict(X_test)
print(metrics.confusion_matrix(Y_test, predictions))
print("Accuracy Score =", metrics.accuracy_score(Y_test, predictions))


# In[52]:


###### 10-fold cross-validation ###################
from sklearn.model_selection import cross_val_score
cross_val_score(mush_tree, X_train, Y_train, cv=5, scoring = "accuracy")


# In[53]:


from sklearn.tree import export_graphviz

## Decision Tree Visualization
export_graphviz(
   mush_tree,
   out_file=dirt+"tree_mush.dot",
   feature_names=feature_names,
   class_names=('0','1'),
   rounded=True,
   filled=True)


# In[58]:


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

