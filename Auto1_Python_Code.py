
# coding: utf-8

# In[168]:


# Loading Libraries 

import numpy as np
import pandas as pd
import scipy 
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sns


# In[169]:


# Laod the dataset name Auto1
Auto1 = pd.read_csv("C:/Users/HP/Downloads/Auto1-DS-TestData.csv")


# In[170]:


# observations of dimensions of the dataset rows & columns
Auto1.shape


# In[171]:


# observations of columns 
Auto1.columns


# In[172]:


# observations of Daata frames
Auto1.info()


# In[173]:


# identifying the Duplicates if any
Auto1.duplicated(subset=None, keep='first')


# In[174]:


# imputation and replacing the missing values
from sklearn.preprocessing import Imputer
Auto1 = Auto1.replace('?', 'NaN')
imp = Imputer(missing_values='NaN', strategy='mean' )
Auto1[['normalized-losses','bore','stroke','horsepower','peak-rpm','price']] = imp.fit_transform(Auto1[['normalized-losses','bore','stroke','horsepower','peak-rpm','price']])
Auto1.head()


# In[175]:


Auto1.make.value_counts().nlargest(10).plot(kind='bar', figsize=(15,5))
plt.title("No of vehicles by Brand")
plt.ylabel('No of vehicles')
plt.xlabel('Brand');


# In[176]:


Auto1['fuel-type'].value_counts().plot(kind='bar',color='green')
plt.title("Frequence Vs Type of Fuel")
plt.ylabel('No of vehicles')
plt.xlabel('Type of fuel');


# In[177]:


# Histogram Figures 

get_ipython().run_line_magic('matplotlib', 'inline')
Auto1.hist(bins=30, figsize=(30,15))
plt.savefig("Histogram Figures")
plt.show()


# In[178]:


# observation of string data missing
Auto1.groupby('num-of-doors').size()


# In[179]:


# replacing the missing values for variable num-of-doors
Auto1 = Auto1.replace("NaN","four")
Auto1.head(35) #for 29 row/observation num-of-doors data is missing so considered till 35 observations


# In[180]:


# check of imputation and repacing for string variable 
Auto1.groupby('num-of-doors').size()


# In[181]:


from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelencoder = LabelEncoder()
for i in ['make','fuel-type','aspiration', 'num-of-doors','body-style','drive-wheels','engine-location','engine-type','num-of-cylinders','fuel-system']:
    Auto1[i] = labelencoder.fit_transform(Auto1[i])
Auto1.head()


# In[147]:


# Descriptive Statistics (Basic understanding of significant variables 
Auto1.describe().round(3)


# In[182]:


# correlation coefficient

corr = Auto1.corr()
cmap = cmap=sns.diverging_palette(5, 250, as_cmap=True)

def magnify():
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])
            ]

corr.style.background_gradient(cmap, axis=1)    .set_properties(**{'max-width': '80px', 'font-size': '9pt'})    .set_caption("CORRELATION RELATIONSHIP BETWEEN VARIABLES")    .set_precision(4)    .set_table_styles(magnify())


# In[183]:


# Bloxplots
sns.jointplot(data=Auto1, x='price', y='engine-size', kind='reg', color='g')
plt.show()


# In[184]:


g = sns.lmplot('normalized-losses',"symboling", Auto1);


# In[185]:


g = sns.lmplot('price',"normalized-losses", Auto1);


# In[186]:


plt.rcParams['figure.figsize']=(10,5)
ax = sns.boxplot(x="drive-wheels", y="price", data=Auto1)


# In[187]:


plt.rcParams['figure.figsize']=(25,10)
ax = sns.boxplot(x="make", y="price", data=Auto1)


# In[188]:


from sklearn.preprocessing import Imputer
Auto1 = Auto1.replace('?', 'NaN')
imp = Imputer(missing_values='NaN', strategy='mean' )
Auto1[['normalized-losses','bore','stroke','horsepower','peak-rpm','price']] = imp.fit_transform(Auto1[['normalized-losses','bore','stroke','horsepower','peak-rpm','price']])
Auto1.head()


# In[189]:


# Spliting the data into train and test dataset in 80% and 20% respectively

import sklearn
from sklearn import model_selection
Y = Auto1['price']
X = Auto1.drop('price',axis =1)

x_train, x_test, y_train,  y_test = sklearn.model_selection.train_test_split(X, Y,train_size=0.8, test_size=0.2, random_state=0)


# In[190]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
lm_1 = regressor.fit(x_train, y_train)


# In[191]:


lm_1.score(x_train,y_train)


# In[192]:


lm_1.score(x_test,y_test)


# In[193]:


Auto1 = Auto1.copy()
names = []
for name in Auto1.columns:
    names.append(name.replace('-', '_'))

Auto1.columns = names


# In[194]:


import statsmodels.formula.api as smf

lm0 = smf.ols(formula= 'price ~ symboling + normalized_losses + wheel_base +  width + height + length + + curb_weight + engine_size + stroke + compression_ratio + peak_rpm + city_mpg + highway_mpg + bore + horsepower' , data =Auto1).fit()

# 
print(lm0.summary())


# In[195]:


import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[196]:


# Split-out validation dataset
array = Auto1.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
seed = 7
scoring = 'accuracy'


# In[197]:


models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[198]:


# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# In[199]:


# Make predictions on validation dataset 
LDA = LinearDiscriminantAnalysis()
LDA.fit(X_train, Y_train)
predictions = LDA.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# In[200]:


NB = GaussianNB()
NB.fit(X_train, Y_train)
predictions = NB.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# In[201]:


CART = DecisionTreeClassifier()
CART.fit(X_train, Y_train)
predictions = CART.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# In[202]:


# Thank you

