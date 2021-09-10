#!/usr/bin/env python
# coding: utf-8

# ### Predicting Wine Quality using Wine Quality Dataset
# - The aim of this project is to predict the quality of wine on a scale of 0–10 given a set of features as inputs. 
# - The dataset used is Wine Quality Data set from UCI MachineLearning Repository. 

# In[1]:


# Installing Libraries: Importing modules from packages

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score


# In[2]:


# Data Gathering: Importing the dataset 

data = pd.read_csv(r'C:\path\dataset.csv')


# In[3]:


# Data Exploration: Drawing insights from data

data.head(10)


# In[4]:


# Data Exploration: Determining the shape of dataset

data.shape


# In[5]:


# Data Exploration: Summarizing the data 

data.describe()


# In[6]:


# Data Cleaning: Checking for missing values

data.info() 


# In[7]:


# Data Cleaning: Checking for null values

data.isnull().sum()


# In[36]:


# Data pre-processing: Checking for outliers through boxplots

l = data.columns.values
number_of_columns=12
number_of_rows = len(l)-1/number_of_columns
plt.figure(figsize=(2*number_of_columns,5*number_of_rows))

for i in range(0,len(l)):
    plt.subplot(number_of_rows+10, number_of_columns-8, i+1)
    sns.boxplot(data[l[i]],color='green')
    plt.tight_layout()
    


# In[38]:


# Data Distribution: Understanding the target variable (Quality)

plt.figure(figsize=(8, 5))
sns.countplot(data["quality"], palette="mako")
plt.title('Relation in quality values')
data["quality"].value_counts()


# In[10]:


# Data Pre-Processing: Classifying category as:
 
# 3,4   -> Bad

# 5,6   -> Medium

# 7,8,9 -> Good

quality = data["quality"].values

# Creating New Column: 'Category' 

category = []    

for num in quality:
    if num<5:
        category.append("Bad")
    elif num>6:
        category.append("Good")
    else:
        category.append("Medium")

        
# Displaying top 5 values of Category 

print(category[:5])     


# In[11]:


# Data transformation: Adding 'Category' column to the dataset

category = pd.DataFrame(data=category, columns=["category"])
data = pd.concat([data,category],axis=1)
data.drop(columns="quality",axis=1)

data.head()


# In[12]:


# Analyze Data: Determining the relationship between classes in 'Category' column

plt.figure(figsize=(8,5))
sns.countplot(data["category"],palette="mako")
plt.title('Relation in category values')

# Counting the number of each class

data["category"].value_counts()


# In[13]:


# Data Exploration and Visualization: Checking the correlation of columns using heatmap

plt.subplots(figsize=(10, 8))
sns.heatmap(data.corr(), cmap=sns.diverging_palette(250, 10, as_cmap=True))


# In[14]:


# Exploratory Data Analysis: Checking the correlation of top 3 columns affecting the target variable

df = pd.DataFrame(data)
selected_columns = df[["sulphates","quality","alcohol","citric acid"]]
new_df = selected_columns.copy()

df = new_df.head(10)
df.plot(kind='bar',figsize=(14,9), colormap='viridis')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[15]:


# Bivariate Analysis: Checking the variation of alcohol in the different qualities of wine

plt.figure(figsize=(8,5))
sns.jointplot(y=data["alcohol"],x=data["quality"],kind="scatter", joint_kws={'color':'c'})


# In[16]:


# Data Pre-Processing: Classifying quality as:
 
data['quality'] = data['quality'].map({3 : 'bad', 4 :'bad', 5: 'medium', 6: 'medium', 7: 'good', 8: 'good'})


# In[17]:


# Data Cleaning: Discarding columns which do not affect the target variable

data.drop(["density","total sulfur dioxide", "volatile acidity","category"], axis = 1, inplace = True)

data.head()


# In[18]:


# Setting features, labels and Encoding the categorical data as:
# Good   ->1
# Medium ->2
# Bad    ->3

le = LabelEncoder()
data['quality'] = le.fit_transform(data['quality'])
data['quality'].value_counts


# In[19]:


# Dividing the dataset into dependent and independent variables

x = data.iloc[:,:8]
y = data.iloc[:,8]


# In[20]:


# Splitting the data into Training and Testing sets

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 42)

# Determining the shapes of training and testing sets

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[21]:


# Standard Scaling: Scaling the data for optimised predictions 

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# ## Model Training 

# ### Random Forest 

# In[22]:


# Creating the model
rfc = RandomForestClassifier(n_estimators = 250,oob_score = True)

# Feeding the training set into the model
rfc.fit(x_train, y_train)

# Predicting the results for the test set
pred_rfc = rfc.predict(x_test)

print(classification_report(y_test, pred_rfc))


# ### Decision Tree

# In[23]:


dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
pred_dtc = dtc.predict(x_test)
print(classification_report(y_test, pred_dtc))


# ### Support Vector Machine 

# In[24]:


svc = SVC()
svc.fit(x_train, y_train)
pred_svc = svc.predict(x_test)
print(classification_report(y_test, pred_svc))


# ### K-Nearest Neighbors 

# In[25]:


knn = KNeighborsClassifier()
knn.fit(x_train,y_train)
pred_knn=knn.predict(x_test)
print(classification_report(y_test, pred_knn))


# ### Logistic Regression

# In[26]:


lor = LogisticRegression()
lor.fit(x_train, y_train)
pred_lor = lor.predict(x_test)
print(classification_report(y_test, pred_lor))


# ### XGBoost

# In[27]:


xbc = xgb.XGBClassifier()
xbc.fit(x_train,y_train)
pred_xbc=xbc.predict(x_test)
print(classification_report(y_test, pred_xbc))


# ### Multi-Layer Perceptron

# In[28]:


mlp = MLPClassifier(hidden_layer_sizes = (100, 100), max_iter = 150)
mlp.fit(x_train,y_train)
pred_mlp=xbc.predict(x_test)
print(classification_report(y_test, pred_mlp))


# In[29]:


# Conclusion: Comparing the results!

conclusion = pd.DataFrame({'Model': ["Random Forest","K-Nearest Neighbors","Logistic Regression","Decision Tree","Support Vector Machine",
                                    "XGBoost","Multi-Layer Perceptron"],
                           'Accuracy': [accuracy_score(y_test,pred_rfc),accuracy_score(y_test,pred_knn),
                                    accuracy_score(y_test,pred_lor),accuracy_score(y_test,pred_dtc),accuracy_score(y_test,pred_svc),
                                       accuracy_score(y_test,pred_xbc),accuracy_score(y_test,pred_mlp)]})
conclusion


# In[30]:


# Visualizing Results

plt.subplots(figsize=(13, 5))
axis = sns.barplot(x = 'Model', y = 'Accuracy', data =conclusion, palette="mako" )
axis.set(xlabel='Model', ylabel='Accuracy')

# Adding annotation to bars
for p in axis.patches:
    height = p.get_height()
    axis.text(p.get_x() + p.get_width()/2, height + 0.005, '{:1.4f}'.format(height), ha="center") 
    
plt.show()


# In[31]:


# Model Evaluation: Evaluating Random Forest model using Cross Validation

model_eval = cross_val_score(estimator = rfc, X = x_train, y = y_train, cv = 10)
model_eval.mean()


# In[32]:


# Random Forest

# Calculating Training and Testing accuracies
print("Training accuracy :", rfc.score(x_train, y_train))
print("Testing accuracy :", rfc.score(x_test, y_test))

# Confusion Matrix
print("Confusion Matrix :\n",confusion_matrix(y_test, pred_rfc))


# In[33]:


# Error Analysis: Calculating and Printing the Out-of-bag (OOB) error

print("OOB Score:",rfc.oob_score_)


# In[34]:


# Data Accuracy: Tabulating Actual vs Predicted values

y_test = np.array(list(y_test))
y_pred = np.array(pred_rfc)

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': pred_rfc.flatten()})
df[:20]


# In[35]:


# Data Accuracy Visualization: Constructing Barplot of the above response

df1 = df.head(20)
df1.plot(kind='bar',figsize=(12,5))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[ ]:




