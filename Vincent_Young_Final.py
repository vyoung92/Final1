#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Final Project
## Vincent Young
### December 13, 2022


# In[2]:


#Q1 Read in the data, call the dataframe "s"  and check the dimensions of the dataframe
import pandas as pd
import streamlit as st
s = pd.read_csv("social_media_usage.csv")
print(s)


# In[3]:


s.shape


# In[4]:


#Q2 Define a function called clean_sm that takes one input, x, and uses `np.where` to check whether x is equal to 1. If it is, make the value of x = 1, otherwise make it 0. Return x. Create a toy dataframe with three rows and two columns and test your function to make sure it works as expected
import numpy as np

def clean_sm(x):
   x= np.where (x == 1, 1,0)
   return x

DF = pd.DataFrame({"column_1":[5,6,7],"column_2":[1,2,3]})

clean_sm(DF.column_2)

print(clean_sm)


# In[5]:


#Q3 Create a new dataframe called "ss". The new dataframe should contain a target column called sm_li which should be a binary variable ( that takes the value of 1 if it is 1 and 0 otherwise (use clean_sm to create this) which indicates whether or not the individual uses LinkedIn, and the following features: income (ordered numeric from 1 to 9, above 9 considered missing), education (ordered numeric from 1 to 8, above 8 considered missing), parent (binary), married (binary), female (binary), and age (numeric, above 98 considered missing). Drop any missing values. Perform exploratory analysis to examine how the features are related to the target.
ss = pd.DataFrame({"sm_li": clean_sm (s.web1h),
                       "income": np.where (s.income > 9, np.nan, s.income),
                       "education": np.where (s.educ2 >8, np.nan, s.educ2),
                       "parent": clean_sm(s.par),
                       "married": clean_sm(s.marital),
                       "gender": clean_sm (s.gender), #1 is male, 0 is not.
                       "age": np.where (s.age > 98, np.nan, s.age)})

ss= ss.dropna()

print(ss)

print(ss.sm_li)


# In[6]:


#Q4 Create a target vector (y) and feature set (X)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
         
y = ss["sm_li"]
x = ss[["income", "education", "parent", "married", "gender", "age"]]


# In[7]:


#Q5 Split the data into training and test sets. Hold out 20% of the data for testing. Explain what each new object contains and how it is used in machine learning
##x_train=80% of data, predicts target during train model. x_test has 20% of data, tests unforseen data. y_train has 80% of data and target for predicting. y_test has 20% of data helps with predicting unseen data. 
x_train, x_test, y_train, y_test = train_test_split(x,
                                                   y,
                                                   stratify = y,
                                                   test_size = 0.2,
                                                   random_state = 500)


# In[8]:


#Q6 Instantiate a logistic regression model and set class_weight to balanced. Fit the model with the training data.
lr = LogisticRegression()
lr.fit(x_train, y_train)


# In[9]:


#Q7 Evaluate the model using the testing data. What is the model accuracy for the model? Use the model to make predictions and then generate a confusion matrix from the model. Interpret the confusion matrix and explain what each number means.
##Top left is true negative, the accurate number of negative examples. Top right is false positive, the negatives classified as positive. Bottom left is false negative, positives classified as negative. Bottom right is true positive, numbers that are accurate positives.
y_pred = lr.predict(x_test)

pd.DataFrame (confusion_matrix (y_test, y_pred),
             columns = ["Predicted negative", "Predictive positive"],
             index = ["Actual negative", "Actual positive"]).style.background_gradient(cmap = "PiYG")


# In[10]:


#Q8 Create the confusion matrix as a dataframe and add informative column names and index names that indicate what each quadrant represents
confusion_matrix = pd.DataFrame({
    "income": [5.0, 8.0], 
    "education": [3.0, 8.0], 
    "parent": [0, 1], 
    "married": [0, 0], 
    "gender": [0, 0], 
    "age": [18, 41]
})

print(confusion_matrix)


# In[11]:


#Q9 Aside from accuracy, there are three other metrics used to evaluate model performance: precision, recall, and F1 score. Use the results in the confusion matrix to calculate each of these metrics by hand. Discuss each metric and give an actual example of when it might be the preferred metric of evaluation. After calculating the metrics by hand, create a classification_report using sklearn and check to ensure your metrics match those of the classification_report.
##Recall
###Precision and Recall should be used together to avoid false positives of either. F1 is a weighted average of Precision and Recall.
40/(40+44)


# In[12]:


##Precision
40/(40+26)


# In[13]:


##F1
(0.6060606060606061*0.47619047619047616)/(0.6060606060606061*0.47619047619047616)
###F1 score would be 2 becuse 1*2


# In[14]:


print(classification_report(y_test, y_pred))


# In[15]:


#Q10 Use the model to make predictions. For instance, what is the probability that a high income (e.g. income=8), with a high level of education (e.g. 7), non-parent who is married female and 42 years old uses LinkedIn? How does the probability change if another person is 82 years old, but otherwise the same?
##high income (e.g. income=8), with a high level of education (e.g. 7), non-parent who is married female and 42 yers
##82 years old, but otherwise the same?
NewData = pd.DataFrame({
    "income": [8.0, 8.0], 
    "education": [7.0, 7.0], 
    "parent": [0, 0], 
    "married": [1, 1], 
    "gender": [0, 0],
    "age": [42, 82]
})

NewData


# In[16]:


NewData["Pred_sm_li"]=lr.predict(NewData)

NewData


# In[18]:


PersonA = [8.0, 7.0, 0, 1, 0, 42]
PersonB = [8.0, 7.0, 0, 1, 0, 82]

predicted_class1 = lr.predict([PersonA])
predicted_class2 = lr.predict([PersonB])


# In[19]:


probs1 = lr.predict_proba([PersonA])
probs2 = lr.predict_proba([PersonB])


# In[20]:


print(f"predicted_class1: {predicted_class1}")
print(f"predicted_class2: {predicted_class2}")


# In[ ]:




