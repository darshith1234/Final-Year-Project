#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import re
import string



# In[6]:


data_fake = pd.read_csv('Fake.csv')
data_true = pd.read_csv('True.csv')


# In[7]:


data_fake.head()


# In[8]:


data_true.head()


# In[9]:


data_fake["class"] = 0
data_true["class"] = 1


# In[10]:


data_fake.shape, data_true.shape #rows and colums


# In[11]:


data_fake_manual_testing= data_fake.tail(10)
for i in range (23480, 23470,-1):
     data_fake.drop([i], axis = 0, inplace = True )
    
data_true_manual_testing= data_true.tail(10)
for i in range (21416, 21406,-1):
     data_true.drop([i], axis = 0, inplace = True)      


# In[12]:


data_fake.shape, data_true.shape


# In[13]:


data_fake_manual_testing['class'] =0
data_true_manual_testing['class'] =1


# In[14]:


data_fake_manual_testing.head(10)


# In[15]:


data_true_manual_testing.head(10)


# In[16]:


data_merge = pd.concat([data_fake, data_true], axis = 0)
data_merge.head(10)


# In[17]:


data_merge.columns


# In[18]:


data = data_merge.drop(['title','subject', 'date'], axis=1)


# In[19]:


data.isnull().sum()


# In[20]:


data = data.sample(frac =1)


# In[21]:


data.head()


# In[22]:


data.reset_index(inplace = True)
data.drop(['index'], axis = 1, inplace= True)


# In[23]:


data.columns


# In[24]:


data.head()


# In[25]:


def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]','', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\s+|www\.\S+', '', text)
    text = re.sub('<.*?>+','', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n','', text)
    text = re.sub('\w*\d\w*','', text)
    return text


# In[26]:


data['text']= data['text'].apply(wordopt)


# In[27]:


x= data['text']
y= data['class']


# In[28]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25)


# In[29]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)


# In[30]:


from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(xv_train, y_train)


# In[31]:


pred_lr = LR.predict(xv_test)


# In[32]:


LR.score(xv_test, y_test)


# In[33]:


print(classification_report(y_test, pred_lr))


# In[34]:


from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)


# In[35]:


pred_dt = DT.predict(xv_test)


# In[36]:


DT.score(xv_test, y_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[37]:


print(classification_report(y_test, pred_dt))


# In[38]:


from sklearn.ensemble import GradientBoostingClassifier

GB = GradientBoostingClassifier(random_state = 0)
GB.fit(xv_train, y_train)


# In[39]:


predit_gb = GB.predict(xv_test)


# In[40]:


GB.score(xv_test, y_test)


# In[41]:


print(classification_report(y_test, predit_gb))


# In[42]:


from sklearn.ensemble import RandomForestClassifier

RF =  RandomForestClassifier(random_state = 0)
RF.fit(xv_train, y_train)


# In[43]:


pred_rf = RF.predict(xv_test)


# In[44]:


RF.score(xv_test, y_test)


# In[45]:


print(classification_report(y_test, pred_rf))


# In[57]:


def output_lable(n):
    if n ==0:
        return "Fake News"
    elif n==1:
        return "Not A Fake News" 
def manual_testing(news):
    testing_news = {"test":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["test"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GB = GB.predict(new_xv_test)
    pred_RF = RF.predict(new_xv_test)
    
    
    return print("\n\nLR Prediction: {} \nDT Prediction: {} \nGB Prediction: {} \nRF Prediction: {}".format(output_lable(pred_LR[0]), output_lable(pred_DT[0]), output_lable(pred_GB[0]), output_lable(pred_RF[0])))



# In[ ]:





# In[58]:


news = str(input())
manual_testing(news)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[48]:


import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
accuracy_scores = []

for threshold in thresholds:
    thresholded_pred_lr = (pred_lr >= threshold).astype(int)
    accuracy_scores.append(accuracy_score(y_test, thresholded_pred_lr))


plt.plot(thresholds, accuracy_scores, marker='o')
plt.title('Accuracy Score vs. Threshold')
plt.xlabel('Threshold')
plt.ylabel('Accuracy Score')
plt.grid(True)
plt.show()


# In[ ]:





# In[49]:


from sklearn.metrics import accuracy_score, precision_score
import matplotlib.pyplot as plt

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

LR = LogisticRegression()
LR.fit(xv_train, y_train)

pred_lr = LR.predict(xv_test)

accuracy = accuracy_score(y_test, pred_lr)

precision = precision_score(y_test, pred_lr, average='weighted')

labels = ['Accuracy', 'Precision']
scores = [accuracy, precision]

plt.bar(labels, scores, color=['blue', 'green'])
plt.title('Accuracy and Precision Scores')
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.show()


# In[ ]:





# In[50]:


from sklearn.metrics import accuracy_score, precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


accuracy = accuracy_score(y_test, predit_gb)
precision = precision_score(y_test, predit_gb)


print(classification_report(y_test, predit_gb))


labels = ['Accuracy', 'Precision']
values = [accuracy, precision]

plt.bar(labels, values, color=['orange', 'pink'])
plt.title('Accuracy and Precision')
plt.show()


# In[ ]:




