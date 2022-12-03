#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd

FILE_NAME = './datasets/pima-indians-diabetes.csv'

dataset = pd.read_csv(FILE_NAME)


# In[4]:


dataset.head()


# In[5]:


dataset.shape


# In[8]:


dataset2 = dataset.values


# In[10]:


X = dataset2[:,0:8]
y = dataset2[:,8]


# In[16]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

#Define the model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[17]:


# compile the keras model
model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])


# In[18]:


# fit the model
model.fit(X, y, epochs=150, batch_size=10)


# In[21]:


# evaluate the keras model
loss , accuracy = model.evaluate(X, y)
print('Accuracy: %.2f ' % (accuracy*100))
print('Loss: %.2f ' % (loss))


# In[22]:


# fit the model
model.fit(X, y, epochs=150, batch_size=10, verbose = 0)


# In[23]:


# evaluate the keras model
loss , accuracy = model.evaluate(X, y)
print('Accuracy: %.2f ' % (accuracy*100))
print('Loss: %.2f ' % (loss))


# # Make Predictions

# In[24]:


# make class predictions with the model
predictions = (model.predict(X) > 0.5).astype(int)
# summarize the first 5 cases
for i in range(5):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))


# In[ ]:


#Assignment

Need to run the code and screen capture each task
1) Load the dataset
2) Define the model with additional hidden layer of 4, activation = relu
3) Compile the model
4) fit the model using 100 epochs and batch size = 20
5) evaluate the model
6) Plot the accuracy and loss of the model

