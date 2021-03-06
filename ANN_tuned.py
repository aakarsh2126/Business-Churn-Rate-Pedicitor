#Artificial Neural Network
#part 1:Data PreProcessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pickle import load
from keras.models import load_model
# Importing the dataset
dataset=pd.read_csv('Churn_Modelling.csv')
X=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

#encoding categorical data
#removing countries in column 0 as 0,1,2
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X_1=LabelEncoder()
X[:,1]=labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2=LabelEncoder()
X[:,2]=labelencoder_X_2.fit_transform(X[:,2])
onehotencoder=OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


#part 2:Creating ANN(Classifier)
#importing keras library that uses tensorflow as backend
import keras
#Defining ANN as sequence of layers
from keras.models import Sequential
#Creating layers of our Neural Network
from keras.layers import Dense
#Creating Dropouts
from keras.layers import Dropout

#Intialising ANN
classifier=Sequential()
#Adding input layer and first Hidden layer with dropout(Regularisation)
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
classifier.add(Dropout(p=0.1))
#Adding second Hidden layer with dropout(Regularisation)
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
classifier.add(Dropout(p=0.1))
#Adding output layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

#Compiling the ANN
#optimizer is the alogrithm for finding global minima(adam from stochastic gradient descent)
#loss is the loss function which is cross entropy function 
#metrics is the performance metric
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#using tensorboard for storing logs
from keras.callbacks import TensorBoard
losses=TensorBoard(log_dir='./logs',batch_size=10)

#fitting the ANN to training set
classifier.fit(X_train,y_train,batch_size=10,nb_epoch=100,callbacks=[losses])
#Saving Model
classifier.save('model.h5')

#part 3: Evaluating,Improving and Tuning the ANN
import keras
#Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
def build_classifier():
    classifier=Sequential()
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
    classifier.add(Dropout(p=0.1))
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
    classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
classifier=KerasClassifier(build_fn=build_classifier,batch_size=10,nb_epoch=100)
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10,n_jobs=-1)
mean=accuracies.mean();
variance=accuracies.std()

#Improving the ANN
#Dropout Regularization to reduce overfitting if needed

#Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
def build_classifier(optimizer):
    classifier=Sequential()
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
    #Adding Dropouts(Regularization)
    classifier.add(Dropout(p=0.1))
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
    #Adding Dropouts(Regularization)
    classifier.add(Dropout(p=0.1))
    classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
    classifier.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
classifier=KerasClassifier(build_fn=build_classifier)
parameters={'batch_size':[25,32],
            'nb_epoch':[100,500],
            'optimizer':['adam','rmsprop']}
grid_search=GridSearchCV(estimator=classifier,
                         param_grid=parameters,
                         scoring='accuracy',
                         cv=10)
grid_search=grid_search.fit(X_train,y_train)
best_parameters=grid_search.best_params_
best_accuracy=grid_search.best_score_

#Saving Model
classifier.save('model.h5')
