# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 20:14:50 2023

@author: ARPITA GUHA NEOGI
"""

import numpy as np
import pickle

#loading the saved model
loaded_model = pickle.load(open('C:/Users/ARPITA GUHA NEOGI/Downloads/trained_model (1).sav','rb'))

input_data = (8,125,96,0,0,0,0.232,54)

#changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = loaded_model.predict(std_data)
print(prediction)

if(prediction == 0):
  print('the person is not diabetic')

else:
  print('The person is diabetic:')
  
  