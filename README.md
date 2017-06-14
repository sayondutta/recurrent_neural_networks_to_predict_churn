Important note on applying a deep learning model a customer base data:
Predicting Customer Churn using past customer behavior data is a approach which is more dependent on the preprocessing of the data and how you are approach that. Since lots of data is sparse and requires lots of hand engineered features.Then feeding them in a machine learning model. Since the data is in a time series format, therefore I tried to use Recurrent Neural Networks (LSTM cell unit). But as I said this might not be the best approach unless the data is huge since a deep learning model will not be able to outperform a traditional machine learning algorithm unless the dataset is extremely large and there is minimum sparseness in the time series sequence of the data. 

Basic Approach of using Dynamic RNN using Tensorflow

Older version of Tensorflow(0.9) was used on a dummy dataset: 

Program:

churn_data_preprocessing.py :

    Dependencies:

        monthly/*

    Outputs:

        churn_x_train.npy  

        churn_y_train.npy

        churn_x_test.npy

        churn_y_test.npy             

rnn_churn.py:

    Dependencies:
    
        churn_x_train.npy
        
        churn_y_train.npy
        
        churn_x_test.npy
        
        churn_y_test.npy
        
    Output:
    
        Trained model
