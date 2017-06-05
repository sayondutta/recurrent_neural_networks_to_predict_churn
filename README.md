Older version of Tensorflow(0.9) was used: 

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
