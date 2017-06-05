
# coding: utf-8

# In[4]:
import numpy as np
import tensorflow as tf
#from random import sample
#from tensorflow.examples.tutorials.mnist import input_data


# In[5]:

tf.set_random_seed(1)
def y_reshape(ydn):
	yd_n = []
	for i in ydn:
	    new=[]
	    if i==0.0:
		new.append(1.0)
		new.append(0.0)
	    else:
		new.append(0.0)
		new.append(1.0)
	    new = np.array(new,dtype=np.float32)
	    yd_n.append(new)
	yd_n = np.array(yd_n,dtype=np.float32)
	return yd_n

# In[6]:

#mnist = input_data.read_data_sets('MNIST',one_hot=True)
xd = np.load('/home/sayon/churn_x_train.npy')
yd = np.load('/home/sayon/churn_y_train.npy')
xt = np.load('/home/sayon/churn_x_test.npy')
yt = np.load('/home/sayon/churn_y_test.npy')
yd = yd.reshape([len(yd),1])
yt = yt.reshape([len(yt),1])
yd = y_reshape(yd)
yt = y_reshape(yt)
#print(xd.shape)
#print(yd.shape)
#print(xt.shape)
#print(yt.shape)
# In[7]:

#hyperparameters
lr = 0.00001 #learning rate
training_iter = 700000 #number of iterations
batch_size = 150

n_inputs = 12 #churn data input (shape = 8x12) [basically features of the dataset]
n_steps = 8 #time steps (number of time layers)
n_hidden_units = 256
n_classes = 2 #churn classes i.e. classes of target labels


# In[8]:

#tf graph inputs
x = tf.placeholder(tf.float32,[None,n_steps,n_inputs])
y = tf.placeholder(tf.float32,[None,n_classes])


# In[9]:

#Define weights
weights = {
    #(n_inputs,n_hidden_units)
    'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
    #(n_hidden_units,n_classes)
    'out':tf.Variable(tf.random_normal([n_hidden_units,n_classes]))
}
#Define bias
biases = {
    #(n_hidden_units,)
    'in':tf.Variable(tf.constant(1.0,shape=[n_hidden_units,])),
    #(n_classes,)
    'out':tf.Variable(tf.constant(1.0,shape=[n_classes,]))
}


# In[10]:

#RNN function
def RNN(X,weights,biases):

	#hidden layer for the input to the cell
	#transposing inputs from the shape of (batch_size,n_steps,n_inputs)
	#to (batch_size*n_steps,n_inputs)
	x_in = tf.reshape(x,[-1,n_inputs])
	#push X to hidder unit layers
	# X_in => (batch_size*n_steps,n_hidden_units)
	x_in = tf.matmul(x_in,weights['in'])+biases['in']
	#X_in reshaped to (batch_size,n_steps,n_hidden_units)
	x_in = tf.reshape(x_in,[-1,n_steps,n_hidden_units])

	#cell : basic LSTM cell
	lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,forget_bias=1,state_is_tuple=True)
	# lstm is divided into two parts (c_state,h_state)

	# dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as X_in.
	# Make sure the time_major is changed accordingly.
	# time major is used to put the 1st cell of the tensor shape received by dynamic_rnn as time_steps or not
	# here it's [batch_size,steps,inputs]  => so time_major is False.

	outputs, final_state = tf.nn.dynamic_rnn(lstm_cell,inputs=x_in,time_major=False,dtype=tf.float32)

	#hidden layer for the final result from the final_state
	#result = tf.matmul(final_state[1],weights['out])+biases['out]
	#or
	#unpack to list [(batches,outputs)....]*steps
	outputs = tf.unpack(tf.transpose(outputs,[1,0,2])) #states is the last output
	#val = tf.transpose(outputs, [1, 0, 2])
	#last = tf.gather(val, int(val.get_shape()[0]) - 1)    
	last =  outputs[-1]
	return last


# In[11]:

#prediction+optimization
last = RNN(x,weights,biases)
pred = tf.matmul(last,weights['out'])+biases['out']

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))
train_op = tf.train.AdamOptimizer(lr).minimize(loss)


# In[12]:

#accuracy_calc
#def accuracy(xt,yt,train=True):
correct_pred = tf.equal(tf.arg_max(pred,1),tf.arg_max(y,1))
with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
            tf.scalar_summary('accuracy',accuracy)	    
# return sess.run(accuracy,feed_dict={x:xt,y:yt})*100

#def predict(xt):
#	global pred
#        return sess.run(pred,feed_dict={x:xt})

# In[15]:

#variable initialize+session_run
init = tf.initialize_all_variables()
merged = tf.merge_all_summaries()

with tf.Session() as sess:
    train_writer = tf.train.SummaryWriter("/home/sayon/churn_rnn_logs/train/",sess.graph)
    test_writer = tf.train.SummaryWriter("/home/sayon/churn_rnn_logs/test/",sess.graph)
    sess.run(init)
    step = 0
    while step*batch_size < training_iter:
        perm = np.random.randint(0,xd.shape[0],batch_size)
    	#rindex =  np.array(sample(xrange(xd.shape[0]), batch_size))
        batch_xs, batch_ys = xd[perm],yd[perm]
        sess.run([train_op],feed_dict={x:batch_xs,y:batch_ys})     #training
        if step%20==0:
	    test_accuracy = sess.run(accuracy,feed_dict={x:xt,y:yt})*100
            train_accuracy = sess.run(accuracy,feed_dict={x:xd,y:yd})*100
            print step,"over",",",'test accuracy:',test_accuracy,'%',',','train accuracy:',train_accuracy,"%"
            train_summary = sess.run(merged,feed_dict={
                                        x:xd,
                                        y:yd
                                    })
            test_summary = sess.run(merged,feed_dict={
                                        x:xt,
                                        y:yt
                                    })
            train_writer.add_summary(train_summary,step)
            test_writer.add_summary(test_summary,step)
         
        step+=1
    #writer.close()

# In[ ]:



