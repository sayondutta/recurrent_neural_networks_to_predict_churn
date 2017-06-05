
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import os


# In[2]:

list_files = os.listdir('/home/sayon/monthly/')
list_files = ['/home/sayon/monthly/'+i for i in list_files]


# In[3]:

list_files


# In[4]:

list_files = sorted(list_files)


# In[5]:

train_files = list_files[:-1]


# In[6]:

training_set_users = pd.read_csv(list_files[-2],header=0)
training_set_users = np.unique(training_set_users.user_id.values)


# In[7]:

test_files = list_files[1:]


# In[8]:

testing_set_users = pd.read_csv(list_files[-1],header=0)
testing_set_users = np.unique(testing_set_users.user_id.values)


# In[9]:

len_train_files = len(train_files)
len_test_files = len(test_files)


# In[10]:

train_stack = pd.DataFrame()
c=0
for i in train_files:
    abc = pd.read_csv(i,header=0)
    abc['time_id']=[c for i in range(abc.shape[0])]
    train_stack = train_stack.append(abc)
    c+=1


# In[11]:

churn_labels = []
for i in range(train_stack.shape[0]):
#     if train_stack.requests_accepted.values[i] == 0.0 and train_stack.requests_sent.values[i] == 0.0 and train_stack.travel_count.values[i] == 0.0:
#         churn_labels.append(1.0)
#     else:
#         churn_labels.append(0.0)
    if train_stack.churn.values[i]==1.0:
        churn_labels.append(0.0)
    else:
        churn_labels.append(1.0)


# In[12]:

train_stack['churn'] = churn_labels


# In[13]:

test_stack = pd.DataFrame()
c=0
for i in test_files:
    abc = pd.read_csv(i,header=0)
    abc['time_id']=[c for i in range(abc.shape[0])]
    test_stack = test_stack.append(abc)
    c+=1


# In[14]:

churn_labels = []
for i in range(test_stack.shape[0]):
#     if test_stack.requests_accepted.values[i] == 0.0 and test_stack.requests_sent.values[i] == 0.0 and test_stack.travel_count.values[i] == 0.0:
#         churn_labels.append(1.0)
#     else:
#         churn_labels.append(0.0)
    if test_stack.churn.values[i]==1.0:
        churn_labels.append(0.0)
    else:
        churn_labels.append(1.0)


# In[15]:

test_stack['churn'] = churn_labels


# In[16]:

train_stack1 = train_stack.sort(['user_id','time_id'],ascending=True).reset_index()
train_stack1.drop(['index','recency_minutes','route_distance'],axis=1,inplace=True)


# In[17]:

users = np.unique(train_stack1.user_id.values)


# In[18]:

len(users)


# In[19]:

test_stack1 = test_stack.sort(['user_id','time_id'],ascending=True).reset_index()
test_stack1.drop(['index','recency_minutes','route_distance'],axis=1,inplace=True)


# In[20]:

userss = np.unique(test_stack1.user_id.values)


# In[21]:

len(userss)


# In[22]:

training_dataset = train_stack1[train_stack1.user_id.isin(training_set_users)]
testing_dataset = test_stack1[test_stack1.user_id.isin(testing_set_users)]


# In[23]:

users = np.unique(training_dataset.user_id.values)
print len(users)
userss = np.unique(testing_dataset.user_id.values)
print len(userss)


# In[27]:

def reshaping_data(stack,len_stack_files):
    users = np.unique(stack.user_id.values)
    x=[]
    y=[]
    z=0
    for j in users:
        #print j,z
        #z=z+1
        #print z
        a = stack[stack.user_id.values==j]
        b = a.time_id.values
        #print b
        c = list(set(range(len_stack_files))-set(list(b)))
        #print c
        #df = pd.DataFrame(columns=a.columns.values)
        if len(c)>0:
            for i in c:
                df = pd.DataFrame(columns=a.columns.values)
                row = [j,net_last_balance(i,a),0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,a.user_gender.values[0],0.0,-1.0,i]
                #print i
                df.loc[-1]=row
                df.index = df.index + 1
                a = a.append(df)
                a = a.sort(['time_id'],ascending=True).reset_index()
                a.drop(['index'],axis=1,inplace=True)

        batch_y = a.iloc[-1,-2]

        for i in a.columns.values:
            a[i]=moving_average(a[i].values)
        batch_x = a.iloc[:,1:-2].values

        x.append(batch_x)
        y.append(batch_y)
        #if z%1000 == 0:
        #    print z
    x = np.asarray(x)
    y = np.asarray(y)

    return x,y

def net_last_balance(i,a2):
    if (i<a2.time_id.values[0] or i==0.0):
        return 0.0
    else:
        return a2.net_last_balance.values[i-1]
    
def moving_average(l):
    ln = []
    ln.append(l[0])
    for i in range(1,len(l)):
        val = (ln[i-1]+l[i])/2
        ln.append(val)
    return ln


# In[28]:

# b = a1.time_id.values
# c=list(set(range(len_stack_files))-set(list(b)))
# df = pd.DataFrame(columns=a1.columns.values)
# for i in c:
#     print i
#     row = [a1.user_id.values[0],net_last_balance(i,a1),0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,a1.user_gender.values[0],0.0,-1.0,i]
#     df.loc[-1]=row
#     df.index = df.index + 1
# a3 = a1.append(df)
# a3 = a3.sort(['time_id'],ascending=True).reset_index()
# a3.drop(['index'],axis=1,inplace=True)


# x = a3.iloc[:,1:-2]
# y = a3.iloc[-1,-2]

# for i in a3.columns.values:
#     a3[i]=moving_average(a3[i].values)


# In[29]:

x_train, y_train = reshaping_data(stack=training_dataset,len_stack_files=len_train_files)


# In[ ]:

x_test, y_test = reshaping_data(stack=testing_dataset,len_stack_files=len_test_files)


# In[ ]:

np.save('/home/sayon/churn_x_train.npy', x_train)
np.save('/home/sayon/churn_y_train.npy',y_train)
np.save('/home/sayon/churn_x_test.npy', x_test)
np.save('/home/sayon/churn_y_test.npy',y_test)

