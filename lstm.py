#coding=gbk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#—————————————————import data——————————————————————
f=open('log-insight-2016.csv')
f_test=open('log-insight-2017.csv')
df=pd.read_csv(f)     #read data
df_test=pd.read_csv(f_test)
data=np.array(df['num'])
data_test=np.array(df_test['num'])
data=data[::-1]
data_test=data_test[::-1]

#plt.figure()
#plt.plot(data)
#plt.show()

normalize_data=(data-np.mean(data))/np.std(data)  #normalization
normalize_data=normalize_data[:,np.newaxis]       #add axis
normalize_data_test=(data_test-np.mean(data_test))/np.std(data_test)  #normalization
normalize_data_test=normalize_data_test[:,np.newaxis]       #add axis

#--------------  generate training dataset and test dataset------------
time_step=14      #time step
rnn_unit=10       #hidden layer units
batch_size=2
input_size=1
output_size=1
lr=0.006         #learning rate

train_x,train_y=[],[]   #training dataset
for i in range(len(normalize_data)-time_step-1):
    x=normalize_data[i:i+time_step]
    y=normalize_data[i+1:i+time_step+1]
    train_x.append(x.tolist())
    train_y.append(y.tolist())

test_x,test_y=[],[]
for j in range(len(normalize_data_test)-time_step-1):
    x=normalize_data_test[j:j+time_step]
    y=normalize_data_test[j+time_step]
    test_x.append(x.tolist())
    test_y.append(y.tolist())

#----------------  define nn variable -------------
X=tf.placeholder(tf.float32, [None,time_step,input_size])    #input
Y=tf.placeholder(tf.float32, [None,time_step,output_size])   #label

weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
         }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
        }

#--------------------------difine lstm------------------
def lstm(batch):      #para
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])

    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)

    init_state=cell.zero_state(batch,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #as the input of the output layer
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states

#--------------------------train model-----------------------
def train_lstm():
    global batch_size
    with tf.variable_scope("lstm") as scope1:
        pred,final_states=lstm(batch_size)
    #loss function
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #train 10000 times
        for i in range(10):
            step=0
            start=0
            end=start+batch_size
            while(end<len(train_x)):
                final_states,loss_=sess.run([train_op,loss],feed_dict={X:train_x[start:end],Y:train_y[start:end]})
                start+=batch_size
                end=start+batch_size
                #save the parameter every 10steps
                if step%10==0:
                    print(i,step,loss_)
                    saver.save(sess,'lstm.model')
                step+=1
train_lstm()

#-------------------prediction-----------------------------
def prediction():
    with tf.variable_scope("lstm", reuse=True) as scope2:
        pred,final_states=lstm(1)      #just input[1,time_step,input_size]
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        saver.restore(sess,'lstm.model')#load lstm model
        #the last line is the training dataset。shape=[1,time_step,input_size]
        #prev_seq=train_x[-1]
        predict=[]

        for i in range(len(normalize_data_test)-time_step-1):
            prev_seq=test_x[i]
            next_seq=sess.run(pred,feed_dict={X:[prev_seq]})
            predict.append(next_seq[-1])

        plt.figure()
        plt.plot(list(range(len(normalize_data_test)-time_step-1)), test_y, color='b')
        plt.plot(list(range(len(normalize_data_test)-time_step-1)), predict, color='r')
        plt.legend(['test_case','predict'],loc='upper right',fontsize=10)
        plt.savefig("log.jpg")
        plt.show()
prediction()

'''
        saver.restore(sess,'stock.model')

        #the last line is the training dataset。shape=[1,time_step,input_size]
        prev_seq=train_x[-1]
        predict=[]

        #get 100 results
        for i in range(100):
            next_seq=sess.run(pred,feed_dict={X:[prev_seq]})
            predict.append(next_seq[-1])
            prev_seq=np.vstack((prev_seq[1:],next_seq[-1]))

'''