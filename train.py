

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from scipy.io import loadmat
import numpy
# Import MNIST data

n_hidden = 200
'''
To classify images using a bidirectional recurrent neural network, we consider
every image row as a sequence of pixels. Because MNIST image shape is 28*28px,
we will then handle 28 sequences of 28 steps for every sample.
'''
class DataSet:
    def __init__(self, features, labels, seq_length,batch_size):
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.offset = 0
        
        self._features = features
        self._labels = labels
        self._seq_length = seq_length

        self._epochs_completed = 0
        self._index_in_epoch = 0

        self._num_examples = labels.shape[0]

        self.seq_length = seq_length
       
    def nextBatchData(self):
        re = []
        for i in range(self.batch_size):
            re.append(self.data[self.offset + i])
        return re
    def nextBatchLabel(self):
        re = []
        for i in range(self.batch_size):
            re.append(self.labels[self.offset + i])
        return re
    def next(self):
        d = self.nextBatchData();
        l = self.nextBatchLabel();
        self.offset += self.batch_size
        # print self.offset
        return np.array(d),np.array(l)

    def next_batch(self, batch_size,shuffle=False):
        start = self._index_in_epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm0)
            self._features = self.features[perm0]
            self._labels = self.labels[perm0]
            self._seq_length = self._seq_length[perm0]

        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            features_rest_part = self._features[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            seq_length_rest_part = self._seq_length[start:self._num_examples]

            # Shuffle the data
            if shuffle:
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._features = self.features[perm]
                self._labels = self.labels[perm]
                self._seq_length = self._seq_length[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            features_new_part = self._features[start:end]
            labels_new_part = self._labels[start:end]
            seq_length_new_part = self._seq_length[start:end]
            return numpy.concatenate((features_rest_part, features_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0), numpy.concatenate((seq_length_rest_part, seq_length_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._features[start:end], self._labels[start:end], self._seq_length[start:end]


def doPad(x, length):
    return np.pad(x,((0,0),(0,length-len(x[0]))), mode='constant', constant_values=0)

def toOneHot(num, totClass):
    vec = np.zeros(1 + totClass)
    vec[num] = 1
    return vec

def loadData(batchSize):
    length = 249
    totClass = 10

    trainData = loadmat('./data/tidigits_mfccs_train.mat')
    # trainData = loadmat('./data/d.mat')
    trainSamples = trainData['tidigits_mfccs_train'][0][0][5][0]
    trainLabels = trainData['tidigits_mfccs_train'][0][0][8][0]
    
    
    tr_seq_length = np.array(map(lambda x: x.shape[1], trainSamples))



    for i in  range(trainSamples.shape[0]):
        cc =  np.mean(trainSamples[i], axis = 1)
        cc = np.matlib.repmat(cc, 249,1).T
        
        
        m = np.mean(trainSamples[i],axis = 1).T
        m = np.matlib.repmat(m,trainSamples[i].shape[1],1).T
        
      
        stdv = np.std(trainSamples[i],axis = 1).T
        stdv = np.matlib.repmat(stdv,trainSamples[i].shape[1],1).T

        trainSamples[i] -= (m)
        trainSamples[i] = np.divide(trainSamples[i], stdv)
        # print np.mean(trainSamples[i], axis = 1)



    trainSamples = np.array((map(lambda x:doPad(x, length), trainSamples)))

    trainSamples = np.dstack(trainSamples)

    trainSamples=np.rollaxis(trainSamples,-1)

    # trainSamples = np.array((map(lambda x:x.T, trainSamples)))
    trainLabels = np.array((map(lambda x:toOneHot(x, totClass), trainLabels)))

    

    #normalization
    # trainSamples -= np.array(map(trainSamples, lambda x:np.mean(x, axis = 0),trainSamples))

    
    tstData = loadmat('./data/tidigits_mfccs_test.mat')
    tstSamples = tstData['tidigits_mfccs_test'][0][0][5][0]
    tstLabels  = tstData['tidigits_mfccs_test'][0][0][8][0]


    for i in  range(tstSamples.shape[0]):
        
        cc =  np.mean(tstSamples[i], axis = 1)
        cc = np.matlib.repmat(cc, 249,1).T
        
        m = np.mean(tstSamples[i],axis = 1)
        m = np.mean(tstSamples[i],axis = 1).T
        
        m = np.matlib.repmat(m,tstSamples[i].shape[1],1).T


        stdv = np.std(tstSamples[i],axis = 1).T
        stdv = np.matlib.repmat(stdv,tstSamples[i].shape[1],1).T
        
        tstSamples[i] -= (m)
        tstSamples[i] = np.divide(tstSamples[i], stdv)

    print tstSamples[i].shape

    ts_seq_length = np.array(map(lambda x: x.shape[1], tstSamples))
    print ts_seq_length
    tstSamples = np.array((map(lambda x:doPad(x, length), tstSamples)))
    tstSamples = np.array((map(lambda x:x.T, tstSamples)))
    tstLabels = np.array((map(lambda x:toOneHot(x, totClass), tstLabels)))


    

    print "Load Compelete"
    

    trainData = DataSet(trainSamples,trainLabels,tr_seq_length,batchSize)
    tstData = DataSet(tstSamples,tstLabels,ts_seq_length,batchSize)
    
    return trainData, tstData


def loadData2(batchSize):
    length = 249
    totClass = 10

    trainData = loadmat('./data/tidigits_mfccs_train.mat')
    # trainData = loadmat('./data/d.mat')
    trainSamples = trainData['tidigits_mfccs_train'][0][0][5][0]
    trainLabels = trainData['tidigits_mfccs_train'][0][0][8][0]
    
    
    tr_seq_length = np.array(map(lambda x: x.shape[1], trainSamples))



    for i in  range(trainSamples.shape[0]):
        cc =  np.mean(trainSamples[i], axis = 1)
        cc = np.matlib.repmat(cc, 249,1).T
        
        
        m = np.mean(trainSamples[i],axis = 1).T
        m = np.matlib.repmat(m,trainSamples[i].shape[1],1).T
        
      
        stdv = np.std(trainSamples[i],axis = 1).T
        stdv = np.matlib.repmat(stdv,trainSamples[i].shape[1],1).T

        trainSamples[i] -= (m)
        trainSamples[i] = np.divide(trainSamples[i], stdv)
        # print np.mean(trainSamples[i], axis = 1)



    trainSamples = np.array((map(lambda x:doPad(x, length), trainSamples)))

    trainSamples = np.dstack(trainSamples)

    trainSamples=np.rollaxis(trainSamples,-1)

    # trainSamples = np.array((map(lambda x:x.T, trainSamples)))
    trainLabels = np.array((map(lambda x:toOneHot(x, totClass), trainLabels)))

    

    #normalization
    # trainSamples -= np.array(map(trainSamples, lambda x:np.mean(x, axis = 0),trainSamples))

    
    tstData = loadmat('./data/tidigits_mfccs_test.mat')
    tstSamples = tstData['tidigits_mfccs_test'][0][0][5][0]
    tstLabels  = tstData['tidigits_mfccs_test'][0][0][8][0]

    
    for i in  range(tstSamples.shape[0]):
        
        cc =  np.mean(tstSamples[i], axis = 1)
        cc = np.matlib.repmat(cc, 249,1).T
        
        m = np.mean(tstSamples[i],axis = 1)
        m = np.mean(tstSamples[i],axis = 1).T
        
        m = np.matlib.repmat(m,tstSamples[i].shape[1],1).T


        stdv = np.std(tstSamples[i],axis = 1).T
        stdv = np.matlib.repmat(stdv,tstSamples[i].shape[1],1).T
        
        tstSamples[i] -= (m)
        tstSamples[i] = np.divide(tstSamples[i], stdv)

    print tstSamples[i].shape

    ts_seq_length = np.array(map(lambda x: x.shape[1], tstSamples))
    print ts_seq_length
    tstSamples = np.array((map(lambda x:doPad(x, length), tstSamples)))
    tstSamples = np.array((map(lambda x:x.T, tstSamples)))
    tstLabels = np.array((map(lambda x:toOneHot(x, totClass), tstLabels)))
    print "Load Compelete"
    

    trainData = DataSet(trainSamples,trainLabels,tr_seq_length,batchSize)
    tstData = DataSet(tstSamples,tstLabels,ts_seq_length,batchSize)
    
    return trainData, tstData


def BiRNN(x, seq_length ,weights, biases):

    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    # x = tf.unstack(x, n_steps, 1)

    # Define lstm cells with tensorflow
    # Forward direction cell
    
    x = tf.unstack(x, 249, 1)

    drop_out = 0.8
    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob=drop_out, output_keep_prob=drop_out)
    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob=drop_out, output_keep_prob=drop_out)
    # Get lstm cell output

    outputs_fw, outputs_bw  = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,cell_bw=lstm_bw_cell,dtype=tf.float32,sequence_length=seq_length,inputs=x)

    # outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

    # Linear activation, using outputs computed above
    return tf.matmul(outputs, weights) + biases
    # Linear activation, using rnn inner loop last output
    # return tf.matmul(outputs[-1], weights['out']) + biases['out']

def RNN(x, seq_length, weights, bias, weights2, bias2):
    drop_out_rate = 0.8

    cell = tf.contrib.rnn.LSTMCell(n_hidden,state_is_tuple = True)
    # cell = tf.contrib.rnn.GRUCell(n_hidden)
    cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=drop_out_rate)
    # cell = tf.contrib.rnn.MultiRNNCell([cell] * 1)
    # cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=drop_out_rate)
    cell = tf.contrib.rnn.MultiRNNCell([cell] * 2)
    

    outputs, state = tf.nn.dynamic_rnn(cell = cell, inputs = x,dtype=tf.float32)
    outputs = tf.transpose(outputs, [1, 0, 2])
    last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)

    
    # tmp = tf.nn.relu(tf.matmul(last, weights) + bias)

    return tf.nn.softmax(tf.matmul(last, weights2) + bias2)

def RNN2(x, seq_length, weights, bias):
    drop_out_rate = 0.8
    
    # cell = tf.contrib.rnn.LSTMCell(n_hidden,state_is_tuple = True)
    # # cell = tf.contrib.rnn.GRUCell(n_hidden)
    # cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=drop_out_rate)
    # # cell = tf.contrib.rnn.MultiRNNCell([cell] * 1)
    # # cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=drop_out_rate)
    # cell = tf.contrib.rnn.MultiRNNCell([cell] * 2)

    multicell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(n_hidden),
		input_keep_prob=drop_out_rate) for _ in range(2)], state_is_tuple=False)
    

    outputs, state = tf.nn.dynamic_rnn(cell = multicell, inputs = x,dtype=tf.float32)
    outputs = tf.transpose(outputs, [1, 0, 2])
    last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)

    
    # tmp = tf.nn.relu(tf.matmul(last, weights) + bias)

    return tf.nn.softmax(tf.matmul(last, weights) + bias)



def main():

    tf.reset_default_graph()

    learning_rate = 0.0001
    #0.0001
    # training_iters = 25000
    training_iters = 400000
    batch_size = 100
    display_step = 10

    tr_data, ts_data = loadData(batch_size);

    batch_x, batch_y, batch_seq = tr_data.next_batch(batch_size)
    

   

    # Network Parameters
    n_input = 39
    n_steps = 249
    n_hidden2 = 200
    n_classes = 11

    x = tf.placeholder("float", [None, n_steps, n_input])
    y = tf.placeholder("float", [None, n_classes])
    seq = tf.placeholder(tf.int32, [None])


    weight = tf.Variable(tf.random_normal([n_hidden, n_hidden]))
    weights2 = tf.Variable(tf.random_normal([n_hidden, n_classes]))
    
    bias = tf.Variable(tf.random_normal([n_hidden]))
    bias2 = tf.Variable(tf.random_normal([n_classes]))


    # pred  = RNN(x, seq, weight, bias, weights2, bias2)

    pred  = RNN2(x, seq, weights2, bias2)
    # pred = BiRNN(x, weight, bias)

    # tv = tf.trainable_variables()
    # regularization_cost = tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ])

    
    
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    
    # cost = cross_entropy = -tf.reduce_sum(y * tf.log(tf.clip_by_value(pred,1e-10,1.0))) 

    cost = cross_entropy = -tf.reduce_sum(y * tf.log(pred))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    #optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss_f)
    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.initialize_all_variables()
    epoch = 0

    test_len = ts_data._num_examples/10
    test_data = ts_data.features[:test_len].reshape((-1, n_steps, n_input))
    test_label = ts_data.labels[:test_len]

    saver = tf.train.Saver()



    with tf.Session() as sess:
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            # print(offset)
            # batch_x = tr_features[offset:(offset + batch_size), :, :]
            # batch_y = tr_labels[offset:(offset + batch_size), :]
            
            batch_x, batch_y, batch_seq = tr_data.next_batch(batch_size)
            #print batch_x[0][0]
            # offset = (step * batch_size) % (tr_labels.shape[0] - batch_size)

            # batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Reshape data to get 28 seq of 28 elements
            batch_x = np.transpose(batch_x,(0,2,1))
            
            
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, seq:batch_seq})
            if step % display_step == 0:
                # Calculate batch accuracy
                # print batch_x[0]
                # print batch_y[0]
                # print sess.run(tf.argmax(y,1),feed_dict = {y:batch_y})
                # print sess.run(tf.argmax(pred,1),feed_dict = {x:batch_x, seq:batch_seq})
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, seq:batch_seq})

                
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, seq:batch_seq})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                "{:.6f}".format(loss) + ", Training Accuracy= " + \
                "{:.5f}".format(acc))
            step += 1
        print("Optimization Finished!")
        model_path = './models/model.ckpt'
        saver.save(sess, model_path)
        # Calculate accuracy for 128 mnist test images
        test_len = ts_data._num_examples
        test_data = ts_data.features[:test_len]
        # test_data = np.transpose(test_data,(0,2,1))
        test_label = ts_data.labels[:test_len]
        test_seq = ts_data.seq_length[:test_len]
        print test_data.shape
        # test_data = test_data.T

        # test_len = mnist.test.images.shape[0]
        # test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
        # test_label = mnist.test.labels[:test_len]

        print test_data.shape
        #acc = sess.run(acc,feed_dict={x: test_data, y: test_label, seq: test_seq})
        print("Testing Accuracy:", \
            sess.run(accuracy,feed_dict={x: test_data, y: test_label, seq: test_seq}))
    

def main2():
    tf.reset_default_graph()

    learning_rate = 0.0001
    # training_iters = 25000
    training_iters = 400000
    batch_size = 100
    display_step = 10

    tr_data, ts_data = loadData(batch_size);

    batch_x, batch_y, batch_seq = tr_data.next_batch(batch_size)

    # Network Parameters
    n_input = 39
    n_steps = 249
    n_hidden2 = 200
    n_classes = 11

    x = tf.placeholder("float", [None, n_steps, n_input])
    y = tf.placeholder("float", [None, n_classes])
    seq = tf.placeholder(tf.int32, [None])


    weight = tf.Variable(tf.random_normal([n_hidden, n_hidden]))
    weights2 = tf.Variable(tf.random_normal([n_hidden, n_classes]))
    
    bias = tf.Variable(tf.random_normal([n_hidden]))
    bias2 = tf.Variable(tf.random_normal([n_classes]))

    pred  = RNN(x, seq, weight, bias, weights2, bias2)

    cost = cross_entropy = -tf.reduce_sum(y * tf.log(pred))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    #optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss_f)
    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.initialize_all_variables()
    nepoch = 100

    test_len = ts_data._num_examples/10
    test_data = ts_data.features[:test_len].reshape((-1, n_steps, n_input))
    test_label = ts_data.labels[:test_len]

    with tf.Session() as sess:
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        for epoch in range(nepoch):
            print epoch
            for curP in range(0, tr_data._num_examples, batch_size):
                batch_x = tr_data.features[curP:curP + batch_size]
                batch_y = tr_data.labels[curP:curP + batch_size]
                batch_seq = tr_data.seq_length[curP:curP + batch_size]
                batch_x = np.transpose(batch_x,(0,2,1))
                # Run optimization op (backprop)
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, seq:batch_seq}) 
            #test
            test_len = batch_size
            
            test_data = ts_data.features[:test_len]
            # test_data = np.transpose(test_data,(0,2,1))
            test_label = ts_data.labels[:test_len]
            test_seq = ts_data.seq_length[:test_len]

            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, seq:batch_seq})
                    # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, seq:batch_seq})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                    "{:.6f}".format(loss) + ", Training Accuracy= " + \
                    "{:.5f}".format(acc))


        print("Optimization Finished!")

        test_len = batch_size
        test_data = ts_data.features[:test_len]
        # test_data = np.transpose(test_data,(0,2,1))
        test_label = ts_data.labels[:test_len]
        test_seq = ts_data.seq_length[:test_len]
        
        print test_data.shape
        cc = sess.run(acc,feed_dict={x: test_data, y: test_label, seq: test_seq})
        print("Testing Accuracy:", \
            sess.run(acc,feed_dict={x: test_data, y: test_label, seq: test_seq}))
if __name__ == '__main__':
    main()