import matplotlib.pyplot as plt
import numpy as np

from scipy.io import loadmat
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn.python.ops import core_rnn as contrib_rnn
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import variable_scope as vs
# Import MNIST data

np.set_printoptions(threshold=np.nan)


n_hidden = 120

class DataSet:
    def __init__(self, features, labels, seq_length,batch_size):
        self.features = features
        self.labels = labels
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.offset = 0
        
        self._features = features
        self._labels = labels
        self._seq_length = seq_length

        self._epochs_completed = 0
        self._index_in_epoch = 0

        self._num_examples = labels.shape[0]

        
       
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
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._features = self.features[perm0]
            self._labels = self.labels[perm0]
            self._seq_length = self.seq_length[perm0]

        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            features_rest_part = self._features[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            seq_length_rest_part = self._seq_length[start:self._num_examples]

            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._features = self.features[perm]
                self._labels = self.labels[perm]
                self._seq_length = self.seq_length[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            features_new_part = self._features[start:end]
            labels_new_part = self._labels[start:end]
            seq_length_new_part = self._seq_length[start:end]
            return np.concatenate((features_rest_part, features_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0), np.concatenate((seq_length_rest_part, seq_length_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._features[start:end], self._labels[start:end], self._seq_length[start:end]
        print start
        print end

def doPad(x, length):
    # return np.pad(x,((0,0),(0,length-len(x[0]))), mode='constant', constant_values=0)
    return np.pad(x,((0,0),(length-len(x[0]),0)), mode='constant', constant_values=0)

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
    
    # caluculate the mean
    trainSamplesL = trainSamples.tolist()
    print trainSamplesL[0].shape   
    trainSamplesStacked = np.concatenate(trainSamplesL, axis = 1)
    print trainSamplesStacked.shape

    cm = np.mean(trainSamplesStacked, axis = 1)
    cstdv = np.std(trainSamplesStacked,axis = 1)

    print cm.shape
    print cstdv.shape

    for i in  range(trainSamples.shape[0]):
        
        m = np.matlib.repmat(cm,trainSamples[i].shape[1],1).T
        
        stdv = np.matlib.repmat(cstdv,trainSamples[i].shape[1],1).T

        trainSamples[i] -= (m)
        trainSamples[i] = np.divide(trainSamples[i], stdv)
        # print np.mean(trainSamples[i], axis = 1)

    trainSamples = np.array((map(lambda x:doPad(x, length), trainSamples)))
    trainSamples = np.array((map(lambda x:x.T, trainSamples)))
    # trainSamples = np.array((map(lambda x:x.T, trainSamples)))
    trainLabels = np.array((map(lambda x:toOneHot(x, totClass), trainLabels)))

    
    #normalization
    # trainSamples -= np.array(map(trainSamples, lambda x:np.mean(x, axis = 0),trainSamples))
    tstData = loadmat('./data/tidigits_mfccs_test.mat')
    tstSamples = tstData['tidigits_mfccs_test'][0][0][5][0]
    tstLabels  = tstData['tidigits_mfccs_test'][0][0][8][0]
    ts_seq_length = np.array(map(lambda x: x.shape[1], tstSamples))
    for i in  range(tstSamples.shape[0]):
        
        m = np.matlib.repmat(cm,tstSamples[i].shape[1],1).T

        stdv = np.matlib.repmat(cstdv,tstSamples[i].shape[1],1).T
        
        tstSamples[i] -= (m)
        tstSamples[i] = np.divide(tstSamples[i], stdv)

    tstSamples = np.array((map(lambda x:doPad(x, length), tstSamples)))
    tstSamples = np.array((map(lambda x:x.T, tstSamples)))
    tstLabels = np.array((map(lambda x:toOneHot(x, totClass), tstLabels)))

    print "Load Compelete"
    

    trainData = DataSet(trainSamples,trainLabels,tr_seq_length,batchSize)
    tstData = DataSet(tstSamples,tstLabels,ts_seq_length,batchSize)
    
    return trainData, tstData

def RNN2(x, seq_length, weights, bias):
    drop_out_rate = 0.75
    multicell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(
                        n_hidden,
                        # state_is_tuple=True
                        ),
        input_keep_prob=drop_out_rate) for _ in range(3)], state_is_tuple=True)
    
    outputs, state = tf.nn.dynamic_rnn(cell = multicell, 
        inputs = x,
        dtype=tf.float32
        # , sequence_length = seq_length
        )
    outputs = tf.transpose(outputs, [1, 0, 2])
    last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)

    # last = outputs[-1]
    return tf.nn.softmax(tf.matmul(last, weights) + bias)


def main():
    

    # Network Parameters
    learning_rate = 1e-4
    training_iters = 400000
    batch_size = 128
    display_step = 10
    n_input = 39
    n_steps = 249
    n_hidden2 = 200
    n_classes = 11

    tr_data, ts_data = loadData(batch_size)

    x = tf.placeholder("float", [None, n_steps, n_input])
    y = tf.placeholder("float", [None, n_classes])
    seq = tf.placeholder(tf.int32, [None])

    weights = tf.Variable(tf.random_normal([n_hidden, n_classes]))
    bias = tf.Variable(tf.random_normal([n_classes]))
    pred  = RNN2(x, seq, weights, bias)
    # pred = tf.nn.softmax(logits)
    # pred  = BiRNN(x, seq, weights2, bias2)
    # pred  = BLSTM(x, seq, weights2, bias2)
    cost = cross_entropy = -tf.reduce_sum(y * tf.log(pred))
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # Initializing the variables
    init = tf.global_variables_initializer()
    epoch = 0

    batch_x, batch_y, batch_seq = tr_data.next_batch(batch_size)
    

    saver = tf.train.Saver()

    pltIter = []
    pltAcc = []
    pltLoss = []

    print "start"
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
           
            batch_x, batch_y, batch_seq = tr_data.next_batch(batch_size)
            # batch_seq = np.array([249 for i in range(batch_size)])
            # return
            # batch_x = np.transpose(batch_x,(0,2,1))
            
          
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, seq:batch_seq})
            if step % display_step == 0:
               
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, seq:batch_seq})
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, seq:batch_seq})

                pltIter.append(step)
                pltAcc.append(acc)
                pltLoss.append(loss)

                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                "{:.6f}".format(loss) + ", Training Accuracy= " + \
                "{:.5f}".format(acc))
            step += 1
        print("Optimization Finished!")


        plt.subplot(211)
        plt.plot(pltIter, pltLoss)

        plt.subplot(212)
        plt.plot(pltIter, pltAcc)
        plt.savefig('./figures/Iter vs Loss & acc',format = 'png')

        plt.show()

        model_path = './models/model.ckpt'
        saver.save(sess, model_path)

        test_len = ts_data._num_examples
        test_data = ts_data.features[:test_len]
        # test_data = np.transpose(test_data,(0,2,1))
        test_label = ts_data.labels[:test_len]
        test_seq = ts_data.seq_length[:test_len]
        #acc = sess.run(acc,feed_dict={x: test_data, y: test_label, seq: test_seq})
        print("Testing Accuracy:", \
            sess.run(accuracy,feed_dict={x: test_data, y: test_label, seq: test_seq}))
    
if __name__ == '__main__':
    main()