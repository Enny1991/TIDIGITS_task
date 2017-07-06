# New comment
import datetime
from scipy.io import savemat
import time
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn_cell import LSTMCell

from loader_TIDIGITS import load_dataset
from tabulate import tabulate


flags = tf.flags

flags.DEFINE_string("data_path", "/Data/Dropbox/AudioGroup/data/tidigits/", "data_path")
flags.DEFINE_string("run_name", "collecting_acts", "name for saving")
flags.DEFINE_boolean("save_model", True, "weather or not save the model")
flags.DEFINE_integer("n_hidden", 100, "hidden units in LSTM layer")
flags.DEFINE_integer("n_middle", 100, "hidden units in FC layer")
flags.DEFINE_integer("n_epochs", 200, "hidden units in FC layer")
flags.DEFINE_integer("batch_size", 128, "hidden units in FC layer")
flags.DEFINE_boolean("restore", False, "Inference?")
FLAGS = flags.FLAGS

tidigits = load_dataset(FLAGS.data_path)

# params
eta = 1e-3
display_step = 5


# Net Params
n_input = 39
n_steps = tidigits.train.max_len
n_out = 10
ts = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
run_name = 'tidigits_hid_{}_middle_{}_epo_{}_b_{}_{}'.format(FLAGS.n_hidden,
                                                             FLAGS.n_middle,
                                                             FLAGS.n_epochs,
                                                             FLAGS.batch_size,
                                                             ts)


def RNN(_X, _weights, _biases, _keep_prob):

    summary_act = []
    # Rearrange the input because tf want s a list of time steps
    # _X = tf.transpose(_X, [1, 0, 2])  # now the input is n_steps, batch_size
    # _X = tf.reshape(_X, [-1, n_input])  # batch_size * n_steps, n_input
    # _X = tf.split(0, n_steps, _X)  # n_steps * (batch_size, n_hidden) # Ii is like having a list of steps

    # cell = LSTMCell(FLAGS.n_hidden, use_peepholes=True, state_is_tuple=True)
    cell = tf.nn.rnn_cell.GRUCell(FLAGS.n_hidden)

    # Get lstm cell output
    outputs, outputs_fw = tf.nn.dynamic_rnn(cell, _X, dtype=tf.float32)
    # outputs, states = tf.nn.bidirectional_rnn(cell_fw=cell, cell_bw=cell, inputs=_X, dtype=tf.float32)

    # summary_act.append(tf.summary.histogram("out_bi", tf.reshape(tf.concat(1, outputs), [-1, 2 * FLAGS.n_hidden])))

    # gru_cell = tf.nn.rnn_cell.GRUCell(n_hidden)
    # cell = MultiRNNCell([lstm_cell] * 2, state_is_tuple=True)
    # run the lstm
    # outputs_2, states = tf.nn.rnn(gru_cell, outputs, dtype=tf.float32)

    # summary_act.append(tf.summary.histogram("out_second", tf.reshape(tf.concat(1, outputs_2), [-1, n_hidden])))

    # we manually apply dropout with this wrapper
    # output_fw, output_bw = outputs
    # states_fw, states_bw = states
    # out = tf.concat(2, [output_fw, output_bw])
    output = tf.transpose(outputs, [1, 0, 2])
    last = tf.gather(output, int(output.get_shape()[0]) - 1)
    out_drop = tf.nn.dropout(last, _keep_prob)
    # we manually create a FCL
    hid = tf.nn.relu(tf.matmul(out_drop, _weights['hidden']) + _biases['hidden'])
    summary_act.append(tf.summary.histogram("out_FC", hid))

    hid_drop = tf.nn.dropout(hid, _keep_prob)
    return tf.matmul(hid_drop, _weights['out']) + _biases['out'], summary_act


def main(_):

    # inputs
    x = tf.placeholder(tf.float32, [None, n_steps, n_input])

    # for dropout
    keep_prob = tf.placeholder(tf.float32)

    # labels
    y = tf.placeholder(tf.float32, [None, n_out])

    # weights from input to hidden
    weights = {
        'input': tf.Variable(tf.random_normal([n_input, 2 * FLAGS.n_middle], dtype=tf.float32)),
        'hidden': tf.Variable(tf.random_normal([FLAGS.n_hidden, FLAGS.n_middle], dtype=tf.float32)),
        'out': tf.Variable(tf.random_normal([FLAGS.n_middle, n_out], dtype=tf.float32))
    }

    biases = {
        'input': tf.Variable(tf.random_normal([2 * FLAGS.n_middle], dtype=tf.float32)),
        'hidden': tf.Variable(tf.random_normal([FLAGS.n_middle], dtype=tf.float32)),
        'out': tf.Variable(tf.random_normal([n_out], dtype=tf.float32))
    }

    # Register weights to be monitored by tensorboard
    w_hid_hist = tf.summary.histogram("weights_hid", weights['hidden'])
    b_hid_hist = tf.summary.histogram("biases_hid", biases['hidden'])
    w_out_hist = tf.summary.histogram("weights_out", weights['out'])
    b_out_hist = tf.summary.histogram("biases_out", biases['out'])

    # Let's define the training and testing operations
    print "Compiling RNN...",
    predictions, summ_acts = RNN(x, weights, biases, keep_prob)
    print "DONE!"

    # built-in cost function
    print "Compiling cost functions...",
    #
    step0 = tf.nn.softmax(predictions)
    # step1 = y * tf.log(step0)
    # step2 = -tf.reduce_sum(step1, reduction_indices=[1])
    # cross_entropy = tf.reduce_mean(step2)
    cost = tf.reduce_mean(
        -tf.reduce_sum(y * tf.log(tf.clip_by_value(step0, 1e-10, 1.0)), reduction_indices=[1]))
    print "DONE!"
    # manual cost function
    # softmax_lstm = tf.nn.softmax(predictions)
    # cost = -tf.reduce_mean(y * tf.log(softmax_lstm))

    # manual dropout
    # h_drop = tf.nn.dropout(pred, keep_prob)

    # register cost to be monitored by tensorboard, unfortunately we need to specify two summaries that look at the
    # same variable and update them at different times if we want to monitor both training and test performance with the
    # same function in the graph
    cost_summary = tf.summary.scalar("cost", cost)
    cost_val_summary = tf.summary.scalar("cost_val", cost)

    # we define the optimizer, not far from Theano
    print "Calculating gradients...",
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    print "DONE!"
    # evaluation
    correct_pred = tf.equal(tf.argmax(predictions, 1), tf.arg_max(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    accuracy_summary = tf.summary.scalar("accuracy", accuracy)
    accuracy_val_summary = tf.summary.scalar("accuracy_val", accuracy)

    # we need to tell tensorflow to initilize all variable and so we define an operation in the graph which is
    # initialize and we will run it later on
    init = tf.initialize_all_variables()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15)
    # Save & Restore
    saver = tf.train.Saver()

    # this does not have to be like that but can be simpler
    # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    with tf.Session() as sess:
        # initialize
        print "Initializing variables...",
        sess.run(init)
        print "DONE!"
        # definition of the object that will write our summaries
        writer = tf.summary.FileWriter("tidigits_run/{}".format(FLAGS.run_name), sess.graph)
        merged_weights = tf.summary.merge([w_hid_hist, w_out_hist, b_out_hist, b_hid_hist] + summ_acts)
        if not FLAGS.restore:

            # training loop
            for step in range(FLAGS.n_epochs):

                # epoch
                for i in range(tidigits.train.num_examples / FLAGS.batch_size + 1):
                    batch_xs, batch_ys, _, _, _ = tidigits.train.next_batch(FLAGS.batch_size)
                    # run evaluates the graph exactly like Theano
                    sess.run(optimizer, feed_dict={x: batch_xs,
                                                   y: batch_ys,
                                                   keep_prob: 0.5})

                # training accuracy
                batch_xs, batch_ys, _, _, _ = tidigits.train.full_batch()

                result = sess.run([merged_weights, accuracy_summary, cost_summary, accuracy, cost],
                                  feed_dict={x: batch_xs,
                                             y: batch_ys,
                                             keep_prob: 1.0})

                writer.add_summary(result[0], step)
                writer.add_summary(result[1], step)
                writer.add_summary(result[2], step)
                acc_val = result[3]
                loss_val = result[4]

                # test accuracy
                test_xs, test_ys, _, _, _ = tidigits.test.full_batch()
                acc_val_sum, cost_val_sum, loss_test, acc_test = sess.run([accuracy_val_summary,
                                                                           cost_val_summary,
                                                                           cost,
                                                                           accuracy],
                                                                          feed_dict={x: test_xs,
                                                                                     y: test_ys,
                                                                                     keep_prob: 1.0})

                writer.add_summary(acc_val_sum, step)
                writer.add_summary(cost_val_sum, step)

                table = [["Train", loss_val, acc_val],
                         ["Test", loss_test, acc_test]]
                headers = ["Epoch={}".format(step), "Cost", "Accuracy"]

                print tabulate(table, headers, tablefmt='grid')

                # Save the variables to disk.
                # if FLAGS.save_model:
                #     save_path = saver.save(sess, "/tmp/model_{}.ckpt".format(run_name), global_step=step)
                #     print("Model saved in file: %s" % save_path)
                #     savemat('weights_TI.mat', {'W': weights['hidden'].eval(sess)})

        else:
            # Restore variables from disk.
            saver.restore(sess, "/tmp/model.ckpt")
            print("Model restored.")

            # test accuracy
            test_xs, test_ys = tidigits.test.full_batch()
            acc_val_sum, cost_val_sum, loss_test, acc_test = sess.run([accuracy_val_summary,
                                                                       cost_val_summary,
                                                                       cost,
                                                                       accuracy],
                                                                      feed_dict={x: test_xs,
                                                                                 y: test_ys,
                                                                                 keep_prob: 1.0})

            table = [["Test", loss_test, acc_test]]
            headers = ["Epoch={}".format("~"), "Cost", "Accuracy"]

            print tabulate(table, headers, tablefmt='grid')


if __name__ == "__main__":
    tf.app.run()

