import tensorflow as tf
import numpy as np
import benchmarkfunctions as fun
import utils

def kernelTF(x1,x2,l = 0.3):
    return tf.exp(-1.0/l**2*tf.reduce_sum((tf.expand_dims(x1,axis=2) - tf.expand_dims(x2,axis=1))**2, axis = 3))

def GPTF(X,A,x, l = 0.3):
    k_xX = kernelTF(tf.expand_dims(x, axis = 1),X)
    return tf.squeeze(tf.matmul(k_xX,  A),axis=(2,))

def normalize(minv, maxv, y):
        return 2*(y-minv)/(maxv-minv)-1.0

def train_rnn_n2n(dim, n_steps = 10, learning_rate=0.001, epochs=1000, n_hidden = 50, batch_size = 160):
    # load data
    X_train, A_train, min_train, max_train = utils.loadData(dim, 'training')
    X_test, A_test, min_test, max_test = utils.loadData(dim, 'testing')

    n_gp_samples = X_train.shape[1]

    # define model
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, dim]))
    }

    biases = {
        'out': tf.Variable(tf.random_normal([dim]))
    }

    size = tf.placeholder(tf.int32,[])

    Xt = tf.placeholder(tf.float32, [None, n_gp_samples, dim])
    At = tf.placeholder(tf.float32, [None, n_gp_samples, 1])
    mint = tf.placeholder(tf.float32, [None, 1])
    maxt = tf.placeholder(tf.float32, [None, 1])

    x_0 = -0.0*tf.ones([size, dim])
    h_0 = tf.ones([size, n_hidden])
    c_0 = tf.ones([size, n_hidden])

    state = (c_0, h_0)
    x = x_0
    y = normalize(mint, maxt, GPTF(Xt,At,x))
    sample_points = [x]

    f_min = y
    f_sum = 0

    # No idea why this is necessary
    cell = tf.contrib.rnn.LSTMCell(num_units = n_hidden, reuse=None)
    cell(tf.concat([x, y], 1), state, scope='rnn_cell')
    cell = tf.contrib.rnn.LSTMCell(num_units = n_hidden, reuse=True)

    for i in range(n_steps):
        h, state = cell(tf.concat([x, y], 1), state, scope='rnn_cell')
        x = tf.tanh(tf.matmul(h, weights['out']) + biases['out'])
        sample_points.append(x)

        y = normalize(mint, maxt, GPTF(Xt,At,x))

        f_min = tf.minimum(y, f_min)
        f_sum += tf.reduce_mean(y)

    f_min = tf.reduce_mean(f_min)
    loss = f_sum / n_steps

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    train_loss_list = []
    test_loss_list = []
    train_fmin_list = []
    test_fmin_list = []

    # Train the Network
    print("------------------------------------------------------------------------------------")
    print("%-30s: %d" % ("Function Dimension", dim) )
    print("%-30s: %d" % ("Number of Training Samples", len(X_train)) )
    print("%-30s: %d" % ("Number of Test Samples", len(X_test)) )
    print("%-30s: %d" % ("Batch size", batch_size) )
    print("%-30s: %d" % ("Number of hidden Units", n_hidden) )
    print("%-30s: %d" % ("Sequence length", n_steps) )
    print("%-30s: %d" % ("Epochs",epochs) )
    print("%-30s: %.5f" % ("Learning rate", learning_rate) )
    print("------------------------------------------------------------------------------------")

    for ep in range(epochs):
        for batch in range(len(X_train)//batch_size):
            X_batch = X_train[batch*batch_size:(batch+1)*batch_size]
            A_batch = A_train[batch*batch_size:(batch+1)*batch_size]
            min_batch = min_train[batch*batch_size:(batch+1)*batch_size]
            max_batch = max_train[batch*batch_size:(batch+1)*batch_size]

            sess.run([train_step], feed_dict={Xt: X_batch, At: A_batch, mint: min_batch, maxt: max_batch, size: X_batch.shape[0]})

        train_loss, train_fmin = sess.run([loss, f_min], feed_dict=\
                                          {Xt: X_train, At: A_train, mint: min_train, maxt: max_train, size: len(X_train)})
        test_loss, test_fmin = sess.run([loss, f_min], feed_dict=\
                                          {Xt: X_test, At: A_test, mint: min_test, maxt: max_test, size:len(X_test)})

        train_loss_list += [train_loss]
        test_loss_list += [test_loss]
        train_fmin_list += [train_fmin]
        test_fmin_list += [test_fmin]

        if ep < 10 or ep % (epochs // 10) == 0 or ep == epochs-1:
            print("Ep: " +"{:4}".format(ep)+" | TrainLoss: "+"{: .3f}".format(train_loss)
                  +" | TrainMin: "+ "{: .3f}".format(train_fmin)+ " | TestLoss: "+
                  "{: .3f}".format(test_loss)+" | TestMin: "+ "{: .3f}".format(test_fmin))

if __name__ == "__main__":
    print("run as main")
    train_rnn_n2n(1, epochs=10)

