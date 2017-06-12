import numpy as np
import tensorflow as tf
import gpfunctions as gp


def get_lstm_weights(n_hidden, forget_bias, dim):      
    # Create LSTM cell
    cell = tf.contrib.rnn.LSTMCell(num_units = n_hidden, reuse=None, forget_bias = forget_bias)
    cell(tf.zeros([1, dim +1]), (tf.zeros([1, n_hidden]),tf.zeros([1, n_hidden])), scope='rnn_cell')
    cell = tf.contrib.rnn.LSTMCell(num_units = n_hidden, reuse=True, forget_bias = forget_bias)

    # Create output weights
    weights = {
        'W_1': tf.Variable(tf.truncated_normal([n_hidden, dim], stddev=0.05)),
        'b_1': tf.Variable(0.1*tf.ones([dim])),
    }

    return cell, weights

def apply_lstm_model(f, cell, weights, n_steps, dim, n_hidden, batch_size):
    
    x_0 = -0.0*tf.ones([batch_size, dim])
    h_0 = tf.zeros([batch_size, n_hidden])
    c_0 = tf.zeros([batch_size, n_hidden])

    state = (c_0, h_0)
    x = x_0
    y = f(x)
    samples_x = [x]
    samples_y = [y]

    for i in range(n_steps):
        h, state = cell(tf.concat([x, y], 1), state, scope='rnn_cell')
        x = tf.tanh(tf.matmul(h, weights['W_1']) + weights['b_1'])
        y = f(x)

        samples_x.append(x)
        samples_y.append(y)

    return samples_x, samples_y

def build_training_graph(n_bumps, dim, n_hidden, forget_bias, n_steps, l):
    # Create Model
    Xt = tf.placeholder(tf.float32, [None, n_bumps, dim])
    At = tf.placeholder(tf.float32, [None, n_bumps, 1])
    mint = tf.placeholder(tf.float32, [None, 1])
    maxt = tf.placeholder(tf.float32, [None, 1])

    f = lambda x: gp.normalize(mint, maxt, gp.GPTF(Xt, At, x, l)) 
    
    cell, weights = get_lstm_weights(n_hidden, forget_bias, dim)

    samples_x, samples_y = apply_lstm_model(f, cell, weights, n_steps, dim, n_hidden, tf.shape(Xt)[0])

    return Xt, At, mint, maxt, samples_x, samples_y, cell, weights

def get_loss(samples_y, loss_type):
    loss_dict = {"MIN" : lambda x : tf.reduce_mean(tf.reduce_min(x, axis = 0)), 
                 "SUM" : lambda x : tf.reduce_mean(tf.reduce_sum(x, axis = 0)),
                 "WSUM" : lambda x : \
                 tf.reduce_mean(tf.reduce_sum(tf.multiply(x, np.linspace(1/(n_steps+1),1, n_steps+1)), axis = 0)),
                 "EI" : lambda x : tf.reduce_mean(tf.reduce_sum(x, axis = 0)) -\
                 tf.reduce_mean(tf.reduce_sum([tf.reduce_min(x[:i+1],\
                                                             axis = 0) for i in range(n_steps)], axis = 0)),
                 "SUMMIN" : lambda x : tf.reduce_mean(tf.reduce_min(x, axis = 0)) +\
                 tf.reduce_mean(tf.reduce_sum(x, axis = 0))
                }

    return loss_dict[loss_type](samples_y)

def get_min(samples_y):
    return tf.reduce_mean(tf.reduce_min(samples_y, axis = 0))

def get_train_step(loss, gradient_clipping):
    rate = tf.placeholder(tf.float32, [])

    optimizer = tf.train.AdamOptimizer(learning_rate=rate)
    gvs = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(grad, -gradient_clipping, gradient_clipping), var) for grad, var in gvs]
    train_step = optimizer.apply_gradients(capped_gvs)

    return train_step, rate

def train_model(sess, placeholders, samples_y, epochs, batch_size, data_train, data_test, rate_init, rate_decay, gradient_clipping, \
                loss_type, log = True): 
    
    X_train, A_train, min_train, max_train = data_train
    X_test, A_test, min_test, max_test = data_test
    n_train = X_train.shape[0]
    
    Xt = placeholders["Xt"]
    At = placeholders["At"]
    mint = placeholders["mint"]
    maxt = placeholders["maxt"]
    
    loss = get_loss(samples_y, loss_type)

    f_min = get_min(samples_y)

    train_step, train_rate = get_train_step(loss, gradient_clipping)
    
    if log:
        train_loss_list = []
        test_loss_list = []
        train_fmin_list = []
        test_fmin_list = []

    learning_rate = rate_init
    
    sess.run(tf.global_variables_initializer())
    for ep in range(epochs):
        learning_rate *= rate_decay
        
        for batch in range(n_train//batch_size):
            X_batch = X_train[batch*batch_size:(batch+1)*batch_size]
            A_batch = A_train[batch*batch_size:(batch+1)*batch_size]
            min_batch = min_train[batch*batch_size:(batch+1)*batch_size]
            max_batch = max_train[batch*batch_size:(batch+1)*batch_size]

            sess.run([train_step],\
                     feed_dict={Xt: X_batch, At: A_batch, mint: min_batch, maxt: max_batch,\
                                train_rate: learning_rate})

        if log:
            train_loss, train_fmin = sess.run([loss, f_min], feed_dict=\
                                              {Xt: X_train, At: A_train, mint: min_train, maxt: max_train})
            test_loss, test_fmin = sess.run([loss, f_min], feed_dict=\
                                              {Xt: X_test, At: A_test, mint: min_test, maxt: max_test})
            train_loss_list += [train_loss]
            test_loss_list += [test_loss]
            train_fmin_list += [train_fmin]
            test_fmin_list += [test_fmin]

        if log and (ep < 10 or ep % (epochs // 10) == 0 or ep == epochs-1):
            print("Ep: " +"{:4}".format(ep)+" | TrainLoss: "+"{: .3f}".format(train_loss)
                  +" | TrainMin: "+ "{: .3f}".format(train_fmin)+ " | TestLoss: "+
                  "{: .3f}".format(test_loss)+" | TestMin: "+ "{: .3f}".format(test_fmin))
    
    print("Done.")
    if log:
        return (train_loss_list, test_loss_list, train_fmin_list, test_fmin_list)
    return None