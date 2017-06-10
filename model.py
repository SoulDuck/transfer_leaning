import tensorflow as tf
import cache

def convolution2d(name,x,out_ch,k=3 , s=2 , padding='SAME'):
    with tf.variable_scope(name) as scope:
        in_ch=x.get_shape()[-1]
        filter=tf.get_variable("w" , [k,k,in_ch , out_ch] , initializer=tf.contrib.layers.xavier_initializer())
        bias=tf.Variable(tf.constant(0.1) , out_ch)
        layer=tf.nn.conv2d(x , filter ,[1,s,s,1] , padding)+bias
        layer=tf.nn.relu(layer , name='relu')
        if __debug__ == True:
            print 'layer shape : ' ,layer.get_shape()

        return layer

def affine(name,x,out_ch ,keep_prob):
    with tf.variable_scope(name) as scope:
        if len(x.get_shape())==4:
            batch, height , width , in_ch=x.get_shape().as_list()
            w_fc=tf.get_variable('w' , [height*width*in_ch ,out_ch] , initializer= tf.contrib.layers.xavier_initializer())
            x = tf.reshape(x, (-1, height * width * in_ch))
        elif len(x.get_shape())==2:
            batch, in_ch = x.get_shape().as_list()
            w_fc=tf.get_variable('w' ,[in_ch ,out_ch] ,initializer=tf.contrib.layers.xavier_initializer())

        b_fc=tf.Variable(tf.constant(0.1 ), out_ch)
        layer=tf.matmul(x , w_fc) + b_fc

        layer=tf.nn.relu(layer)
        layer=tf.nn.dropout(layer , keep_prob)
        print 'layer name :'
        print 'layer shape :',layer.get_shape()
        print 'layer dropout rate :',keep_prob
        return layer
def gap(name,x , n_classes ):
    in_ch=x.get_shape()[-1]
    gap_x=tf.reduce_mean(x, (1,2))
    with tf.variable_scope(name) as scope:
        gap_w=tf.get_variable('w' , shape=[in_ch , n_classes] , initializer=tf.random_normal_initializer(0,0.01) , trainable=True)
    y_conv=tf.matmul(gap_x, gap_w , name='y_conv')

def algorithm(y_conv, y_, learning_rate):
    """
    :param y_conv: logits
    :param y_: labels  type : one hot vector
    :param learning_rate: learning rate
    :return:  pred,pred_cls , cost , correct_pred ,accuracy
    """
    if __debug__ == True:
        print y_conv.get_shape()
        print y_.get_shape()

    pred = tf.nn.softmax(y_conv, name='softmax')
    pred_cls = tf.argmax(pred, axis=1, name='pred_cls')
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_), name='cost')
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    correct_pred = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1), name='correct_pred')
    accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32), name='accuracy')

    return pred, pred_cls, cost, train_op, correct_pred, accuracy