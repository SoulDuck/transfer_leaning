import cache
import numpy as np
import tensorflow as tf
from model import convolution2d ,affine , gap , algorithm




if __name__=='__main__':
    x_=tf.placeholder(dtype=tf.float32 ,shape=[None , 2048])
    y_=tf.placeholder(dtype=tf.float32 , shape=[None  ,2 ])
    n_classes=2
    y_conv=affine('pretrain_fc' ,x_, n_classes , 0.5 )
    pred, pred_cls, cost, train_op, correct_pred, accuracy=algorithm(y_conv,y_,learning_rate=0.001)

    labels=np.asarray([[0,0]])
    input_data=cache.save_and_restore_cache( '_' , save_path='./cache.pkl')
    input_data=input_data.reshape([1,-1])
    if __debug__==True:
        print 'y_conv shape : ', y_conv.get_shape()
        print 'n_classes : ', n_classes
        print 'input data shape ',np.shape(input_data)
        print 'input label shape',np.shape(labels)
    feed_dict={x_:input_data  , y_:labels}
    sess=tf.Session()
    init_op=tf.global_variables_initializer()
    sess.run(init_op,feed_dict=feed_dict)
    sess.run(train_op , feed_dict=feed_dict)
    pred_=sess.run(pred ,feed_dict=feed_dict)
    print pred_



