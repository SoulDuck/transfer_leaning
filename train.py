import cache
import numpy as np
import tensorflow as tf
from model import convolution2d ,affine , gap , algorithm
import data



if __name__=='__main__':
    normal_cache_data=np.load('/home/mediwhale/data/eye/cache/normal/normal_caches_1.npy')
    abnormal_cache_data = np.load('/home/mediwhale/data/eye/cache/abnormal/glaucoma/galucoma_caches.npy')
    train_data ,train_cls, test_data , test_cls=data.get_cache_data_normal_vs_abnormal(normal_cache_data,abnormal_cache_data)
    train_labels=data.cls2onehot(train_cls , 2)
    test_labels = data.cls2onehot(test_cls, 2)

    #train_data=cache.save_and_restore_cache( '_' , save_path='./cache.pkl')
    #input_data=input_data.reshape([1,-1])
    #train_labels=np.asarray([[0,0]])

    x_=tf.placeholder(dtype=tf.float32 ,shape=[None , 2048])
    y_=tf.placeholder(dtype=tf.float32 , shape=[None  ,2 ])
    n_classes=2
    y_conv=affine('pretrain_fc' ,x_, n_classes , 0.5 )
    pred, pred_cls, cost, train_op, correct_pred, accuracy=algorithm(y_conv,y_,learning_rate=0.001)

    if __debug__==True:
        print 'y_conv shape : ', y_conv.get_shape()
        print 'n_classes : ', n_classes
        print 'input data shape ',np.shape(train_data)
        print 'input label shape',np.shape(train_labels)
    feed_dict={x_:train_data  , y_:train_labels}
    sess=tf.Session()
    init_op=tf.global_variables_initializer()
    sess.run(init_op,feed_dict=feed_dict)
    sess.run(train_op , feed_dict=feed_dict)
    pred_=sess.run(pred ,feed_dict=feed_dict)
    print pred_



