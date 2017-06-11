import numpy as np
import random

def cls2onehot(cls, depth):
    labels=np.zeros([len(cls),2])
    for i,ind in enumerate(cls):

        labels[i][ind:ind+1]=1
    if __debug__==True:
        print 'show sample cls and converted labels'
        print cls[:10]
        print labels[:10]
        print cls[-10:]
        print labels[-10:]
    return labels
def get_cache_data_normal_vs_abnormal(normal_cache_data , abnormal_cache_data ,train_ratio=0.95):


    NORMAL_INDEX=0
    ABNORMAL_INDEX=1

    n_normal=len(normal_cache_data)
    n_abnormal=len(abnormal_cache_data)
    n_train_normal=int(float(n_normal)*train_ratio)
    n_test_normal=n_normal-n_train_normal
    n_train_abnormal = int(float(n_abnormal) * train_ratio)
    n_test_abnormal = n_abnormal - n_train_abnormal

    train_normal_indics=random.sample(range(n_normal) , n_train_normal)
    test_normal_indics = random.sample(range(n_normal), n_train_normal)
    train_normal_caches=normal_cache_data[train_normal_indics]
    test_normal_caches=normal_cache_data[test_normal_indics]

    train_abnormal_indics=random.sample(range(n_abnormal) , n_train_abnormal)
    test_abnormal_indics = random.sample(range(n_abnormal), n_train_abnormal)
    train_abnormal_caches = normal_cache_data[train_abnormal_indics]
    test_abnormal_caches = normal_cache_data[test_abnormal_indics]

    train_data=np.concatenate((train_normal_caches , train_abnormal_caches),axis=0)
    test_data=np.concatenate((test_normal_caches , test_abnormal_caches),axis=0)

    normal_labels = np.zeros([n_normal])
    normal_labels=normal_labels.astype(np.int32)
    abnormal_labels = np.ones([n_abnormal])
    abnormal_labels=abnormal_labels.astype(np.int32)

    train_normal_labels=normal_labels[:n_train_normal]
    test_normal_labels = normal_labels[n_train_normal:]
    train_abnormal_labels = abnormal_labels[:n_train_abnormal]
    test_abnormal_labels = abnormal_labels[n_train_abnormal:]
    train_labels=np.concatenate((train_normal_labels,train_abnormal_labels))
    test_labels = np.concatenate((test_normal_labels, test_abnormal_labels))

    if __debug__== True:
        print 'NORMAL LABEL :0'
        print 'ABNORMAL LABEL :1'
        print '########data##############'
        print '########noraml############'
        print 'the number of normal :' , n_normal
        print 'the number of train normal :', n_train_normal
        print 'the number of test normal :', n_test_normal
        print 'the shape of train normal caches :',train_normal_caches.shape
        print 'the shape of test normal caches :',test_normal_caches.shape
        print '########abnoraml##########'
        print 'the number of abnormal :', n_abnormal
        print 'the number of train abnormal :', n_train_abnormal
        print 'the number of test abnormal :', n_test_abnormal
        print 'the shape of train abnormal caches :',train_abnormal_caches.shape
        print 'the shape of test abnormal caches :',test_abnormal_caches.shape
        print '##########################'
        print 'the total of trianing data:',n_train_normal + n_train_abnormal
        print 'the total of test data:',n_test_normal + n_test_abnormal
        print 'the shape of training data shape:' ,train_data.shape
        print 'the shape of test data shape:' ,test_data.shape
        print 'the shape of training labels shape:', train_labels.shape
        print 'the shape of test labels shape:', test_labels.shape

    return train_data ,train_labels , test_data, test_labels


if __name__== '__main__':
    normal_cache_data=np.load('/home/mediwhale/data/eye/cache/normal/normal_caches_1.npy')
    abnormal_cache_data = np.load('/home/mediwhale/data/eye/cache/abnormal/glaucoma/galucoma_caches.npy')

    get_cache_data_normal_vs_abnormal(normal_cache_data , abnormal_cache_data)

