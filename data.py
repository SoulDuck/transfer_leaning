import numpy as np

def get_cache_data_normal_vs_abnormal(normal_cache_data , abnormal_cache_data ,train_ratio=0.95):

    n_normal=len(normal_cache_data)
    n_abnormal=len(abnormal_cache_data)

    n_train_normal=n_normal/0.95
    n_test_normal=n_normal-n_train_normal
    n_train_abnormal = n_abnormal / 0.95
    n_test_abnormal = n_abnormal - n_train_abnormal

    train_normal_indics=random.sample(range(n_normal) , n_train_normal)
    test_normal_indics = random.sample(range(n_normal), n_train_normal)

    train_abnormal_indics=random.sample(range(n_abnormal) , n_train_abnormal)
    test_abnormal_indics = random.sample(range(n_abnormal), n_train_abnormal)

    if __debug__== True:
        print '########noraml############'
        print 'the number of normal :' , n_normal
        print 'the number of train normal :', n_train_normal
        print 'the number of test normal :', n_test_normal

        print '########abnoraml##########'
        print 'the number of abnormal :', n_abnormal
        print 'the number of train abnormal :', n_train_abnormal
        print 'the number of test abnormal :', n_test_abnormal

        print '##########################'
        print 'the total of trianing data:',n_train_normal + n_train_abnormal
        print 'the total of trianing data:',n_test_normal + n_test_abnormal
    return train_data ,train_labels , test_data, test_labels


