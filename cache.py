import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os,sys
import pickle
import PIL
import glob
from PIL import Image

def show_progress(i,max_iter):
    msg='\r Progress {0}/{1}'.format(i,max_iter)
    sys.stdout.write(msg)
    sys.stdout.flush()

def make_numpy_images(paths):

    tmp=[]
    for i,path in enumerate(paths):
        try:
            img=Image.open(path)
        except IOError as ioe:
            continue
        img=np.asarray(img)
        if i==0:
            print np.shape(np.shape(img))
        show_progress(i, len(paths))
        #print np.shape(img)
        tmp.append(img)
    imgs=np.asarray(tmp)

    if __debug__==True:
        print 'images shape :',np.shape(imgs)
    return imgs

def get_caches(sess, tensor_input_image ,  tensor_transfer_layer,images):

    if len(np.shape(images)) ==3:
        h,w,ch=np.shape(images)
        images=images.reshape([1,h,w,ch])
        print np.shape(images)
    tmp=[]
    for i,image in enumerate(images):
        show_progress(i,len(images))
        feed_dict = {tensor_input_image: image}
        cache=sess.run(tensor_transfer_layer , feed_dict=feed_dict )
        cache=np.squeeze(cache)
        tmp.append(cache)
    caches=np.asarray(tmp)
    return caches

def save_and_restore_cache(cache , save_path):
    name = save_path.split('/')[-1]
    if os.path.exists(save_path):
        with open(save_path, mode='rb') as file:
            cache = pickle.load(file)

        print("- Data loaded from : " + save_path)
        return cache
    else:
        # The cache-file does not exist.

        # Call the function / class-init with the supplied arguments.

        # Save the data to a cache-file.
        with open(save_path, mode='wb') as file:
            pickle.dump(cache, file)

        print("- Cache Data saved to : " + save_path)
    return cache
def restore_graph( graph_def_path):
    graph = tf.Graph()
    with graph.as_default():
        with tf.gfile.FastGFile(graph_def_path, 'rb') as file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file.read())
            tf.import_graph_def(graph_def, name='')

    tensor_input_image=graph.get_tensor_by_name("DecodeJpeg:0")
    tensor_softmax=graph.get_tensor_by_name('softmax:0')
    tensor_cost=graph.get_tensor_by_name('softmax/logits:0')
    tensor_transfer_layer=graph.get_tensor_by_name('pool_3:0')
    tensor_resized_imgae=graph.get_tensor_by_name('ResizeBilinear:0')
    tensor_top_conv=graph.get_tensor_by_name('mixed_10/join:0')
    sess=tf.Session(graph=graph)

    print 'graph restore succeed!'
    return sess, tensor_input_image,tensor_softmax , tensor_cost , tensor_resized_imgae , tensor_transfer_layer , tensor_top_conv

def check_cache(cache):
    reshape_cache = cache.reshape([32, 64])
    plt.imshow(img)
    plt.show()
    plt.imshow(reshape_cache)
    plt.show()
    conv_cache = get_caches(sess, tensor_input_image, tensor_top_conv, img)
    plt.imshow(conv_cache[:, :, 0])
    plt.show()
    print np.shape(cache)
    cache = save_and_restore_cache(cache, './cache.pkl')
    print np.shape(cache)

def_graph_path = './pretrained_model/classify_image_graph_def.pb'
sess, tensor_input_image, tensor_softmax, tensor_cost, tensor_resized_imgae, tensor_transfer_layer, tensor_top_conv = restore_graph(
    def_graph_path)



if __name__ =='__main__':
    folder_path = '/home/mediwhale/data/eye/resize_eye/normal/';extension = '*.png'
    target_paths = glob.glob(folder_path + extension)


    images = make_numpy_images(target_paths[:10000] )
    caches = get_caches(sess, tensor_input_image, tensor_transfer_layer, images)
    np.save('/home/mediwhale/data/eye/cache/normal/normal_caches_1.npy' , caches )
    caches=np.load('/home/mediwhale/data/eye/cache/normal/normal_caches_1.npy')
    print np.shape(caches)

    images = make_numpy_images(target_paths[10000:20000])
    caches = get_caches(sess, tensor_input_image, tensor_transfer_layer, images)
    np.save('/home/mediwhale/data/eye/cache/normal/normal_caches_2.npy', caches)
    caches = np.load('/home/mediwhale/data/eye/cache/normal/normal_caches_2.npy')
    print np.shape(caches)


    images = make_numpy_images(target_paths[20000:])
    caches = get_caches(sess, tensor_input_image, tensor_transfer_layer, images)
    np.save('/home/mediwhale/data/eye/cache/normal/normal_caches_3.npy', caches)
    caches = np.load('/home/mediwhale/data/eye/cache/normal/normal_caches_3.npy')
    print np.shape(caches)

    """
    images=make_numpy_images(folder_path,extension)
    img=Image.open('./sample_image/79101_20130730_L.png')
    img=np.asarray(img)
    cache=get_caches(sess, tensor_input_image , tensor_transfer_layer , img)
    caches=get_caches(sess, tensor_input_image , tensor_transfer_layer , images)
    print np.shape(caches)
    check_cache(cache)
    """