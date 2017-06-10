import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import PIL
from PIL import Image
def get_cache(sess, tensor_input_image ,  tensor_transfer_layer,image):
    feed_dict = {tensor_input_image: image}
    cache=sess.run(tensor_transfer_layer , feed_dict=feed_dict )
    cache=np.squeeze(cache)
    return cache

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

if __name__ =='__main__':

    img=Image.open('./sample_image/79101_20130730_L.png')
    def_graph_path='/Users/seongjungkim/PycharmProjects/transfer_leaning_git/Pretrained_Model/classify_image_graph_def.pb'
    sess, tensor_input_image, tensor_softmax, tensor_cost, tensor_resized_imgae, tensor_transfer_layer , tensor_top_conv=restore_graph(def_graph_path)
    cache=get_cache(sess, tensor_input_image , tensor_transfer_layer , img)
    reshape_cache=cache.reshape([32,64])
    plt.imshow(img)
    plt.show()
    plt.imshow(reshape_cache)
    plt.show()
    conv_cache=get_cache(sess, tensor_input_image , tensor_top_conv , img)
    plt.imshow(conv_cache[:,:,0])
    plt.show()
    print np.shape(cache)
    cache=save_and_restore_cache(cache,'./cache.pkl')
    print np.shape(cache)
