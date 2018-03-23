import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# dir = 'C:/Users/lsy-641/PycharmProjects/cat_vs_dog/train'
# BATCH_SIZE = 2
# CAPACITY = 256
# W = 208
# H = 208

def get_file(dir):
    cats = []
    label_cats= []
    dogs = []
    label_dogs = []
    for file in os.listdir(dir):
        name = file.split(sep='.')
        if 'cat' in name[0]:
            cats.append(dir +'/'+ file)
            label_cats.append(1)
        else:
            if 'dog' in name[0]:
                dogs.append(dir +'/' + file)
                label_dogs.append(0)

    image_list = np.hstack( ( cats ,dogs ) )
    label_list = np.hstack( ( label_cats , label_dogs ) )
    temp = np.array([image_list , label_list])
    temp = temp.transpose()#转置
    np.random.shuffle(temp)

    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(i) for i in label_list]


    return image_list , label_list

# 测试 get_files
#imgs , label = get_file(dir)
#j=0
#for i in imgs  :
#    if j< 3:
#        print("img:",i)
#    j+=1


def get_batch(image , label , W , H ,batch_size, capacity):
    image = tf.cast(image , tf.string)
    # label1 = label
    label = tf.cast(label , tf.int32)
    input_queue = tf.train.slice_input_producer([image , label])

    label = input_queue[1]
    # label= tf.placeholder(tf.int32, shape=[None, 2], name='Placeholder')
    # i = 0
    # for l in label1:
    #     if l == 1:
    #         tf.add_n([1,0])
    #         # tf.concat(0, [label1, [[1,0]]])
    #     else:
    #         tf.add_n([0, 1])
    #         # tf.concat(0, [label1, [[0, 1]]])
    #     i+=1

    image_contents = tf.read_file(input_queue[0]) #读图片

    image = tf.image.decode_jpeg(image_contents, channels=3) #解码

    image = tf.image.resize_images(image, [H, W], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#不知道为什么不能标准化 ，标准化后出现负数
#   image = tf.div(image,255)
#    image = tf.image.per_image_standardization(image)
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=32, capacity=capacity)
    label_batch = tf.reshape(label_batch , [batch_size])
    image_batch = tf.cast(image_batch , tf.float32)

    return image_batch ,label_batch

#
# image_list, label_list = get_file(dir)
# image_batch, label_batch = get_batch(image_list, label_list, W, H, BATCH_SIZE, CAPACITY)
#
# with tf.Session() as sess:#coordinator 和 start_queue_runners监控queue状态，不停出对入队
#     i = 0
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord = coord)
#
#     try:
#         while not coord.should_stop() and i<1:
#             image , label = sess.run( [image_batch , label_batch] )
#             for j in np.arange(BATCH_SIZE) :
#                 print('label : %d '% label[j])
# #              image[j] = np.asarray(image[j]), dtype='uint8')
#                 plt.imshow(image[j,:,:,:])
#                 print(image[j,:,:,:])
#                 plt.show()
#             i+=1
#     except tf.errors.OutOfRangeError:
#         print('done!')
#     finally :
#         coord.request_stop( )
#     coord.join(threads)
#     sess.close()
#
#
