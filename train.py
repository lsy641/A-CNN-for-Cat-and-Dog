import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
import cat_vs_dog
import conv


N_CLASSES = 2  # 2个输出神经元，［1，0］ 或者 ［0，1］猫和狗的概率
IMG_W = 208  # 重新定义图片的大小，图片如果过大则训练比较慢
IMG_H = 208
BATCH_SIZE = 32  #每批数据的大小
CAPACITY = 256
MAX_STEP = 20000 # 训练的步数，应当 >= 10000
learning_rate = 0.0001 # 学习率，建议刚开始的 learning_rate <= 0.0001


def run_training():
    # 数据集
    dir = 'C:/Users/lsy-641/PycharmProjects/cat_vs_dog/train'   #My dir--20170727-csq
    #logs_train_dir 存放训练模型的过程的数据，在tensorboard 中查看
    logs_train_dir = 'C:/Users/lsy-641/PycharmProjects/cat_vs_dog/mid'
    # 获取图片和标签集
    train, train_label = cat_vs_dog.get_file(dir)
    # 生成批次
    train_batch, train_label_batch = cat_vs_dog.get_batch(train,
                                                          train_label,
                                                          IMG_W,
                                                          IMG_H,
                                                          BATCH_SIZE,
                                                          CAPACITY)
    #初始化参数
    parameters = conv.initialize_parameters()
    # 进入模型
    train_logits = conv.inference(train_batch, BATCH_SIZE, N_CLASSES,parameters)
    # 获取 loss
    train_loss = conv.losses(train_logits, train_label_batch)
    # 训练
    train_op = conv.trainning(train_loss, learning_rate)
    # 获取准确率
    train__acc = conv.evaluation(train_logits, train_label_batch)
    # 合并 summary
    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    # 保存summary
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()
    # saver.restore(sess, "C:\Users\lsy-641\PycharmProjects\cat_vs_dog\mid\model.ckpt-999")
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # tf.reset_default_graph()
    #下面两句是为了恢复模型
    model_file=tf.train.latest_checkpoint('mid/')
    saver.restore(sess,model_file)
    # imported_meta = tf.train.import_meta_graph("C:/Users/lsy-641/PycharmProjects/cat_vs_dog/mid/model.ckpt-999.meta")
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                    break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])

            if step % 100== 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step % 1000 == 0 or (step + 1) == MAX_STEP:
                # 每隔2000步保存一下模型，模型保存在 checkpoint_path 中
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()

# train
run_training()