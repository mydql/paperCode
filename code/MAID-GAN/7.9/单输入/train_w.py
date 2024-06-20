import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.applications.vgg16 import VGG16
import cv2
import datetime
import math



import  os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


### ljy改5：限制显存
# gpus = tf.config.experimental.list_physical_devices('GPU')  # 获取GPU列表
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
#     # 失效： tf.config.experimental.set_per_process_memory_fraction(0.25)
#     # 第一个参数为原则哪块GPU，只有一块则是gpu[0],后面的memory_limt是限制的显存大小，单位为M
#     tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*12)])

from model import Sample1,Generator,Discriminator_2
from data import read_data,merge

tf.random.set_seed(234)

x_train = read_data('train.h5')
# x_train = x_train[0:1280]
x_train = x_train[0:10001]

x_validation = read_data('train.h5')
x_validation = x_validation[10001:11001]

x_test = read_data('test_lenna.h5')

x_train = tf.convert_to_tensor(x_train,dtype=tf.float32)/127.5 - 1.
x_validation = tf.convert_to_tensor(x_validation,dtype = tf.float32)/127.5 - 1.
x_test = tf.convert_to_tensor(x_test,dtype = tf.float32)/127.5 - 1.


train_db = tf.data.Dataset.from_tensor_slices(x_train)
train_db = train_db.shuffle(10000).batch(128)

validation_db = tf.data.Dataset.from_tensor_slices(x_validation)
validation_db = validation_db.batch(128)


test_db = tf.data.Dataset.from_tensor_slices(x_test)
test_db = test_db.batch(16)





def celoss_ones(logits):
    # [b, 1]
    # [b] = [1, 1, 1, 1,]
    #loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits))
    logits = tf.squeeze(logits)
    #loss = tf.keras.losses.categorical_crossentropy(y_pred=logits,y_true=tf.ones_like(logits))
    #return - tf.reduce_mean(logits)
    loss = tf.reduce_mean(tf.square(logits - tf.ones_like(logits)))
    return tf.reduce_mean(loss)



def celoss_zeros(logits):
    # [b, 1]
    # [b] = [1, 1, 1, 1,]
    #loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=tf.zeros_like(logits))
    logits = tf.squeeze(logits)
    #loss = tf.keras.losses.categorical_crossentropy(y_pred=logits, y_true=tf.zeros_like(logits))
    #return tf.reduce_mean(logits)
    loss = tf.reduce_mean(tf.square(logits-tf.zeros_like(logits)))
    return tf.reduce_mean(loss)


def gradient_penalty(discriminator, batch_x, fake_image):

    batchsz = batch_x.shape[0]

    # [b, h, w, c]
    t = tf.random.uniform([batchsz, 1, 1, 1])
    # [b, 1, 1, 1] => [b, h, w, c]
    t = tf.broadcast_to(t, batch_x.shape)

    interplate = t * batch_x + (1 - t) * fake_image

    with tf.GradientTape() as tape:
        tape.watch([interplate])
        d_interplote_logits = discriminator(interplate, training=True)
    grads = tape.gradient(d_interplote_logits, interplate)

    # grads:[b, h, w, c] => [b, -1]
    grads = tf.reshape(grads, [grads.shape[0], -1])
    gp = tf.norm(grads, axis=1) #[b]
    gp = tf.reduce_mean( (gp-1)**2 )

    return gp



# def d_loss_fn(sample,model1,discriminator,batch_x,training):
#     # 1. treat real image as real
#     # 2. treat generated image as fake
#     real_input1 = sample(batch_x)
#     fake_image = model1(real_input1, training)
#     fake_input1 = sample(fake_image)
#     real_inputs = (real_input1,batch_x)
#     fake_inputs = (fake_input1,fake_image)
#     # d_real_logits = discriminator(real_inputs)
#     # d_fake_logits = discriminator(fake_inputs)
#     d_real_logits = discriminator(batch_x)
#     d_fake_logits = discriminator(fake_image)
#
#
#     d_loss_real = celoss_ones(d_real_logits)
#     d_loss_fake = celoss_zeros(d_fake_logits)
#     #gp = gradient_penalty(discriminator, out, fake_image)
#
#     loss = (d_loss_real + d_loss_fake)/2.
#
#     return loss
#
#
# def g_loss_fn(sample,model1,discriminator,batch_x, training):
#     input1 = sample(batch_x)
#     fake_image = model1(input1, training)
#     input1 = sample(fake_image)
#     inputs = (input1,fake_image)
#     #d_fake_logits = discriminator(inputs)
#     d_fake_logits = discriminator(fake_image)
#     loss2 = celoss_ones(d_fake_logits)
#     fake_image = (fake_image+1)/2.
#     batch_x = (batch_x+1)/2.
#     loss1 = tf.reduce_mean(tf.abs(batch_x-fake_image))
#     mse = tf.reduce_mean(tf.square(batch_x-fake_image))
#
#     #print(tf.reduce_max(fake_image),tf.reduce_min(fake_image))
#
#
#
#     return loss1+0.01*loss2,mse,loss1,loss2


#以下是生成器中间层反馈输入到判别器
def d_loss_fn(sample,model1,discriminator,batch_x,training):
    # 1. treat real image as real
    # 2. treat generated image as fake
    real_input1 = sample(batch_x)
    fake_image,_ = model1(real_input1, training)
    d_real_logits = discriminator(batch_x)
    d_fake_logits = discriminator(fake_image)


    d_loss_real = celoss_ones(d_real_logits)
    d_loss_fake = celoss_zeros(d_fake_logits)
    #gp = gradient_penalty(discriminator, out, fake_image)

    loss = (d_loss_real + d_loss_fake)/2.

    return loss


def g_loss_fn(sample,model1,discriminator,batch_x, training):
    input1 = sample(batch_x)
    fake_image,_= model1(input1, training)
    d_fake_logits = discriminator(fake_image)
    loss2 = celoss_ones(d_fake_logits)
    fake_image = (fake_image+1)/2.
    batch_x = (batch_x+1)/2.
    #loss_vgg = VGGloss(batch_x,fake_image)
    loss1 = tf.reduce_mean(tf.abs(batch_x-fake_image))
    mse = tf.reduce_mean(tf.square(batch_x-fake_image))

    #print(tf.reduce_max(fake_image),tf.reduce_min(fake_image))



    return loss1+0.001*loss2,mse,loss1,loss2

# def VGGloss(true,fake):
#     mod = VGG16(include_top=False,weights='imagenet')
#     for layers in mod.layers:
#         layers.trainable = False
#     true = tf.concat([true,true,true],axis = -1)
#     fake = tf.concat([fake,fake,fake],axis = -1)
#     #print(true_out.shape)
#     out = tf.reduce_mean(tf.square(mod(true)-mod(fake)))
#     return out

def VGGloss(true,fake):
    mod = VGG16(include_top=False,weights='imagenet')
    true = tf.concat([true,true,true],axis = -1)
    fake = tf.concat([fake,fake,fake],axis = -1)
    #print(true_out.shape)
    out = tf.reduce_mean(tf.square(mod(true)-mod(fake)))
    return out


def main(train = 0):
    #超参数：
    learning_rate = 0.0001
    g_learning_rate = 0.0008
    d_learning_rate = 0.0008
    h_dim = int(96*96*0.05)
    #training = True
    best_loss = 1.
    best_loss_fc = 1
    train_best_loss = 1.
    train_best_loss_fc = 1

    sample = Sample1()

    model1 = Generator()

    # model = SwinTransformer( patch_size=4,
    #                  embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
    #                  window_size=7, mlp_ratio=4., qkv_bias=True,
    #                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
    #                  norm_layer=layers.LayerNormalization, name=None, )

    #model1.load_weights(r'./save_weights048/model1/model1.ckpt')
    #model2.load_weights(r'./save_weights048/model2/model2.ckpt')

    discriminator = Discriminator_2()





    optimizer_g =optimizers.RMSprop(learning_rate = learning_rate)
    #optimizer_d = optimizers.RMSprop(learning_rate = 0.00001)
    optimizer_d = optimizers.RMSprop(learning_rate=0.0001)






    #variables = sample.trainable_variables+model1.trainable_variables+model2.trainable_variables
    if train == 1:

        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        log_dir = 'logs/' + current_time
        summary_writer = tf.summary.create_file_writer(log_dir)

        #model1.load_weights('./save_weights048/save_weights.ckpt')

        # for epoch in range(51):
        #     training = True
        #
        #     mse_loss = []
        #     mean_loss1 = []
        #     for step, x in enumerate(train_db):
        #         with tf.GradientTape() as type:
        #             out = sample(x)
        #             out = model1(out,training)
        #             x = (x+1.)/2.
        #             out = (out+1.)/2.
        #             loss = tf.reduce_mean(tf.abs(x-out))
        #             mse = tf.reduce_mean(tf.square(x-out))
        #             mse_loss.append(mse)
        #             mean_loss1.append(loss)
        #         grads_g = type.gradient(loss,sample.trainable_variables+model1.trainable_variables)
        #
        #         optimizer_g.apply_gradients(zip(grads_g,sample.trainable_variables+model1.trainable_variables))
        #
        #     mse_loss = tf.reduce_mean(mse_loss)
        #     mean_loss1 = tf.reduce_mean(mean_loss1)
        #     print('epoch:', epoch, 'mse', float(mse_loss), 'loss1:', float(mean_loss1))
        #     with summary_writer.as_default():
        #         tf.summary.scalar('loss1', float(mean_loss1), step=epoch)
        #         tf.summary.scalar('mse', float(mse_loss), step=epoch)
        #
        #     if epoch % 2 == 0:
        #         validation_loss = []
        #         for step, x in enumerate(validation_db):
        #             out = sample(x)
        #             out = model1(out)
        #             out = (out+1)/2.
        #             x = (x+1)/2.
        #             loss = tf.reduce_mean(tf.square(out - x))
        #             validation_loss.append(loss)
        #         loss = tf.reduce_mean(validation_loss)
        #         psnr = (10 * (math.log10(1.0 / loss)))
        #         if float(loss) < best_loss:
        #             best_loss = float(loss)
        #             sample.save_weights(r'./save_weights048/paper/sample/sample.ckpt')
        #             model1.save_weights(r'./save_weights048/paper/model1/model1.ckpt')
        #         #out = (out-tf.reduce_min(out))/(tf.reduce_max(out)-tf.reduce_min(out))
        #         out = out*255.
        #         print('validation_loss:', float(loss), 'psnr:', psnr)
        #         cv2.imwrite('./recon/{}.png'.format(epoch), tf.squeeze(out[0]).numpy())
        #
        # for epoch in range(51,101):
        #     training = False
        #     d_mean_loss = []
        #     for step, x in enumerate(train_db):
        #         with tf.GradientTape() as type:
        #             loss_d = d_loss_fn(sample, model1, discriminator, x, training=training)
        #             # print(loss_d)
        #             d_mean_loss.append(loss_d)
        #         grads_d = type.gradient(loss_d, discriminator.trainable_variables)
        #
        #         optimizer_d.apply_gradients(zip(grads_d, discriminator.trainable_variables))
        #
        #     d_mean_loss = tf.reduce_mean(d_mean_loss)
        #     print('epoch:', epoch, 'mse', float(mse_loss), 'loss1:', float(mean_loss1),'loss-d:', float(d_mean_loss))
        #     with summary_writer.as_default():
        #         tf.summary.scalar('loss1', float(mean_loss1), step=epoch)
        #         tf.summary.scalar('mse', float(mse_loss), step=epoch)
        #         tf.summary.scalar('loss-d', float(d_mean_loss), step=epoch)
        #
        for epoch in range(1):
            for step, x in enumerate(train_db):
                with tf.GradientTape() as type:
                    loss_d = d_loss_fn(sample,model1,discriminator,x,training = True)
                    print('epoch:',epoch,loss_d)
                grads_d = type.gradient(loss_d,discriminator.trainable_variables)
                optimizer_d.apply_gradients(zip(grads_d, discriminator.trainable_variables))


        for epoch in range(301):
            g_mean_loss = []
            d_mean_loss = []
            mean_loss1 = []
            mean_loss2 = []
            for step, x in enumerate(train_db):
                with tf.GradientTape() as type:
                    loss_d = d_loss_fn(sample, model1, discriminator, x, training=True)
                    # print('epoch:', epoch, 'step:', step, 'lossd:', loss_d)
                    d_mean_loss.append(loss_d)

                grads_d = type.gradient(loss_d, discriminator.trainable_variables)
                optimizer_d.apply_gradients(zip(grads_d, discriminator.trainable_variables))

                with tf.GradientTape() as type1:
                    loss_g, mse, loss1, loss2 = g_loss_fn(sample, model1, discriminator, x, training=True)
                    # loss3 = tf.float16(loss3)
                    # print('loss1:', float(loss1), 'loss2:', float(loss2))
                    # mse_loss.append(mse)
                    g_mean_loss.append(loss_g)
                    mean_loss1.append(loss1)
                    mean_loss2.append(loss2)
                grads_g = type1.gradient(loss_g, sample.trainable_variables + model1.trainable_variables)

                optimizer_g.apply_gradients(zip(grads_g, sample.trainable_variables + model1.trainable_variables))

            g_mean_loss = tf.reduce_mean(g_mean_loss)
            d_mean_loss = tf.reduce_mean(d_mean_loss)
            mean_loss1 = tf.reduce_mean(mean_loss1)
            mean_loss2 = tf.reduce_mean(mean_loss2)

            # print(tf.norm(fc_net.trainable_variables[0]), tf.nn.l2_loss(fc_net.trainable_variables[0]))
            print('epoch:', epoch, 'loss-g:', float(g_mean_loss), 'loss1:', float(mean_loss1), 'loss2:', float(mean_loss2), 'loss-d:', float(d_mean_loss))
            with summary_writer.as_default():
                #tf.summary.scalar('mse', float(mse_loss), step=epoch)
                tf.summary.scalar('loss-g', float(g_mean_loss), step=epoch)
                tf.summary.scalar('loss1', float(mean_loss1), step=epoch)
                tf.summary.scalar('loss2', float(mean_loss2), step=epoch)
                tf.summary.scalar('loss-d', float(d_mean_loss), step=epoch)

            sample.save_weights(r'./last/sample/sample.ckpt')
            model1.save_weights(r'./last/model1/model1.ckpt')
            discriminator.save_weights(r'./last/discriminator/discriminator.ckpt')

            if epoch % 5 == 0:
                validation_loss = []
                for x in validation_db:
                    out = sample(x)
                    out,_ = model1(out)
                    out = (out+1)/2.
                    x = (x+1)/2.
                    loss = tf.reduce_mean(tf.square(out - x))
                    validation_loss.append(loss)
                loss = tf.reduce_mean(validation_loss)
                psnr = (10 * (math.log10(1.0 / loss)))
                if float(loss) < best_loss:
                    best_loss = float(loss)
                    sample.save_weights(r'./save_weights/sample/sample.ckpt')
                    model1.save_weights(r'./save_weights/model1/model1.ckpt')
                    discriminator.save_weights(r'./save_weights/discriminator/discriminator.ckpt')
                #out = (out-tf.reduce_min(out))/(tf.reduce_max(out)-tf.reduce_min(out))
                out = out*255.
                print('validation_loss:', float(loss), 'psnr:', psnr)
                # cv2.imwrite('./recon/{}.png'.format(epoch), tf.squeeze(out[0]).numpy())






    else:
        sample.load_weights(r'./save_weights/sample/sample.ckpt')
        model1.load_weights(r'./save_weights/model1/model1.ckpt')


        for x in test_db:
            out = sample(x)
            out, _ = model1(out, training=False)
            out = (out + 1) / 2.
            x = (x + 1) / 2.
            out = merge(out,[4,4])
            x  = merge(x,[4,4])

            loss = tf.reduce_mean(tf.square(out - x))
            psnr = (10 * (math.log10(1.0 / loss)))
            ssim = tf.reduce_mean(tf.image.ssim(out, x, max_val=1))

            out = out * 255.
            x = x * 255.


        cv2.imwrite('./test_recon/lenna_{}_{}.bmp'.format('%.4f'%psnr,'%.4f'%ssim), tf.squeeze(out).numpy())

        cv2.imwrite('test_recon/x_lenna.bmp', tf.squeeze(x).numpy())


if __name__ == '__main__':
    main(train= 0)




