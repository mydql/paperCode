import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow import keras
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'



def shift_ps(I, h, w):
    bsize, a, b, c = I.get_shape().as_list()  # i是一个四维元组
    #print("bsize_before", bsize)
    bsize = tf.shape(I)[0]  # Handling Dimension(None) type for undefined batch dim
    #print("bsize_after", bsize)
    X = tf.reshape(I, (bsize, a, b, h, w))  # 变成了五维？
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1

    X = tf.split(X, a, 1)  # a, [bsize, b, r, r]

    X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)
    #print(X.shape)  # bsize, b, a*r, r
    X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, a*r, b*r
    X = tf.reshape(X, (bsize, a * h, b * w, 1))
    return X


class Sample1(keras.Model):
    def __init__(self):
        super(Sample1, self).__init__()
        self.conv = layers.Conv2D(49,32,32,padding='valid')

    def call(self, inputs):
        #print(inputs.shape)
        out = self.conv(inputs)
        out = shift_ps(out,7,7)
        return out

class Sample2(keras.Model):
    def __init__(self):
        super(Sample2, self).__init__()
        self.conv = layers.Conv2D(1024,7,7,padding='valid')

    def call(self, inputs):
        out = self.conv(inputs)
        out = shift_ps(out,32,32)
        return out

def channel_shuffle( x, num_groups):
    n, h, w, c = x.shape.as_list()
    x_reshaped = tf.reshape(x, [-1, h, w, num_groups, c // num_groups]) # 先合并重组
    x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3]) # 转置
    output = tf.reshape(x_transposed, [-1, h, w, c]) # 摊平
    return output

def subpixel(input):
    b,h,w,c = input.shape
    output = 0
    for i in range(c//4):
        if i == 0:
            output = shift_ps(input[:,:,:,4*i:4*i+4],2,2)
        else:
            feature_ = shift_ps(input[:,:,:,4*i:4*i+4],2,2)
            output = tf.concat([output,feature_],axis=-1)

    return output


class SpectralNormalization(tf.keras.layers.Wrapper):
    def __init__(self, layer, iteration=1, eps=1e-12, training=True, **kwargs):
        self.iteration = iteration
        self.eps = eps
        self.do_power_iteration = training
        if not isinstance(layer, tf.keras.layers.Layer):
            raise ValueError(
                'Please initialize `TimeDistributed` layer with a '
                '`Layer` instance. You passed: {input}'.format(input=layer))
        super(SpectralNormalization, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        self.layer.build(input_shape)

        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()

        self.v = self.add_weight(shape=(1, self.w_shape[0] * self.w_shape[1] * self.w_shape[2]),
                                 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                 trainable=False,
                                 name='sn_v',
                                 dtype=tf.float32)

        self.u = self.add_weight(shape=(1, self.w_shape[-1]),
                                 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                 trainable=False,
                                 name='sn_u',
                                 dtype=tf.float32)

        super(SpectralNormalization, self).build()

    def call(self, inputs):
        self.update_weights()
        output = self.layer(inputs)
        self.restore_weights()  # Restore weights because of this formula "W = W - alpha * W_SN`"
        return output

    def update_weights(self):
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])

        u_hat = self.u
        v_hat = self.v  # init v vector

        if self.do_power_iteration:
            for _ in range(self.iteration):
                v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
                v_hat = v_ / (tf.reduce_sum(v_ ** 2) ** 0.5 + self.eps)

                u_ = tf.matmul(v_hat, w_reshaped)
                u_hat = u_ / (tf.reduce_sum(u_ ** 2) ** 0.5 + self.eps)

        sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
        self.u.assign(u_hat)
        self.v.assign(v_hat)

        self.layer.kernel.assign(self.w / sigma)

    def restore_weights(self):
        self.layer.kernel.assign(self.w)
'''
DepthwiseConv2D 卷积
class down_res_block(keras.Model):
    def __init__(self,filter_num,kernel_size=3,stride=1,num_group = 8):
        super(down_res_block, self).__init__()
        self.num_group = num_group
        self.conv = layers.Conv2D(filter_num,1,strides=1,padding='same')


        self.conv1 = layers.DepthwiseConv2D(kernel_size,strides=stride,padding='same')
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(filter_num,kernel_size,strides=1,padding='same',groups=num_group)
        self.bn2 = layers.BatchNormalization()

        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num,1,stride))
        else:
            self.downsample = lambda x:x

    def call(self,inputs,training = None):

        out = self.conv(inputs)
        out = channel_shuffle(out,self.num_group)
        out = self.conv1(out)
        out = tf.nn.leaky_relu(self.bn1(out,training = training))
        out = self.conv2(out)
        out = self.bn2(out,training = training)

        identity = self.downsample(inputs)

        out = tf.nn.leaky_relu(layers.add([out,identity]))

        return out
'''
class down_res_sn_block(keras.Model):
    def __init__(self,filter_num,kernel_size=3,stride=1):
        super(down_res_sn_block, self).__init__()

        self.filter_num = filter_num
        self.kernel_size = kernel_size
        self.stride = stride


        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num,1,stride))
        else:
            self.downsample = lambda x:x

    def call(self,inputs):

        out = SpectralNormalization(layers.Conv2D(self.filter_num,self.kernel_size,self.stride,padding='same'))(inputs)
        out = tf.nn.leaky_relu(out)
        out = SpectralNormalization(layers.Conv2D(self.filter_num,self.kernel_size,1,padding='same'))(out)
        out = tf.nn.leaky_relu(out)

        identity = self.downsample(inputs)

        out = tf.nn.leaky_relu(layers.add([out,identity]))

        return out

'''
class up_res_block(keras.Model):
    def __init__(self,filter_num,kernel_size=3,stride=1,num_group = 8):
        super(up_res_block, self).__init__()

        self.num_group = num_group
        self.conv = layers.Conv2D(filter_num,kernel_size,1,padding='same')
        self.Upsample = layers.UpSampling2D()

        self.conv1 = layers.DepthwiseConv2D(kernel_size,strides=1,padding='same')
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(filter_num,kernel_size,strides=1,padding='same',groups=num_group)
        self.bn2 = layers.BatchNormalization()

        if stride != 1:
            self.upsample = Sequential()
            self.upsample.add(layers.Conv2DTranspose(filter_num,1,stride))
        else:
            self.upsample = lambda x:x

    def call(self,inputs,training = None):
        out = self.conv(inputs)
        out = channel_shuffle(out,self.num_group)
        out = self.Upsample(out)
        out = self.conv1(out)
        out = tf.nn.leaky_relu(self.bn1(out,training = training))
        out = self.conv2(out)
        out = self.bn2(out,training = training)

        identity = self.upsample(inputs)

        out = tf.nn.leaky_relu(layers.add([out,identity]))

        return out




class Block(keras.Model):
    def __init__(self,kernel_size = 3):
        super(Block, self).__init__()

        self.down_block1 = down_res_block(64,kernel_size=kernel_size,stride=2,num_group=8)
        self.down_block2 = down_res_block(128,kernel_size=kernel_size,stride=2,num_group=8)
        self.down_block3 = down_res_block(256,kernel_size=kernel_size,stride=2,num_group=8)

        self.up_block1 = up_res_block(1,kernel_size = kernel_size,stride=2,num_group=1)
        self.up_block2 = up_res_block(64,kernel_size = kernel_size,stride=2,num_group=8)
        self.up_block3 = up_res_block(128,kernel_size=kernel_size,stride=2,num_group=8)

        self.conv1 = layers.Conv2D(1,kernel_size=kernel_size,strides=1,padding='same')
        self.conv2 = layers.Conv2D(64,kernel_size= kernel_size,strides=1,padding='same')
        self.conv3 = layers.Conv2D(128,kernel_size= kernel_size,strides=1,padding='same')

        self.middle31 = layers.Conv2DTranspose(128,kernel_size = kernel_size,strides=2,padding='same')
        self.middle32 = layers.Conv2D(128,kernel_size = kernel_size,strides=1,padding='same')

        self.middle21 = layers.Conv2DTranspose(64,kernel_size = kernel_size,strides=2,padding='same')
        self.middle22 = layers.Conv2D(64,kernel_size = kernel_size,strides=1,padding='same')

        self.middle11 = layers.Conv2DTranspose(1,kernel_size = kernel_size,strides= 2,padding='same')
        self.middle12 = layers.Conv2D(1,kernel_size = kernel_size,strides=1,padding='same')


    def call(self, inputs, training=None, mask=None):

        out1 = self.down_block1(inputs,training = training)
        out2 = self.down_block2(out1,training = training)
        out3 = self.down_block3(out2,training = training)

        middle_31 = self.middle31(out3)+out2
        middle_32 = self.middle32(middle_31)

        middle_21 = self.middle21(middle_32) + out1
        middle_22 = self.middle22(middle_21)

        middle_11 = self.middle11(middle_22) + inputs
        middle_12 = self.middle12(middle_11)


        out4 = self.up_block3(out3)+middle_32
        out5 = self.up_block2(out4) + middle_22
        out = self.up_block1(out5) + middle_12

        return out
'''
"""
class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

        self.alpha1 = tf.Variable(initial_value=1.,trainable=True,dtype=tf.float32)
        self.alpha2 = tf.Variable(initial_value=1., trainable=True, dtype=tf.float32)
        self.alpha3 = tf.Variable(initial_value=1., trainable=True, dtype=tf.float32)
        self.alpha4 = tf.Variable(initial_value=1., trainable=True, dtype=tf.float32)
        self.alpha5 = tf.Variable(initial_value=1., trainable=True, dtype=tf.float32)

        self.block1 = Block(kernel_size=7)
        self.block2 = Block(kernel_size=5)
        self.block3 = Block(kernel_size=3)


    def call(self, inputs, training=None, mask=None):

        out1 = self.block1(inputs,training = None) + self.alpha3*inputs
        out2 = self.block2(out1) + self.alpha4*out1
        out = self.block3(out2) + self.alpha5*out2 + self.alpha1*out1 + self.alpha2*out2

        return out

"""
'''
class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()


        self.block1 = Block(kernel_size=7)




    def call(self, inputs, training=None, mask=None):

        out = self.block1(inputs,training = None)

        return out

'''

# class Discriminator(keras.Model):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#
#         self.conv1 = down_res_sn_block(32,5,5)
#         self.conv2 = down_res_sn_block(64,5,5)
#         self.conv3 = down_res_sn_block(128,5,5)
#
#         self.flatten = layers.Flatten()
#         self.dense1 = layers.Dense(128)
#         self.dense2 = layers.Dense(64)
#         self.dense3 = layers.Dense(1)
#
#     def call(self, inputs):
#         out = self.conv1(inputs)
#         out = self.conv2(out)
#         out = self.conv3(out)
#         out = self.flatten(out)
#         #print(out.shape)
#         out = tf.nn.leaky_relu(self.dense1(out))
#         out = tf.nn.leaky_relu(self.dense2(out))
#         out = self.dense3(out)
#
#
#         return out
#-------------------步长为2的卷积下采样------------------------
class down_res_block2(keras.Model):
    def __init__(self,filter_num,kernel_size=3,stride=1):
        super(down_res_block2, self).__init__()

        self.conv1 = layers.Conv2D(filter_num,kernel_size,strides=stride,padding='same')
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(filter_num,kernel_size,strides=1,padding='same')
        self.bn2 = layers.BatchNormalization()

        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num,1,stride))
        else:
            self.downsample = lambda x:x

    def call(self,inputs,training = None):
        out = tf.nn.leaky_relu(self.bn1(self.conv1(inputs),training = training))
        out = tf.nn.leaky_relu(self.bn2(self.conv2(out),training = training))

        identity = self.downsample(inputs)

        out = tf.nn.leaky_relu(layers.add([out,identity]))

        return out

#------------最大值池化下采样-------------
class down_res_block(keras.Model):
    def __init__(self,filter_num,kernel_size=3,stride=1):
        super(down_res_block, self).__init__()

        self.conv1 = layers.Conv2D(filter_num,kernel_size,strides=1,padding='same')
        self.bn1 = layers.BatchNormalization()
        self.maxpool = layers.MaxPooling2D()

        self.conv2 = layers.Conv2D(filter_num,kernel_size,strides=1,padding='same')
        self.bn2 = layers.BatchNormalization()

        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num,1,1))
            self.downsample.add(layers.MaxPooling2D())
        else:
            self.downsample = lambda x:x
        #
        # if stride != 1:
        #     self.downsample_conv1=layers.Conv2D(filter_num,1,1)
        #     self.downsample_conv2=layers.MaxPooling2D()
        # else:
        #     self.downsample = lambda x:x

    def call(self,inputs,training = None):
        out = tf.nn.leaky_relu(self.bn1(self.conv1(inputs),training = training))
        out = self.maxpool(out)
        out = tf.nn.leaky_relu(self.bn2(self.conv2(out),training = training))

        identity = self.downsample(inputs)

        out = tf.nn.leaky_relu(out+identity)

        return out


#----------反卷积上采样---------------
# class up_res_block(keras.Model):
#     def __init__(self,filter_num,kernel_size=3,stride=1):
#         super(up_res_block, self).__init__()
#
#         self.conv1 = layers.Conv2DTranspose(filter_num,kernel_size,strides=2,padding='same')
#         self.bn1 = layers.BatchNormalization()
#
#         self.conv2 = layers.Conv2D(filter_num,kernel_size,strides=1,padding='same')
#         self.bn2 = layers.BatchNormalization()
#
#         if stride != 1:
#             self.upsample = Sequential()
#             self.upsample.add(layers.Conv2DTranspose(filter_num,2,2,padding='same'))
#         else:
#             self.upsample = lambda x:x
#
#     def call(self,inputs,training = None):
#         out = self.bn1(self.conv1(inputs),training = training)
#         out = tf.nn.leaky_relu(out)
#
#         out = tf.nn.leaky_relu(self.bn2(self.conv2(out),training = training))
#
#         identity = self.upsample(inputs)
#
#         #print(identity.shape,out.shape)
#
#         out = tf.nn.leaky_relu(layers.add([out,identity]))
#
#         return out

#----------------亚像素卷积上采样--------------------
class up_res_block(keras.Model):
    def __init__(self,filter_num,kernel_size=3,stride=1):
        super(up_res_block, self).__init__()
        self.filter_num = filter_num

        self.conv1 = layers.Conv2D(filter_num*4,kernel_size,strides=1,padding='same')
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(filter_num,kernel_size,strides=1,padding='same')
        self.bn2 = layers.BatchNormalization()

        if stride != 1:
            self.upsample = Sequential()
            self.upsample.add(layers.UpSampling2D())
            self.upsample.add(layers.Conv2D(filter_num,1,1,padding='same'))
        else:
            self.upsample = lambda x:x

        # if stride != 1:
        #     self.upsample_conv1 = layers.UpSampling2D()
        #     self.upsample_conv2 = layers.Conv2D(filter_num,1,1,padding='same')
        # else:
        #     self.upsample = lambda x:x

    def call(self,inputs,training = None):
        out = tf.nn.leaky_relu(self.bn1(self.conv1(inputs),training = training))
        out = subpixel(out)
        #print(out.shape)

        out = tf.nn.leaky_relu(self.bn2(self.conv2(out),training = training))

        identity = self.upsample(inputs)

        #print(identity.shape,out.shape)

        out = tf.nn.leaky_relu(out+identity)

        return out



# class Generator(keras.Model):
#     def __init__(self):
#         super(Generator, self).__init__()
#
#         self.sample = Sample2()
#
#         self.down_block13 = down_res_block(64,3,2)
#         self.down_block15 = down_res_block(64,5,2)
#         self.down_block17 = down_res_block(64,7,2)
#
#         self.down_block23 = down_res_block(128,3,2)
#         self.down_block25 = down_res_block(128,5,2)
#         self.down_block27 = down_res_block(128,7,2)
#
#         self.down_block33 = down_res_block(256, 3, 2)
#         self.down_block35 = down_res_block(256, 5, 2)
#         self.down_block37 = down_res_block(256, 7, 2)
#
#         self.up_block33 = up_res_block(128, 3, 2)
#         self.up_block35 = up_res_block(128, 5, 2)
#         self.up_block37 = up_res_block(128, 7, 2)
#
#         self.up_block23 = up_res_block(64, 3, 2)
#         self.up_block25 = up_res_block(64, 5, 2)
#         self.up_block27 = up_res_block(64, 7, 2)
#
#         self.up_block13 = up_res_block(1, 3, 2)
#         self.up_block15 = up_res_block(1, 5, 2)
#         self.up_block17 = up_res_block(1, 7, 2)
#
#         self.conv = layers.Conv2D(256,1,strides=1)
#
#         #self.c_attention = C_attention()
#         #self.s_attention = S_attention()
#
#         self.middle31 = layers.Conv2DTranspose(128*3, kernel_size=1, strides=2, padding='same')
#         self.middle32 = layers.Conv2D(128*3, kernel_size=1, strides=1, padding='same')
#
#         self.middle21 = layers.Conv2DTranspose(64*3, kernel_size=1, strides=2, padding='same')
#         self.middle22 = layers.Conv2D(64*3, kernel_size=1, strides=1, padding='same')
#
#         #self.middle11 = layers.Conv2DTranspose(1, kernel_size=1, strides=2, padding='same')
#         #self.middle12 = layers.Conv2D(1, kernel_size=1, strides=1, padding='same')
#
#
#
#     def call(self, inputs,training = None):
#         _ = self.sample(inputs)
#         out13 = self.down_block13(_,training = training)
#         out15 = self.down_block15(_,training = training)
#         out17 = self.down_block17(_,training = training)
#         out1 = tf.concat([out13,out15,out17],axis=-1)
#
#         out23 = self.down_block23(out1, training=training)
#         out25 = self.down_block25(out1, training=training)
#         out27 = self.down_block27(out1, training=training)
#         out2 = tf.concat([out23, out25, out27], axis=-1)
#
#         out33 = self.down_block33(out2, training=training)
#         out35 = self.down_block35(out2, training=training)
#         out37 = self.down_block37(out2, training=training)
#         out3 = tf.concat([out33, out35, out37], axis=-1)
#
#         middle31 = self.middle31(out3) + out2
#         middle32 = self.middle32(middle31)
#
#         middle21 = self.middle21(middle32) + out1
#         middle22 = self.middle22(middle21)
#
#         #middle11 = self.middle11(middle22) + inputs
#         #middle12 = self.middle12(middle11)
#
#
#         out4 = self.conv(out3)
#
#         #out4 = self.c_attention(out4)
#         #out4 = self.s_attention(out4)
#
#         out53 = self.up_block33(out4,training=training)
#         out55 = self.up_block35(out4,training=training)
#         out57 = self.up_block37(out4,training=training)
#         out5 = tf.concat([out53,out55,out57],axis=-1) + middle32
#
#         out63 = self.up_block23(out5, training=training)
#         out65 = self.up_block25(out5, training=training)
#         out67 = self.up_block27(out5, training=training)
#         out6 = tf.concat([out63, out65, out67], axis=-1) + middle22
#
#         out73 = self.up_block13(out6, training=training)
#         out75 = self.up_block15(out6, training=training)
#         out77 = self.up_block17(out6, training=training)
#         out = tf.tanh(out73 + out75 + out77) #+ middle12
#
#         return out,out4



#----------亚像素上采样生成器----------
# class Generator(keras.Model):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.sample = Sample2()
#         self.down_block13 = down_res_block(64,3,2)
#         self.down_block15 = down_res_block(64,5,2)
#         self.down_block17 = down_res_block(64,7,2)
#
#         self.down_block23 = down_res_block(128,3,2)
#         self.down_block25 = down_res_block(128,5,2)
#         self.down_block27 = down_res_block(128,7,2)
#
#         self.down_block33 = down_res_block(256, 3, 2)
#         self.down_block35 = down_res_block(256, 5, 2)
#         self.down_block37 = down_res_block(256, 7, 2)
#
#         self.up_block33 = up_res_block(128, 3, 2)
#         self.up_block35 = up_res_block(128, 5, 2)
#         self.up_block37 = up_res_block(128, 7, 2)
#
#         self.up_block23 = up_res_block(64, 3, 2)
#         self.up_block25 = up_res_block(64, 5, 2)
#         self.up_block27 = up_res_block(64, 7, 2)
#
#         self.up_block13 = up_res_block(1, 3, 2)
#         self.up_block15 = up_res_block(1, 5, 2)
#         self.up_block17 = up_res_block(1, 7, 2)
#
#         self.conv = layers.Conv2D(256,1,strides=1)
#
#         #self.c_attention = C_attention()
#         #self.s_attention = S_attention()
#
#         self.middle31 = layers.UpSampling2D()
#         self.middle32 = layers.Conv2D(128*3, kernel_size=1, strides=1, padding='same')
#
#         self.middle21 = layers.UpSampling2D()
#         self.middle22= layers.Conv2D(64*3, kernel_size=1, strides=1, padding='same')
#
#         #self.middle11 = layers.Conv2DTranspose(1, kernel_size=1, strides=2, padding='same')
#         #self.middle12 = layers.Conv2D(1, kernel_size=1, strides=1, padding='same')
#
#
#
#     def call(self, inputs,training = None):
#         _ = self.sample(inputs)
#
#         out13 = self.down_block13(_,training = training)
#         out15 = self.down_block15(_,training = training)
#         out17 = self.down_block17(_,training = training)
#         out1 = tf.concat([out13,out15,out17],axis=-1)
#
#         out23 = self.down_block23(out1, training=training)
#         out25 = self.down_block25(out1, training=training)
#         out27 = self.down_block27(out1, training=training)
#         out2 = tf.concat([out23, out25, out27], axis=-1)
#
#         out33 = self.down_block33(out2, training=training)
#         out35 = self.down_block35(out2, training=training)
#         out37 = self.down_block37(out2, training=training)
#         out3 = tf.concat([out33, out35, out37], axis=-1)
#
#         middle31 = self.middle31(out3)
#         middle32 = self.middle32(middle31)+out2
#
#         middle21 = self.middle21(middle32)
#         middle22 = self.middle22(middle21)+out1
#
#         #middle11 = self.middle11(middle22) + inputs
#         #middle12 = self.middle12(middle11)
#
#
#         out4 = self.conv(out3)
#
#         #out4 = self.c_attention(out4)
#         #out4 = self.s_attention(out4)
#
#         out53 = self.up_block33(out4,training=training)
#         out55 = self.up_block35(out4,training=training)
#         out57 = self.up_block37(out4,training=training)
#         out5 = tf.concat([out53,out55,out57],axis=-1) + middle32
#
#         out63 = self.up_block23(out5, training=training)
#         out65 = self.up_block25(out5, training=training)
#         out67 = self.up_block27(out5, training=training)
#         out6 = tf.concat([out63, out65, out67], axis=-1) + middle22
#
#         out73 = self.up_block13(out6, training=training)
#         out75 = self.up_block15(out6, training=training)
#         out77 = self.up_block17(out6, training=training)
#         out = tf.tanh(out73 + out75 + out77) #+ middle12
#
#         return out,out4


# 非对称亚像素卷积生成器
class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.sample = Sample2()
        self.down_block13 = down_res_block(32,3,2)
        self.down_block15 = down_res_block(32,5,2)
        self.down_block17 = down_res_block(32,7,2)

        self.down_block23 = down_res_block(64,3,2)
        self.down_block25 = down_res_block(64,5,2)
        self.down_block27 = down_res_block(64,7,2)

        self.down_block33 = down_res_block(128, 3, 2)
        self.down_block35 = down_res_block(128, 5, 2)
        self.down_block37 = down_res_block(128, 7, 2)

        self.up_block3 = up_res_block(64, 3, 2)

        self.up_block2 = up_res_block(32, 3, 2)

        self.up_block1 = up_res_block(1, 3, 2)

        self.conv3 = layers.Conv2D(64*4,1,strides=1)
        self.conv2 = layers.Conv2D(64,1,strides=1)
        self.conv1 = layers.Conv2D(32,1,strides=1)
        self.conv4 = layers.Conv2D(32*4,1,strides=1)


    def call(self, inputs,training = None):
        _ = self.sample(inputs)

        out13 = self.down_block13(_,training = training)
        out15 = self.down_block15(_,training = training)
        out17 = self.down_block17(_,training = training)
        out1 = tf.concat([out13,out15,out17],axis=-1)
        #out1 = out13+out15+out17

        out23 = self.down_block23(out13, training=training)
        out25 = self.down_block25(out15, training=training)
        out27 = self.down_block27(out17, training=training)
        out2 = tf.concat([out23, out25, out27], axis=-1)
        #out2 = out23+out25+out27

        out33 = self.down_block33(out23, training=training)
        out35 = self.down_block35(out25, training=training)
        out37 = self.down_block37(out27, training=training)
        out3 = tf.concat([out33, out35, out37], axis=-1)
        #out3 = out33+out35+out37

        out4 = self.conv3(out3)

        middle31 = subpixel(out4)
        middle32 = middle31+self.conv2(out2)

        middle21 = subpixel(self.conv4(middle32))
        middle22 = middle21+self.conv1(out1)

        out5 = self.up_block3(out4,training=training) +middle32

        out6 = self.up_block2(out5, training=training)+middle22

        out7 = self.up_block1(out6, training=training)
        out = tf.tanh(out7)

        return out,out4
#



#resnet 做生成器
# class Generator(keras.Model):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.down_block13 = down_res_block(64,3,2)
#
#
#         self.down_block23 = down_res_block(128,3,2)
#
#
#         self.down_block33 = down_res_block(256, 3, 2)
#
#
#         self.up_block33 = up_res_block(128, 3, 2)
#
#
#         self.up_block23 = up_res_block(64, 3, 2)
#
#
#         self.up_block13 = up_res_block(1, 3, 2)
#
#
#         self.conv = layers.Conv2D(256,1,strides=1)
#
#
#
#
#     def call(self, inputs,training = None):
#         out1 = self.down_block13(inputs,training = training)
#
#
#         out2 = self.down_block23(out1, training=training)
#
#
#         out3 = self.down_block33(out2, training=training)
#
#
#
#
#         out4 = self.conv(out3)
#
#         #out4 = self.c_attention(out4)
#         #out4 = self.s_attention(out4)
#
#         out5 = self.up_block33(out4,training=training) +out2
#
#
#         out6 = self.up_block23(out5, training=training) + out1
#
#         out7 = self.up_block13(out6, training=training)+inputs
#
#
#         return out7

#resnet 做生成器,中间层反馈
# class Generator(keras.Model):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.sample = Sample2()
#         self.down_block13 = down_res_block(64,3,2)
#
#
#         self.down_block23 = down_res_block(128,3,2)
#
#
#         self.down_block33 = down_res_block(256, 3, 2)
#
#
#         self.up_block33 = up_res_block(128, 3, 2)
#
#
#         self.up_block23 = up_res_block(64, 3, 2)
#
#
#         self.up_block13 = up_res_block(1, 3, 2)
#
#
#         self.conv = layers.Conv2D(256,1,strides=1)
#
#
#
#
#     def call(self, inputs,training = None):
#         _ = self.sample(inputs)
#         out1 = self.down_block13(_,training = training)
#
#
#         out2 = self.down_block23(out1, training=training)
#
#
#         out3 = self.down_block33(out2, training=training)
#
#
#
#
#         out4 = self.conv(out3)
#
#         #out4 = self.c_attention(out4)
#         #out4 = self.s_attention(out4)
#
#         out5 = self.up_block33(out4,training=training) +out2
#
#
#         out6 = self.up_block23(out5, training=training) + out1
#
#         out7 = self.up_block13(out6, training=training)+_
#
#
#
#         return out7,out4




#双输入判别器  乘上W
class Discriminator_w(keras.Model):
    def __init__(self):
        super(Discriminator_w, self).__init__()

        self.conv21 = layers.Conv2D(32,4,4,padding='same')
        self.conv22 = layers.Conv2D(1,3,1,padding='valid')

        self.conv1 = down_res_sn_block(32,3,2)
        self.conv2 = down_res_sn_block(64,3,2)
        self.conv3 = down_res_sn_block(128,3,2)

        self.conv5 = down_res_sn_block(32, 3, 2)
        self.conv6 = down_res_sn_block(64, 3, 2)
        self.conv7 = down_res_sn_block(128, 3, 2)

        self.flatten1 = layers.Flatten()
        self.flatten2 = layers.Flatten()
        self.dense1 = layers.Dense(128)
        self.dense2 = layers.Dense(64)
        self.dense3 = layers.Dense(1)

        self.kernel1 = tf.Variable(tf.random.truncated_normal([512, 256], mean=0, stddev=0.1))
        self.kernel2 = tf.Variable(tf.random.truncated_normal([512, 256], mean=0, stddev=0.1))

        #self.info1 = layers.Conv2D(32,7,1,padding='valid')
        #self.info2 = layers.Conv2D(32,6,1,padding='valid')



    def call(self, inputs):

        #info1 = self.info1(info)
        #info2 = self.info2(info1)

        out1 = self.conv1(inputs[0])
        # print(out1.shape)
        #out1 = tf.concat([out1,info2],axis=-1)
        out1 = self.conv2(out1)
        out1 = self.conv3(out1)
        out1 = self.flatten1(out1)   #128
        # print(out1.shape)
        out1 = tf.matmul(out1, self.kernel1)

        out2 = self.conv21(inputs[1])
        # print(out2.shape)
        out2 = self.conv22(out2)#[B,14,14,1]
        # print(out2.shape)
        out2 = self.conv5(out2)
        #out2 = tf.concat([out2,info2],axis=-1)
        #print(out2.shape)
        out2 = self.conv6(out2)
        out2 = self.conv7(out2)
        out2 = self.flatten2(out2)  #512
        # print('out2.shape',out2.shape)
        out2 = tf.matmul(out2,self.kernel2)

        out = out1+out2

        #out = self.flatten(out)
        #print(out.shape)
        out = tf.nn.relu(self.dense1(out))
        out = tf.nn.relu(self.dense2(out))
        out = tf.nn.sigmoid(self.dense3(out))


        return out





#单图片输入判别器
class Discriminator_2(keras.Model):
    def __init__(self):
        super(Discriminator_2, self).__init__()

        self.conv21 = layers.Conv2D(32, 5, 5, padding='same')
        self.conv22 = layers.Conv2DTranspose(1, 5, 1, padding='valid')
        self.conv1 = down_res_sn_block(32,3,3)
        self.conv2 = down_res_sn_block(64,3,3)
        self.conv3 = down_res_sn_block(128,5,5)


        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128)
        self.dense2 = layers.Dense(64)
        self.dense3 = layers.Dense(1)

    def call(self, inputs):
        out1 = self.conv21(inputs)
        out1 = self.conv22(out1)
        out1 = self.conv1(out1)
        out1 = self.conv2(out1)
        out1 = self.conv3(out1)

        out = self.flatten(out1)
        #print(out.shape)
        out = tf.nn.leaky_relu(self.dense1(out))
        out = tf.nn.leaky_relu(self.dense2(out))
        out = self.dense3(out)


        return out





if __name__ == '__main__':
    x1 = tf.random.truncated_normal([4,14,14,1],dtype=tf.float32)
    x2 = tf.random.truncated_normal([4, 64, 64, 1], dtype=tf.float32)
    #x3 = tf.random.truncated_normal([32, 224, 224, 1], dtype=tf.float32)

    s1 = Sample1()
    s2 = Sample2()
    out = s1(x2)
    print(out.shape)

    dis = Discriminator_w()
    input = (x1,x2)
    out = dis(input)
    print(out.shape)






