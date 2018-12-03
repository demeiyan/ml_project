# basic knowledge
* 使用tensorflow 进行矩阵求导

b = a\*a +I,c = mean(b+a),\*表示elementwise multiplication，mean表示求和再取平均，I表示单位矩阵
求 c 对a在a=[[2.,3.,4.],[2.,2.,5.],[1.,3.,4.]]的梯度，代码如下：

``` 
import tensorflow as tf
import numpy as np
# tf.multiply表示elementwise multiplication
# tf.matmul表示正常的矩阵相乘，x*y
a = tf.Variable([[2.,3.,4.],[2.,2.,5.],[1.,3.,4.]])
#a = tf.Variable(np.array(a))
#b = tf.add(a,[[2.,3.,4.],[2.,2.,5.],[1.,3.,4.]])
b = tf.add(tf.multiply(a,a) , tf.Variable([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]))
c = tf.reduce_mean(tf.add(b,a))
grads = tf.gradients(c,a)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    with tf.device("/cpu:0"):
        sess.run(init)
        print(sess.run(c))
        print(sess.run(grads))
```
输出为：

``` 
13.0
[array([[ 0.55555558,  0.77777779,  1.        ],
       [ 0.55555558,  0.55555558,  1.22222233],
       [ 0.33333334,  0.77777779,  1.        ]], dtype=float32)]
```


