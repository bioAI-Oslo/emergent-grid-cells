import numpy as np
import tensorflow as tf


class GridCellEncoder(tf.keras.layers.Layer):
    """
    Poor idea in trying to make a trainable grid cell encoder. The 
    problem is that for example trainable/differentiable rotation matrices 
    doesn't work.
    """
    def __init__(
        self, units=32, f=2.2, gmax=1, r0=tf.zeros((1, 2, 1), dtype=tf.float32)
    ):
        super(GridCellEncoder, self).__init__()
        self.units = units
        self.f = f  # user-defined spatial frequency
        self.gmax = gmax
        self.r0 = r0

        # create 60degrees 2D rotation matrix
        # self.relative_R = self.rotation_matrix(60 * np.pi / 180, static=True)
        theta = 60 * np.pi / 180
        #c, s = tf.math.cos(theta), tf.math.sin(theta)
        #self.relative_R = tf.constant([[c, -s], [s, c]], dtype=tf.float32)
        #print(self.relative_R.shape)
        self.k1 = tf.constant(
            [[[1.0], [0.0]]], dtype=tf.float32
        )  # init wave vector. unit length in x-direction
        c, s = np.cos(theta), np.sin(theta)
        self.relative_R = tf.constant([[c, -s], [s, c]], dtype=tf.float32)

    def rotation_matrix(self, theta, static=False):
        """
        2d-rotation matrix
        """
        c, s = np.cos(theta), np.sin(theta)
        """
        return (
            tf.constant(((c, -s), (s, c)), dtype=tf.float32)
            if static
            else tf.Variable(((c, -s), (s, c)), dtype=tf.float32)
        )
        """
        #return tf.Variable(((c, -s), (s, c)), dtype=tf.float32)
        return tf.constant([[c, -s], [s, c]], dtype=tf.float32)

    def build(self, input_shape):
        self.orientation_offsets = self.add_weight(
            shape=(self.units,),
            initializer=tf.keras.initializers.RandomUniform(
                minval=0.0, maxval=2 * np.pi, seed=None
            ),
            trainable=True,
        )

        self.f = self.add_weight(
            shape=(self.units,),
            initializer=tf.keras.initializers.RandomUniform(
                minval=self.f / 2, maxval=2 * self.f, seed=None
            ),
            trainable=True,
        )

        """
        c, s = tf.math.cos(self.theta), tf.math.sin(self.theta)
        self.theta = self.add_weight(
            shape=(2,2),
            initializer=tf.keras.initializers.Constant(60 * np.pi / 180),
            trainable=False,
        )
        
        c, s = tf.math.cos(self.theta), tf.math.sin(self.theta)
        self.relative_R = tf.constant([[[c, -s], [s, c]]], dtype=tf.float32)
        """

    def call(self, inputs):
        """(batch_size,2)
        
        Doesn't work because creating a rotation matrix is non-differentiable

        """
        init_R = self.rotation_matrix(self.orientation_offsets)
        print(init_R.shape)
        init_R = tf.transpose(init_R)  # shape=(self.units,2,2)
        print(init_R.shape)

        k1 = init_R @ self.k1  # shape=(self.units,2,1)
        k2 = self.relative_R @ k1  # rotate k1 by 60degrees using R
        k3 = self.relative_R @ k2  # rotate k2 by 60degrees using R
        ks = tf.concat([k1, k2, k3], axis=-1)  # shape=(self.units,2,3)
        ks *= (
            2 * np.pi
        )  # spatial angular frequency (unit-movement in space is one period)
        print(ks.shape, self.f.shape)
        ks *= self.f  # user-defined spatial frequency

        if tf.rank(inputs) == 2:
            inputs = inputs[:, None]

        ws = (np.cos((inputs - self.r0) @ self.ks) + 0.5) / 3
        ws = ws * self.gmax * 2 / 3
        return tf.squeeze(ws)
