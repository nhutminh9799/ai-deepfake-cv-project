# adamax_sgd_mix_new.py — dùng legacy optimizer cho TF 2.16 / Mac Metal
import tensorflow as tf
from tensorflow.keras.optimizers.legacy import Optimizer as LegacyOptimizer

class AdamaxSGDMix(LegacyOptimizer):
    """
    Trộn AdaMax và SGD-momentum theo alpha:
        step = alpha * step_adamax + (1 - alpha) * step_sgd

    Ghi chú:
    - learning_rate là hệ số scale áp dụng đồng thời cho lr_adamax và lr_sgd,
      để các callback như ReduceLROnPlateau có hiệu lực.
    """

    def __init__(
        self,
        lr_adamax=1e-4,
        lr_sgd=5e-4,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        momentum=0.9,
        nesterov=False,
        alpha=0.7,
        learning_rate=1.0,   # scale factor cho cả hai lr
        name="AdamaxSGDMix",
        **kwargs,
    ):
        super().__init__(name, **kwargs)

        # Hyper chính
        self._set_hyper("learning_rate", learning_rate)  # để Keras/Callbacks truy cập optimizer.lr / optimizer.learning_rate
        self._set_hyper("lr_adamax", lr_adamax)
        self._set_hyper("lr_sgd", lr_sgd)
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self._set_hyper("momentum", momentum)

        self.epsilon = epsilon
        self.nesterov = nesterov

        # Alpha là biến có thể thay đổi theo epoch bởi callback
        self.alpha = tf.Variable(alpha, trainable=False, dtype=tf.float32, name="alpha")

        # Đếm bước cho bias-correction của Adamax
        self.iterations_adamax = tf.Variable(0, dtype=tf.int64, trainable=False, name="iter_adamax")

    # Alias để tương thích tối đa với các callback truy cập .lr
    @property
    def lr(self):
        return self._get_hyper("learning_rate")

    @lr.setter
    def lr(self, value):
        self._set_hyper("learning_rate", value)

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")  # 1st moment (Adamax)
            self.add_slot(var, "u")  # infinity norm (Adamax)
            self.add_slot(var, "v")  # velocity (SGD-momentum)

    @tf.function(jit_compile=False)
    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype

        # Lấy hyper
        lr_scale = tf.cast(self._get_hyper("learning_rate"), var_dtype)  # scale chung
        lr_adamax = lr_scale * tf.cast(self._get_hyper("lr_adamax"), var_dtype)
        lr_sgd    = lr_scale * tf.cast(self._get_hyper("lr_sgd"), var_dtype)

        beta_1 = tf.cast(self._get_hyper("beta_1"), var_dtype)
        beta_2 = tf.cast(self._get_hyper("beta_2"), var_dtype)
        momentum = tf.cast(self._get_hyper("momentum"), var_dtype)
        eps = tf.cast(self.epsilon, var_dtype)
        alpha = tf.cast(self.alpha, var_dtype)

        # Slots
        m = self.get_slot(var, "m")
        u = self.get_slot(var, "u")
        v = self.get_slot(var, "v")

        # --- Updates (thu thập vào ops) ---
        ops = []
        m_t = m.assign(beta_1 * m + (1.0 - beta_1) * grad)
        ops.append(m_t)
        u_t = u.assign(tf.maximum(beta_2 * u, tf.abs(grad)))
        ops.append(u_t)

        self.iterations_adamax.assign_add(1)
        t = tf.cast(self.iterations_adamax, var_dtype)

        # Adamax bias correction & step
        m_hat = m_t / (1.0 - tf.pow(beta_1, t))
        step_adamax = - lr_adamax * (m_hat / (u_t + eps))

        # SGD-momentum (Nesterov optional)
        v_t = v.assign(momentum * v - lr_sgd * grad)
        ops.append(v_t)
        step_sgd = momentum * v_t - lr_sgd * grad if self.nesterov else v_t

        # Mix hai bước cập nhật
        mixed_step = alpha * step_adamax + (1.0 - alpha) * step_sgd
        ops.append(var.assign_add(mixed_step))

        # Trả về Tensor để đảm bảo control_dependencies được thực thi
        with tf.control_dependencies(ops):
            return tf.identity(var)

    @tf.function(jit_compile=False)
    def _resource_apply_sparse(self, grad, var, indices):
        dense_shape = tf.shape(var)
        grad_dense = tf.scatter_nd(tf.expand_dims(indices, 1), grad, dense_shape)
        return self._resource_apply_dense(grad_dense, var)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "learning_rate": float(tf.keras.backend.get_value(self._get_hyper("learning_rate"))),
            "lr_adamax":     float(tf.keras.backend.get_value(self._get_hyper("lr_adamax"))),
            "lr_sgd":        float(tf.keras.backend.get_value(self._get_hyper("lr_sgd"))),
            "beta_1":        float(tf.keras.backend.get_value(self._get_hyper("beta_1"))),
            "beta_2":        float(tf.keras.backend.get_value(self._get_hyper("beta_2"))),
            "epsilon":       float(self.epsilon),
            "momentum":      float(tf.keras.backend.get_value(self._get_hyper("momentum"))),
            "nesterov":      bool(self.nesterov),
            "alpha":         float(self.alpha.numpy()),
        })
        return cfg


class AlphaAnneal(tf.keras.callbacks.Callback):
    def __init__(self, optimizer, start=0.8, end=0.1, mode="cosine"):
        super().__init__()
        self.opt = optimizer
        self.start = float(start)
        self.end = float(end)
        self.mode = mode

    def on_train_begin(self, logs=None):
        self.total_epochs = self.params.get("epochs", 1)
        self.opt.alpha.assign(self.start)

    def on_epoch_begin(self, epoch, logs=None):
        import math
        p = epoch / max(1, self.total_epochs - 1)
        if self.mode == "cosine":
            val = self.end + 0.5*(self.start - self.end)*(1 + math.cos(math.pi * p))
        else:
            val = self.start + p * (self.end - self.start)
        self.opt.alpha.assign(val)
