# adamax_sgd_mix.py — Keras 3 compatible (TF 2.16.x, keras>=3)
import tensorflow as tf
import keras
from keras.optimizers import Optimizer

class AdamaxSGDMix(Optimizer):
    """
    Mix AdaMax & SGD-momentum:
        step = alpha * step_adamax + (1 - alpha) * step_sgd

    _learning_rate là scale chung cho lr_adamax & lr_sgd (để scheduler/callback tác động).
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
        learning_rate=1.0,  # float hoặc schedule
        name="AdamaxSGDMix",
        **kwargs,
    ):
        # KHÔNG truyền learning_rate vào super().__init__
        super().__init__(name=name, **kwargs)

        # dùng biến nội bộ để tránh ValueError của property base
        self._learning_rate = learning_rate

        # hyperparams
        self.lr_adamax = float(lr_adamax)
        self.lr_sgd    = float(lr_sgd)
        self.beta_1    = float(beta_1)
        self.beta_2    = float(beta_2)
        self.epsilon   = float(epsilon)
        self.momentum  = float(momentum)
        self.nesterov  = bool(nesterov)

        # alpha có thể anneal bởi callback
        self.alpha = tf.Variable(float(alpha), trainable=False, dtype=tf.float32, name="alpha")

        # slot containers & mapping
        self._m = None
        self._u = None
        self._v = None
        self._var_to_idx = None  # _var_key(var) -> index
        self._slots_built = False

    # cho phép đọc/ghi learning_rate từ callback (ReduceLROnPlateau)
    @property
    def learning_rate(self):
        return self._learning_rate
    @learning_rate.setter
    def learning_rate(self, value):
        self._learning_rate = value

    def build(self, var_list):
        # chỉ tạo slot 1 lần để tránh lỗi "tf.Variable created on second call"
        if self._slots_built:
            return
        self._m = [self.add_variable_from_reference(v, "m") for v in var_list]
        self._u = [self.add_variable_from_reference(v, "u") for v in var_list]
        self._v = [self.add_variable_from_reference(v, "v") for v in var_list]
        self._var_to_idx = {self._var_key(v): i for i, v in enumerate(var_list)}
        super().build(var_list)
        self._slots_built = True

    def _idx(self, variable):
        return self._var_to_idx[self._var_key(variable)]

    def update_step(self, grad, variable, learning_rate=None):
        if grad is None:
            return

        i = self._idx(variable)
        m = self._m[i]
        u = self._u[i]
        v = self._v[i]

        var_dtype = variable.dtype
        beta_1 = tf.cast(self.beta_1, var_dtype)
        beta_2 = tf.cast(self.beta_2, var_dtype)
        eps    = tf.cast(self.epsilon, var_dtype)
        mom    = tf.cast(self.momentum, var_dtype)
        alpha  = tf.cast(self.alpha, var_dtype)

        # ưu tiên LR do backend truyền vào; fallback _learning_rate (hỗ trợ schedule)
        lr_scale = learning_rate if learning_rate is not None else self._learning_rate
        if callable(lr_scale):
            lr_scale = lr_scale(self.iterations)
        lr_scale = tf.cast(lr_scale, var_dtype)

        lr_adamax = lr_scale * tf.cast(self.lr_adamax, var_dtype)
        lr_sgd    = lr_scale * tf.cast(self.lr_sgd, var_dtype)

        # --- Adamax ---
        m.assign(beta_1 * m + (1.0 - beta_1) * grad)
        u.assign(tf.maximum(beta_2 * u, tf.abs(grad)))

        # bias correction cho m
        t = tf.cast(self.iterations + 1, var_dtype)
        m_hat = m / (1.0 - tf.pow(beta_1, t))
        step_adamax = - lr_adamax * (m_hat / (u + eps))

        # --- SGD-momentum (Nesterov optional) ---
        v.assign(mom * v - lr_sgd * grad)
        v_now = v
        step_sgd = mom * v_now - lr_sgd * grad if self.nesterov else v_now

        # --- mix & apply ---
        mixed_step = alpha * step_adamax + (1.0 - alpha) * step_sgd
        variable.assign_add(mixed_step)

    def get_config(self):
        base = super().get_config()
        base.update({
            "lr_adamax": self.lr_adamax,
            "lr_sgd":    self.lr_sgd,
            "beta_1":    self.beta_1,
            "beta_2":    self.beta_2,
            "epsilon":   self.epsilon,
            "momentum":  self.momentum,
            "nesterov":  self.nesterov,
            "alpha":     float(self.alpha.numpy() if tf.executing_eagerly() else self.alpha),
            "learning_rate": keras.saving.serialize_keras_object(self._learning_rate),
        })
        return base


class AlphaAnneal(keras.callbacks.Callback):
    def __init__(self, optimizer: AdamaxSGDMix, start=0.8, end=0.1, mode="cosine"):
        super().__init__()
        self.opt = optimizer
        self.start = float(start)
        self.end = float(end)
        self.mode = mode

    def on_train_begin(self, logs=None):
        self.total_epochs = int(self.params.get("epochs", 1))
        self.opt.alpha.assign(self.start)

    def on_epoch_begin(self, epoch, logs=None):
        import math
        p = epoch / max(1, self.total_epochs - 1)
        if self.mode == "cosine":
            val = self.end + 0.5*(self.start - self.end)*(1 + math.cos(math.pi * p))
        else:
            val = self.start + p * (self.end - self.start)
        self.opt.alpha.assign(val)
