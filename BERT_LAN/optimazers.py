# @Time: 2020/8/14 19:01
# @Author: R.Jian
# @Note: 自定义分层学习率

from tensorflow.python.training import training_ops
import tensorflow as tf

class Adam_with_layer(tf.keras.optimizers.Adam):
    def __init__(self, lr=0.001,dict_lr_schedul=None):
        super(Adam_with_layer, self).__init__(learning_rate=lr)
        self.dict_lr_schedul = dict_lr_schedul
    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')

        if not self.amsgrad:
            return training_ops.resource_apply_adam(
                var.handle,
                m.handle,
                v.handle,
                coefficients['beta_1_power'],
                coefficients['beta_2_power'],
                # coefficients['lr_t'],  # replaced by next
                coefficients['lr_t']*self.lr_with_layer(var),
                coefficients['beta_1_t'],
                coefficients['beta_2_t'],
                coefficients['epsilon'],
                grad,
                use_locking=self._use_locking)
        else:
            vhat = self.get_slot(var, 'vhat')
            return training_ops.resource_apply_adam_with_amsgrad(
                var.handle,
                m.handle,
                v.handle,
                vhat.handle,
                coefficients['beta_1_power'],
                coefficients['beta_2_power'],
                # coefficients['lr_t'],# replaced by next
                coefficients['lr_t'] * self.lr_wide_layer(var),
                coefficients['beta_1_t'],
                coefficients['beta_2_t'],
                coefficients['epsilon'],
                grad,
                use_locking=self._use_locking)

    def lr_with_layer(self,var):
        """
        分层学习率
        :return:
        """
        name = var.name
        for k, v in self.dict_lr_schedul.items():
            for title in v:
                if name.find(title) != -1:
                    return k
        return 1