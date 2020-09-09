# @Time: 2020/8/14 11:41
# @Author: R.Jian
# @Note: 自定义优化器

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.distribute import values as ds_values
from tensorflow.python.util import nest
from tensorflow.python.eager import context
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.training import training_ops
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import _filter_grads

class mine_op(tf.keras.optimizers.Adam):
    def __init__(self, lr=0.01):
        print("init")
        super(mine_op, self).__init__(lr=lr)


    # def get_updates(self, loss, params):
    #     tf.print("get_updates")
    #     return super(mine_op, self).get_updates(loss, params)
    #
    # def minimize(self, loss, var_list, grad_loss=None, name=None):
    #     tf.print("minimize")
    #     return super(mine_op, self).minimize(loss, var_list, grad_loss, name)
    #
    # def apply_gradients(self,
    #                     grads_and_vars,
    #                     name=None,
    #                     experimental_aggregate_gradients=True):
    #     tf.print("apply_gradients")
    #     return super(mine_op, self).apply_gradients(grads_and_vars, name, experimental_aggregate_gradients)
    #
    # def _aggregate_gradients(self, grads_and_vars):
    #     tf.print("_aggregate_gradients")
    #     return super(mine_op, self)._aggregate_gradients(grads_and_vars)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        print(var)
        if not self.amsgrad:
            return training_ops.resource_apply_adam(
                var.handle,
                m.handle,
                v.handle,
                coefficients['beta_1_power'],
                coefficients['beta_2_power'],
                coefficients['lr_t'],
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
                coefficients['lr_t'],
                coefficients['beta_1_t'],
                coefficients['beta_2_t'],
                coefficients['epsilon'],
                grad,
                use_locking=self._use_locking)

    # def _distributed_apply(self, distribution, grads_and_vars, name, apply_state):
    #     """`apply_gradients` using a `DistributionStrategy`."""
    #
    #     def apply_grad_to_update_var(var, grad):
    #         """Apply gradient to variable."""
    #         if isinstance(var, ops.Tensor):
    #             raise NotImplementedError("Trying to update a Tensor ", var)
    #
    #         apply_kwargs = {}
    #         if isinstance(grad, ops.IndexedSlices):
    #             if var.constraint is not None:
    #                 raise RuntimeError(
    #                     "Cannot use a constraint function on a sparse variable.")
    #             if "apply_state" in self._sparse_apply_args:
    #                 apply_kwargs["apply_state"] = apply_state
    #             return self._resource_apply_sparse_duplicate_indices(
    #                 grad.values, var, grad.indices, **apply_kwargs)
    #
    #         if "apply_state" in self._dense_apply_args:
    #             apply_kwargs["apply_state"] = apply_state
    #         update_op = self._resource_apply_dense(grad, var, **apply_kwargs)
    #         if var.constraint is not None:
    #             with ops.control_dependencies([update_op]):
    #                 return var.assign(var.constraint(var))
    #         else:
    #             return update_op
    #
    #     eagerly_outside_functions = ops.executing_eagerly_outside_functions()
    #     update_ops = []
    #     with ops.name_scope(name or self._name, skip_on_eager=True):
    #         for grad, var in grads_and_vars:
    #             # TODO(crccw): It's not allowed to assign PerReplica value to
    #             # MirroredVariable.  Remove this after we relax this restriction.
    #             def _assume_mirrored(grad):
    #                 if isinstance(grad, ds_values.PerReplica):
    #                     return ds_values.Mirrored(grad.values)
    #                 return grad
    #
    #             grad = nest.map_structure(_assume_mirrored, grad)
    #             # Colocate the update with variables to avoid unnecessary communication
    #             # delays. See b/136304694.
    #             with distribution.extended.colocate_vars_with(var):
    #                 with ops.name_scope("update" if eagerly_outside_functions else
    #                                     "update_" + var.op.name, skip_on_eager=True):
    #                     update_ops.extend(distribution.extended.update(
    #                         var, apply_grad_to_update_var, args=(grad,), group=False))
    #
    #         any_symbolic = any(isinstance(i, ops.Operation) or
    #                            tf_utils.is_symbolic_tensor(i) for i in update_ops)
    #         if not context.executing_eagerly() or any_symbolic:
    #             # If the current context is graph mode or any of the update ops are
    #             # symbolic then the step update should be carried out under a graph
    #             # context. (eager updates execute immediately)
    #             with ops._get_graph_from_inputs(update_ops).as_default():  # pylint: disable=protected-access
    #                 with ops.control_dependencies(update_ops):
    #                     return self._iterations.assign_add(1).op
    #
    #         return self._iterations.assign_add(1)
    #
    #     # tf.print("_distributed_apply")
    #     # tf.print(list(grads_and_vars),summarize=100)
    #     # print(list(grads_and_vars))
    #     # return super(mine_op,self)._distributed_apply(distribution, grads_and_vars, name, apply_state)
