""" library to take autodiff and execute a computation graph """
from __future__ import absolute_import

import time
import numpy as np
from . import ndarray, gpu_op

class Node(object):
    """Node in a computation graph."""
    def __init__(self):
        self.inputs = []
        self.op = None
        self.const_attr = None
        self.name = ""
        self._fusion_pattern = None
        self._backend = "cuda"

    def __add__(self, other):
        if isinstance(other, Node):
            return add_op(self, other)
        return add_byconst_op(self, other)

    def __sub__(self, other):
        if isinstance(other, Node):
            return sub_op(self, other)
        return add_byconst_op(self, -float(other))

    def __rsub__(self, other):
        if isinstance(other, Node):
            return sub_op(other, self)
        return add_byconst_op(mul_byconst_op(self, -1.0), float(other))

    def __mul__(self, other):
        if isinstance(other, Node):
            return mul_op(self, other)
        return mul_byconst_op(self, other)

    def __truediv__(self, other):
        if isinstance(other, Node):
            return div_op(self, other)
        return mul_byconst_op(self, 1.0 / float(other))

    __div__ = __truediv__
    __radd__ = __add__
    __rmul__ = __mul__

    def __str__(self):
        """Allow print to display node name."""
        return self.name


def Variable(name):
    """User defined variables in an expression.
        e.g. x = Variable(name = "x")
    """
    placeholder_node = placeholder_op()
    placeholder_node.name = name
    return placeholder_node


class Op(object):
    """Op represents operations performed on nodes."""
    def __call__(self):
        """Create a new node and associate the op object with the node.

        Returns
        -------
        The new node object.
        """
        new_node = Node()
        new_node.op = self
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        """Given values of input nodes, compute the output value.

        Parameters
        ----------
        node: node that performs the compute.
        input_vals: values of input nodes.
        output_val: output value of the node, modified in-place.
        use_numpy: bool flag whether to use numpy for compute
        """
        raise NotImplementedError

    def gradient(self, node, output_grad):
        """Given output gradient, compute partial gradient to each input node.

        Parameters
        ----------
        node: node that performs the gradient.
        output_grad: output gradient summed from children nodes' contributions

        Returns
        -------
        A list of gradient contributions to each input node respectively.
        """
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        """Given shapes of input nodes, compute shape of output node.

        Implementation note:
        It's simpler to treat shape of constants as (1,), so that constants can
        be stored as a numpy array too and you would need fewer special case
        handling.

        Parameters
        ----------
        node: node whose shape is being inferred.
        input_vals: shapes of input nodes.

        Returns
        -------
        A tuple representing the shape of output node.
        """
        raise NotImplementedError


class AddOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s+%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        if use_numpy:
            # output_val[:] allows modify in-place
            output_val[:] = input_vals[0] + input_vals[1]
        else:
            if input_vals[0].shape == input_vals[1].shape:
                gpu_op.matrix_elementwise_add(
                    input_vals[0], input_vals[1], output_val)
            else:
                if input_vals[1].shape == (1,):
                    const_val = input_vals[1].asnumpy()[0]
                    gpu_op.matrix_elementwise_add_by_const(
                        input_vals[0], const_val, output_val)
                elif input_vals[0].shape == (1,):
                    const_val = input_vals[0].asnumpy()[0]
                    gpu_op.matrix_elementwise_add_by_const(
                        input_vals[1], const_val, output_val)

    def gradient(self, node, output_grad):
        return [output_grad, output_grad]

    def infer_shape(self, node, input_shapes):
        """Need to handle input_vals[0].shape != input_vals[1].shape"""
        """TODO: Your code here"""
        output_shape = input_shapes[0]
        if input_shapes[0] != input_shapes[1]:
          output_shape = broadcast_rule(input_shapes[0], input_shapes[1])
        return output_shape

class AddByConstOp(Op):
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s+%s)" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        if use_numpy:
            output_val[:] = input_vals[0] + node.const_attr
        else:
            gpu_op.matrix_elementwise_add_by_const(
                input_vals[0], node.const_attr, output_val)

    def gradient(self, node, output_grad):
        return [output_grad]

    def infer_shape(self, node, input_shapes):
        """TODO: Your code here"""
        output_shape = input_shapes[0]
        return output_shape


MatAddOp = AddOp


class SubOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s-%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        if use_numpy:
            output_val[:] = input_vals[0] - input_vals[1]
        else:
            gpu_op.matrix_elementwise_sub(
                input_vals[0], input_vals[1], output_val)

    def gradient(self, node, output_grad):
        return [output_grad, -1.0 * output_grad]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        if input_shapes[0] != input_shapes[1]:
            return broadcast_rule(input_shapes[0], input_shapes[1])
        return input_shapes[0]


class DivOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s/%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        if use_numpy:
            output_val[:] = input_vals[0] / input_vals[1]
        else:
            gpu_op.matrix_elementwise_div(
                input_vals[0], input_vals[1], output_val)

    def gradient(self, node, output_grad):
        grad_A = div_op(output_grad, node.inputs[1])
        grad_B = -1.0 * div_op(
            output_grad * node.inputs[0], node.inputs[1] * node.inputs[1])
        return [grad_A, grad_B]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        assert input_shapes[0] == input_shapes[1]
        return input_shapes[0]


class SqrtOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Sqrt(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        if use_numpy:
            output_val[:] = np.sqrt(input_vals[0])
        else:
            gpu_op.sqrt(input_vals[0], output_val)

    def gradient(self, node, output_grad):
        return [0.5 * div_op(output_grad, node)]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]

class MulOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s*%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        if use_numpy:
            output_val[:] = input_vals[0] * input_vals[1]
        else:
            if input_vals[0].shape == input_vals[1].shape:
                gpu_op.matrix_elementwise_multiply(
                    input_vals[0], input_vals[1], output_val)
            else:
                if input_vals[1].shape == (1,):
                    const_val = input_vals[1].asnumpy()[0]
                    gpu_op.matrix_elementwise_multiply_by_const(
                        input_vals[0], const_val, output_val)
                elif input_vals[0].shape == (1,):
                    const_val = input_vals[0].asnumpy()[0]
                    gpu_op.matrix_elementwise_multiply_by_const(
                        input_vals[1], const_val, output_val)

    def gradient(self, node, output_grad):
        return [node.inputs[1] * output_grad, node.inputs[0] * output_grad]

    def infer_shape(self, node, input_shapes):
        """Need to handle input_vals[0].shape != input_vals[1].shape"""
        """TODO: Your code here"""
        output_shape = input_shapes[0]
        if input_shapes[0] != input_shapes[1]:
          output_shape = broadcast_rule(input_shapes[0], input_shapes[1])
        return output_shape

class MulByConstOp(Op):
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s*%s)" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        if use_numpy:
            output_val[:] = input_vals[0] * node.const_attr
        else:
            gpu_op.matrix_elementwise_multiply_by_const(
                input_vals[0], node.const_attr, output_val)

    def gradient(self, node, output_grad):
        return [node.const_attr * output_grad]

    def infer_shape(self, node, input_shapes):
        """TODO: Your code here"""
        output_shape = input_shapes[0]
        return output_shape

class MatMulOp(Op):
    def __call__(self, node_A, node_B, trans_A=False, trans_B=False):
        new_node = Op.__call__(self)
        new_node.matmul_attr_trans_A = trans_A
        new_node.matmul_attr_trans_B = trans_B
        new_node.inputs = [node_A, node_B]
        new_node.name = "MatMul(%s,%s,%s,%s)" % (
            node_A.name, node_B.name, str(trans_A), str(trans_B))
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        if use_numpy:
            if ((node.matmul_attr_trans_A is False) and
                    (node.matmul_attr_trans_B is False)):
                output_val[:] = np.matmul(input_vals[0], input_vals[1])
            elif ((node.matmul_attr_trans_A is True) and
                    (node.matmul_attr_trans_B is False)):
                output_val[:] = np.matmul(
                    np.transpose(input_vals[0]), input_vals[1])
            elif ((node.matmul_attr_trans_A is False) and
                    (node.matmul_attr_trans_B is True)):
                output_val[:] = np.matmul(
                    input_vals[0], np.transpose(input_vals[1]))
            elif ((node.matmul_attr_trans_A is True) and
                    (node.matmul_attr_trans_B is True)):
                output_val[:] = np.matmul(
                    np.transpose(input_vals[0]), np.transpose(input_vals[1]))
        else:
            gpu_op.matrix_multiply(
                input_vals[0], node.matmul_attr_trans_A,
                input_vals[1], node.matmul_attr_trans_B,
                output_val)

    def gradient(self, node, output_grad):
        if ((node.matmul_attr_trans_A is False) and
                (node.matmul_attr_trans_B is False)):
            lhs_grad = matmul_op(
                output_grad, node.inputs[1], trans_A=False, trans_B=True)
            rhs_grad = matmul_op(
                node.inputs[0], output_grad, trans_A=True, trans_B=False)
        elif ((node.matmul_attr_trans_A is True) and
                (node.matmul_attr_trans_B is False)):
            lhs_grad = matmul_op(
                node.inputs[1], output_grad, trans_A=False, trans_B=True)
            rhs_grad = matmul_op(
                node.inputs[0], output_grad, trans_A=False, trans_B=False)
        elif ((node.matmul_attr_trans_A is False) and
                (node.matmul_attr_trans_B is True)):
            lhs_grad = matmul_op(
                output_grad, node.inputs[1], trans_A=False, trans_B=False)
            rhs_grad = matmul_op(
                output_grad, node.inputs[0], trans_A=True, trans_B=False)
        elif ((node.matmul_attr_trans_A is True) and
                (node.matmul_attr_trans_B is True)):
            lhs_grad = matmul_op(
                node.inputs[1], output_grad, trans_A=True, trans_B=True)
            rhs_grad = matmul_op(
                output_grad, node.inputs[0], trans_A=True, trans_B=True)
        return [lhs_grad, rhs_grad]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        shape_A = input_shapes[0]
        shape_B = input_shapes[1]
        if node.matmul_attr_trans_A:
            shape_A = (shape_A[1], shape_A[0])
        if node.matmul_attr_trans_B:
            shape_B = (shape_B[1], shape_B[0])
        assert shape_A[1] == shape_B[0]
        return (shape_A[0], shape_B[1])

class PlaceholderOp(Op):
    def __call__(self):
        """Creates a variable node."""
        new_node = Op.__call__(self)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert False, "placeholder %s values provided by feed_dict" % node.name

    def gradient(self, node, output_grad):
        return None

    def infer_shape(self, node, input_shapes):
        assert False, "placeholder %s shape provided by feed_shape" % node.name


class ZerosLikeOp(Op):
    def __call__(self, node_A):
        """Creates a node that represents np.zeros(node_A.shape)."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Zeroslike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        if use_numpy:
            output_val[:] = np.zeros(input_vals[0].shape)
        else:
            gpu_op.array_set(output_val, 0)

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


class OnesLikeOp(Op):
    def __call__(self, node_A):
        """Creates a node that represents np.ones(node_A.shape)."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Oneslike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        if use_numpy:
            output_val[:] = np.ones(input_vals[0].shape)
        else:
            gpu_op.array_set(output_val, 1)

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


class ReduceSumAxisZeroOp(Op):
    def __call__(self, node_A):
        """Creates a node that represents np.sum(node_A, axis=0)."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "ReduceSumAxisZero(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        if use_numpy:
            assert isinstance(input_vals[0], np.ndarray)
            output_val[:] = np.sum(input_vals[0], axis=0)
        else:
            gpu_op.reduce_sum_axis_zero(input_vals[0], output_val)

    def gradient(self, node, output_grad):
        return [broadcastto_op(output_grad, node.inputs[0])]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        in_shape = input_shapes[0]
        if len(in_shape) == 1:
            return (1,)
        return in_shape[1:]


class BroadcastToOp(Op):
    def __call__(self, node_A, node_B):
        """Creates a node that represents np.broadcast_to(node_A, node_B.shape)."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "BroadcastTo(%s,%s.shape)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        if use_numpy:
            output_val[:] = np.broadcast_to(input_vals[0], input_vals[1].shape)
        else:
            gpu_op.broadcast_to(input_vals[0], output_val)

    def gradient(self, node, output_grad):
        grad_A = reducesumaxiszero_op(output_grad)
        grad_B = zeroslike_op(node.inputs[1])
        return [grad_A, grad_B]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        return input_shapes[1]


class ReduceMeanAxisOneOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "ReduceMeanAxisOne(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        if use_numpy:
            output_val[:] = np.mean(input_vals[0], axis=1)
        else:
            gpu_op.reduce_mean_axis_one(input_vals[0], output_val)

    def gradient(self, node, output_grad):
        return [broadcastaxisone_op(output_grad, node.inputs[0])]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        return (input_shapes[0][0],)


class BroadcastAxisOneOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "BroadcastAxisOne(%s,%s.shape)" % (
            node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        if use_numpy:
            output_val[:] = np.broadcast_to(
                input_vals[0][:, np.newaxis], input_vals[1].shape)
        else:
            gpu_op.broadcast_axis_one(input_vals[0], output_val)

    def gradient(self, node, output_grad):
        return [reducemeanaxisone_op(output_grad), zeroslike_op(node.inputs[1])]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        return input_shapes[1]


class BroadcastScalarOp(Op):
    def __call__(self, node_scalar, node_target):
        new_node = Op.__call__(self)
        new_node.inputs = [node_scalar, node_target]
        new_node.name = "BroadcastScalar(%s,%s.shape)" % (
            node_scalar.name, node_target.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        if use_numpy:
            output_val[:] = np.broadcast_to(input_vals[0], input_vals[1].shape)
        else:
            gpu_op.broadcast_scalar(input_vals[0], output_val)

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        return input_shapes[1]

def softmax_func(y):
    """Numerically stable softmax."""
    b = y - np.max(y, axis=1, keepdims=True)
    expb = np.exp(b)
    softmax = expb / np.sum(expb, axis=1, keepdims=True)
    return softmax


def gelu_func(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x ** 3))))


def gelu_grad_func(x, dy):
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    coeff = 0.044715
    inner = sqrt_2_over_pi * (x + coeff * (x ** 3))
    tanh_inner = np.tanh(inner)
    sech2 = 1.0 - tanh_inner * tanh_inner
    d_inner = sqrt_2_over_pi * (1.0 + 3.0 * coeff * (x ** 2))
    dx = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * d_inner
    return dy * dx


class SoftmaxCrossEntropyOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "SoftmaxXEntropy(%s,%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        y = input_vals[0]
        y_ = input_vals[1]
        if use_numpy:
            softmax = softmax_func(y)
            cross_entropy = np.mean(
                -np.sum(y_ * np.log(softmax), axis=1), keepdims=True)
            output_val[:] = cross_entropy
        else:
            gpu_op.softmax_cross_entropy(y, y_, output_val)

    def gradient(self, node, output_grad):
        grad_A = (softmax_op(node.inputs[0]) + -1 * node.inputs[1])*output_grad
        grad_A._fusion_pattern = {
            "kind": "softmax_ce_grad",
            "logits": node.inputs[0],
            "labels": node.inputs[1],
            "out_grad": output_grad,
        }
        grad_B = zeroslike_op(node.inputs[1])
        return [grad_A, grad_B]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        return (1,)


class SoftmaxCrossEntropyGradOp(Op):
    def __call__(self, logits_node, labels_node, out_grad_node, backend="cuda"):
        new_node = Op.__call__(self)
        new_node.inputs = [logits_node, labels_node, out_grad_node]
        new_node._backend = backend
        new_node.backend = backend
        new_node.name = "SoftmaxXEntropyGrad(%s,%s,%s)" % (
            logits_node.name, labels_node.name, out_grad_node.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 3
        logits, labels, out_grad = input_vals
        if use_numpy:
            output_val[:] = (softmax_func(logits) - labels) * out_grad
        elif getattr(node, "_backend", "cuda") == "triton":
            from . import triton_ops
            triton_ops.triton_softmax_ce_grad(logits, labels, out_grad, output_val)
        else:
            gpu_op.softmax_cross_entropy_grad(logits, labels, out_grad, output_val)

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 3
        return input_shapes[0]

class SoftmaxOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Softmax(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        if use_numpy:
            output_val[:] = softmax_func(input_vals[0])
        else:
            gpu_op.softmax(input_vals[0], output_val)

    def gradient(self, node, output_grad):
        # Do not directly use SoftmaxOp, use SoftmaxCrossEntropyOp instead.
        # Not allowing taking 2nd derivative of SoftmaxCrossEntropyOp.
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        """TODO: Your code here"""
        output_shape = input_shapes[0]
        return output_shape

class ReluOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Relu(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        if use_numpy:
            output_val[:] = np.maximum(input_vals[0], 0)
        else:
            gpu_op.relu(input_vals[0], output_val)

    def gradient(self, node, output_grad):
        return [relu_gradient_op(node.inputs[0], output_grad)]

    def infer_shape(self, node, input_shapes):
        """TODO: Your code here"""
        output_shape = input_shapes[0]
        return output_shape

class ReluGradientOp(Op):
    def __call__(self, node_A, node_B):
        """node_B is output_grad"""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "ReluGradient(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        if use_numpy:
            # heaviside function, 0.5 at x=0
            output_val[:] = (np.sign(input_vals[0]) + 1) * 0.5 * input_vals[1]
        else:
            gpu_op.relu_gradient(input_vals[0], input_vals[1], output_val)

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        """TODO: Your code here"""
        output_shape = input_shapes[0]
        return output_shape


class GeluOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Gelu(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        if use_numpy:
            output_val[:] = gelu_func(input_vals[0])
        else:
            gpu_op.gelu(input_vals[0], output_val)

    def gradient(self, node, output_grad):
        return [gelu_gradient_op(node.inputs[0], output_grad)]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


class GeluGradientOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "GeluGradient(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        if use_numpy:
            output_val[:] = gelu_grad_func(input_vals[0], input_vals[1])
        else:
            gpu_op.gelu_gradient(input_vals[0], input_vals[1], output_val)

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        return input_shapes[0]


class FusedLinearBiasActOp(Op):
    def __call__(self, x_node, w_node, b_node=None, trans_A=False, trans_B=False, act_type=0, backend="cuda"):
        if isinstance(act_type, str):
            act_map = {"none": 0, "relu": 1, "gelu": 2}
            act_type = act_map.get(act_type.lower(), 0)
        new_node = Op.__call__(self)
        new_node.inputs = [x_node, w_node, b_node] if b_node is not None else [x_node, w_node]
        new_node.matmul_attr_trans_A = trans_A
        new_node.matmul_attr_trans_B = trans_B
        new_node.act_type = int(act_type)
        new_node._backend = backend
        new_node.backend = backend
        act_names = {0: "None", 1: "Relu", 2: "Gelu"}
        b_name = b_node.name if b_node is not None else "None"
        new_node.name = "FusedLinear(%s,%s,%s,act=%s)" % (
            x_node.name, w_node.name, b_name, act_names.get(new_node.act_type, str(new_node.act_type)))
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) in (2, 3)
        if len(input_vals) == 3:
            x, w, b = input_vals
        else:
            x, w = input_vals
            b = None
        backend = getattr(node, "_backend", getattr(node, "backend", "cuda"))
        if use_numpy:
            a = np.transpose(x) if node.matmul_attr_trans_A else x
            weight = np.transpose(w) if node.matmul_attr_trans_B else w
            z = np.matmul(a, weight)
            if b is not None:
                z = z + b
            if node.act_type == 1:
                output_val[:] = np.maximum(z, 0.0)
            elif node.act_type == 2:
                output_val[:] = gelu_func(z)
            else:
                output_val[:] = z
        elif (backend == "triton"
              and not node.matmul_attr_trans_A
              and not node.matmul_attr_trans_B):
            from . import triton_ops
            gpu_op.inc_kernel_launch_count(1)
            triton_ops.triton_fused_linear(x, w, b, output_val, act_type=node.act_type)
        else:
            gpu_op.fused_linear(
                x, node.matmul_attr_trans_A,
                w, node.matmul_attr_trans_B,
                b, output_val, act_type=node.act_type)

    def gradient(self, node, output_grad):
        if len(node.inputs) == 3:
            x, w, b = node.inputs
        else:
            x, w = node.inputs
            b = None
        pre_act = fused_linear_op(
            x, w, b,
            trans_A=node.matmul_attr_trans_A,
            trans_B=node.matmul_attr_trans_B,
            act_type=0,
            backend=getattr(node, "_backend", getattr(node, "backend", "cuda")))
        if node.act_type == 1:
            dz = relu_gradient_op(pre_act, output_grad)
        elif node.act_type == 2:
            dz = gelu_gradient_op(pre_act, output_grad)
        else:
            dz = output_grad
        fake_mm = matmul_op(
            x, w,
            trans_A=node.matmul_attr_trans_A,
            trans_B=node.matmul_attr_trans_B)
        dx, dw = fake_mm.op.gradient(fake_mm, dz)
        if b is not None:
            db = reducesumaxiszero_op(dz)
            return [dx, dw, db]
        return [dx, dw]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) in (2, 3)
        shape_A = input_shapes[0]
        shape_B = input_shapes[1]
        if node.matmul_attr_trans_A:
            shape_A = (shape_A[1], shape_A[0])
        if node.matmul_attr_trans_B:
            shape_B = (shape_B[1], shape_B[0])
        assert shape_A[1] == shape_B[0]
        return (shape_A[0], shape_B[1])

class LayerNormOp(Op):
    def __call__(self, x_node, gamma_node, beta_node, eps=1e-5, backend="cuda"):
        new_node = Op.__call__(self)
        new_node.inputs = [x_node, gamma_node, beta_node]
        new_node.eps = eps
        new_node._backend = backend
        new_node.backend = backend
        new_node.name = "LayerNorm(%s,%s,%s)" % (
            x_node.name, gamma_node.name, beta_node.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 3
        x, gamma, beta = input_vals
        backend = getattr(node, "_backend", getattr(node, "backend", "cuda"))
        if use_numpy:
            mean = np.mean(x, axis=1, keepdims=True)
            var = np.var(x, axis=1, keepdims=True)
            x_hat = (x - mean) / np.sqrt(var + node.eps)
            output_val[:] = x_hat * gamma + beta
        elif backend == "triton":
            from . import triton_ops
            gpu_op.inc_kernel_launch_count(1)
            triton_ops.triton_layernorm(x, gamma, beta, output_val, eps=node.eps)
        else:
            gpu_op.layernorm(x, gamma, beta, output_val, eps=node.eps)

    def gradient(self, node, output_grad):
        x, gamma, _ = node.inputs
        backend = getattr(node, "_backend", getattr(node, "backend", "cuda"))
        dx = layernorm_grad_op(output_grad, x, gamma, eps=node.eps, which="dx", backend=backend)
        dgamma = layernorm_grad_op(output_grad, x, gamma, eps=node.eps, which="dgamma", backend=backend)
        dbeta = layernorm_grad_op(output_grad, x, gamma, eps=node.eps, which="dbeta", backend=backend)
        return [dx, dgamma, dbeta]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 3
        return input_shapes[0]


class LayerNormGradientOp(Op):
    def __call__(self, dy_node, x_node, gamma_node, eps=1e-5, which="dx", backend="cuda"):
        new_node = Op.__call__(self)
        new_node.inputs = [dy_node, x_node, gamma_node]
        new_node.eps = eps
        new_node.which = which
        new_node._backend = backend
        new_node.backend = backend
        new_node.name = "LayerNormGrad_%s(%s)" % (which, x_node.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 3
        dy, x, gamma = input_vals
        if use_numpy:
            mean = np.mean(x, axis=1, keepdims=True)
            var = np.var(x, axis=1, keepdims=True)
            rstd = 1.0 / np.sqrt(var + node.eps)
            x_hat = (x - mean) * rstd
            if node.which == "dx":
                dy_gamma = dy * gamma
                c1 = np.mean(dy_gamma, axis=1, keepdims=True)
                c2 = np.mean(dy_gamma * x_hat, axis=1, keepdims=True)
                output_val[:] = rstd * (dy_gamma - c1 - x_hat * c2)
            elif node.which == "dgamma":
                output_val[:] = np.sum(dy * x_hat, axis=0)
            elif node.which == "dbeta":
                output_val[:] = np.sum(dy, axis=0)
        else:
            if node.which == "dx":
                gpu_op.layernorm_gradient(dy, x, gamma, output_val, None, None, eps=node.eps)
            elif node.which == "dgamma":
                gpu_op.layernorm_gradient(dy, x, gamma, None, output_val, None, eps=node.eps)
            elif node.which == "dbeta":
                gpu_op.reduce_sum_axis_zero(dy, output_val)

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 3
        if node.which == "dx":
            return input_shapes[1]
        return input_shapes[2]


class AdaLNOp(Op):
    def __call__(self, x_node, scale_node, shift_node, eps=1e-5, backend="cuda"):
        new_node = Op.__call__(self)
        new_node.inputs = [x_node, scale_node, shift_node]
        new_node.eps = eps
        new_node._backend = backend
        new_node.backend = backend
        new_node.name = "AdaLN(%s,%s,%s)" % (
            x_node.name, scale_node.name, shift_node.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 3
        x, scale, shift = input_vals
        backend = getattr(node, "_backend", getattr(node, "backend", "cuda"))
        if use_numpy:
            mean = np.mean(x, axis=1, keepdims=True)
            var = np.var(x, axis=1, keepdims=True)
            x_hat = (x - mean) / np.sqrt(var + node.eps)
            output_val[:] = x_hat * (1.0 + scale) + shift
        elif backend == "triton":
            from . import triton_ops
            gpu_op.inc_kernel_launch_count(1)
            triton_ops.triton_adaln(x, scale, shift, output_val, eps=node.eps)
        else:
            gpu_op.adaln(x, scale, shift, output_val, eps=node.eps)

    def gradient(self, node, output_grad):
        x, scale, _ = node.inputs
        backend = getattr(node, "_backend", getattr(node, "backend", "cuda"))
        dx = adaln_grad_op(output_grad, x, scale, eps=node.eps, which="dx", backend=backend)
        dscale = adaln_grad_op(output_grad, x, scale, eps=node.eps, which="dscale", backend=backend)
        dshift = adaln_grad_op(output_grad, x, scale, eps=node.eps, which="dshift", backend=backend)
        return [dx, dscale, dshift]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 3
        return input_shapes[0]


class AdaLNGradientOp(Op):
    def __call__(self, dy_node, x_node, scale_node, eps=1e-5, which="dx", backend="cuda"):
        new_node = Op.__call__(self)
        new_node.inputs = [dy_node, x_node, scale_node]
        new_node.eps = eps
        new_node.which = which
        new_node._backend = backend
        new_node.backend = backend
        new_node.name = "AdaLNGrad_%s(%s)" % (which, x_node.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 3
        dy, x, scale = input_vals
        if use_numpy:
            mean = np.mean(x, axis=1, keepdims=True)
            var = np.var(x, axis=1, keepdims=True)
            rstd = 1.0 / np.sqrt(var + node.eps)
            x_hat = (x - mean) * rstd
            if node.which == "dx":
                dy_s = dy * (1.0 + scale)
                c1 = np.mean(dy_s, axis=1, keepdims=True)
                c2 = np.mean(dy_s * x_hat, axis=1, keepdims=True)
                output_val[:] = rstd * (dy_s - c1 - x_hat * c2)
            elif node.which == "dscale":
                output_val[:] = dy * x_hat
            elif node.which == "dshift":
                output_val[:] = dy
        else:
            if node.which == "dx":
                gpu_op.adaln_gradient(dy, x, scale, output_val, None, None, eps=node.eps)
            elif node.which == "dscale":
                ctx = x.ctx
                dx_tmp = ndarray.empty(x.shape, ctx=ctx)
                gpu_op.adaln_gradient(dy, x, scale, dx_tmp, output_val, None, eps=node.eps)
            elif node.which == "dshift":
                gpu_op.matrix_elementwise_add_by_const(dy, 0.0, output_val)

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 3
        return input_shapes[1]


# Create global singletons of operators.
add_op = AddOp()
sub_op = SubOp()
mul_op = MulOp()
div_op = DivOp()
sqrt_op = SqrtOp()
add_byconst_op = AddByConstOp()
mul_byconst_op = MulByConstOp()
matmul_op = MatMulOp()
placeholder_op = PlaceholderOp()
oneslike_op = OnesLikeOp()
zeroslike_op = ZerosLikeOp()
reducesumaxiszero_op = ReduceSumAxisZeroOp()
broadcastto_op = BroadcastToOp()
reducemeanaxisone_op = ReduceMeanAxisOneOp()
broadcastaxisone_op = BroadcastAxisOneOp()
broadcastscalar_op = BroadcastScalarOp()
softmaxcrossentropy_op = SoftmaxCrossEntropyOp()
softmaxcrossentropy_grad_op = SoftmaxCrossEntropyGradOp()
softmax_cross_entropy_grad_op = softmaxcrossentropy_grad_op
softmax_op = SoftmaxOp()
relu_op = ReluOp()
relu_gradient_op = ReluGradientOp()
gelu_op = GeluOp()
gelu_gradient_op = GeluGradientOp()
fused_linear_op = FusedLinearBiasActOp()
layernorm_op = LayerNormOp()
layernorm_grad_op = LayerNormGradientOp()
adaln_op = AdaLNOp()
adaln_grad_op = AdaLNGradientOp()


def layernorm_unfused_op(x_node, gamma_node, beta_node, eps=1e-5):
    """Decomposed LayerNorm graph annotated for FuseNormPass."""
    mean_1d = reducemeanaxisone_op(x_node)
    mean_2d = broadcastaxisone_op(mean_1d, x_node)
    diff = sub_op(x_node, mean_2d)
    sq = mul_op(diff, diff)
    var_1d = reducemeanaxisone_op(sq)
    var_eps_1d = add_byconst_op(var_1d, eps)
    std_1d = sqrt_op(var_eps_1d)
    std_2d = broadcastaxisone_op(std_1d, x_node)
    x_hat = div_op(diff, std_2d)
    gamma_2d = broadcastto_op(gamma_node, x_node)
    beta_2d = broadcastto_op(beta_node, x_node)
    scaled = mul_op(x_hat, gamma_2d)
    out = add_op(scaled, beta_2d)
    out._fusion_pattern = {
        "kind": "layernorm",
        "x": x_node,
        "gamma": gamma_node,
        "beta": beta_node,
        "eps": eps,
        "internal_nodes": [
            mean_1d, mean_2d, diff, sq, var_1d, var_eps_1d,
            std_1d, std_2d, x_hat, gamma_2d, beta_2d, scaled, out,
        ],
    }
    return out


def adaln_unfused_op(x_node, scale_node, shift_node, eps=1e-5):
    """Decomposed AdaLN graph annotated for FuseNormPass."""
    mean_1d = reducemeanaxisone_op(x_node)
    mean_2d = broadcastaxisone_op(mean_1d, x_node)
    diff = sub_op(x_node, mean_2d)
    sq = mul_op(diff, diff)
    var_1d = reducemeanaxisone_op(sq)
    var_eps_1d = add_byconst_op(var_1d, eps)
    std_1d = sqrt_op(var_eps_1d)
    std_2d = broadcastaxisone_op(std_1d, x_node)
    x_hat = div_op(diff, std_2d)
    scale_plus_one = add_byconst_op(scale_node, 1.0)
    scaled = mul_op(x_hat, scale_plus_one)
    out = add_op(scaled, shift_node)
    out._fusion_pattern = {
        "kind": "adaln",
        "x": x_node,
        "scale": scale_node,
        "shift": shift_node,
        "eps": eps,
        "internal_nodes": [
            mean_1d, mean_2d, diff, sq, var_1d, var_eps_1d,
            std_1d, std_2d, x_hat, scale_plus_one, scaled, out,
        ],
    }
    return out


class Executor(object):
    """Executor computes values for given set of nodes in computation graph."""
    def __init__(self, eval_node_list, ctx=None,
                 enable_fusion=False,
                 scheduler_policy="default",
                 enable_memory_planning=False,
                 backend="cuda"):
        self.ctx = ctx
        self.enable_fusion = enable_fusion
        self.scheduler_policy = scheduler_policy
        self.enable_memory_planning = enable_memory_planning
        self.backend = backend

        if enable_fusion:
            from .fusion import PassManager
            self.eval_node_list = PassManager.default(backend=self.backend).run(list(eval_node_list))
        else:
            self.eval_node_list = list(eval_node_list)
        self.fused_eval_node_list = self.eval_node_list

        self.topo_order = find_topo_sort(self.eval_node_list)
        for node in self.topo_order:
            node._backend = self.backend

        self.node_to_shape_map = None
        self.node_to_arr_map = None
        self.feed_shapes = None
        self.memory_plan_info = None
        self.last_profile = None

    def infer_shape(self, feed_shapes):
        """Given shapes of feed_dict nodes, infer shape for all nodes in graph."""
        self.node_to_shape_map = {}
        for node in self.topo_order:
            if node in feed_shapes:
                self.node_to_shape_map[node] = feed_shapes[node]
            else:
                input_shapes = [self.node_to_shape_map[n] for n in node.inputs]
                self.node_to_shape_map[node] = node.op.infer_shape(
                    node, input_shapes)

    def memory_plan(self, feed_shapes):
        """Allocates ndarray.NDArray for every node except feed_dict nodes."""
        if self.scheduler_policy == "min_peak_memory":
            from .scheduler import MinPeakMemoryScheduler
            sched = MinPeakMemoryScheduler()
            self.topo_order = sched.schedule(
                self.eval_node_list, self.node_to_shape_map, feed_shapes)
            for node in self.topo_order:
                node._backend = self.backend

        if self.enable_memory_planning:
            from .scheduler import MemoryPlanner
            planner = MemoryPlanner()
            self.node_to_arr_map, self.memory_plan_info = planner.plan(
                self.topo_order,
                self.eval_node_list,
                self.node_to_shape_map,
                feed_shapes,
                self.ctx,
            )
        else:
            from .scheduler import compute_schedule_peak_memory
            self.node_to_arr_map = {}
            for node in self.topo_order:
                if node not in feed_shapes:
                    self.node_to_arr_map[node] = ndarray.empty(
                        self.node_to_shape_map[node], ctx=self.ctx)
            naive_bytes = sum(
                int(np.prod(self.node_to_shape_map[n])) * 4
                for n in self.topo_order if n not in feed_shapes
            )
            self.memory_plan_info = {
                "naive_total_bytes": naive_bytes,
                "planned_pool_bytes": naive_bytes,
                "schedule_peak_live_bytes": compute_schedule_peak_memory(
                    self.topo_order, self.eval_node_list,
                    self.node_to_shape_map, feed_shapes),
                "num_Order_nodes": len(self.topo_order),
                "num_physical_buffers": len(self.node_to_arr_map),
            }

    def run(self, feed_dict, convert_to_numpy_ret_vals=False):
        """
        Parameters
        ----------
        feed_dict: a dictionary of node->np.ndarray supplied by user.
        convert_to_numpy_ret_vals: whether to convert ret vals to np.array

        Returns
        -------
        A list of values for nodes in eval_node_list. NDArray or np.ndarray.
        """
        def are_feed_shapes_equal(sa, sb):
            if (not isinstance(sa, dict)) or (not isinstance(sb, dict)):
                return False
            unmatched_item = set(sa.items()) ^ set(sb.items())
            return len(unmatched_item) == 0

        # Assume self.ctx is None implies numpy array and numpy ops.
        use_numpy = self.ctx is None
        node_to_val_map = {}
        for node, value in feed_dict.items():
            if use_numpy:
                # all values passed in feed_dict must be np.ndarray
                assert isinstance(value, np.ndarray)
                node_to_val_map[node] = value
            else:
                # convert values to ndarray.NDArray if necessary
                if isinstance(value, np.ndarray):
                    node_to_val_map[node] = ndarray.array(value, ctx=self.ctx)
                elif isinstance(value, ndarray.NDArray):
                    node_to_val_map[node] = value
                else:
                    assert False, "feed_dict value type not supported"

        # collect shapes for all placeholders
        feed_shapes = {}
        for node in node_to_val_map:
            feed_shapes[node] = node_to_val_map[node].shape

        # infer shape if feed_shapes changed since last run
        # e.g. call run() on test data after trainng
        if (not are_feed_shapes_equal(feed_shapes, self.feed_shapes)):
            self.infer_shape(feed_shapes)
            self.feed_shapes = feed_shapes
            # plan memory if using GPU
            if (not use_numpy):
                self.memory_plan(feed_shapes)

        # Traverse graph in topo order and compute values for all nodes.
        op_timings = []
        t_start = time.perf_counter()
        for node in self.topo_order:
            if node in node_to_val_map:
                # Skip placeholder nodes. Values already provided by feed_dict.
                continue
            input_vals = [node_to_val_map[n] for n in node.inputs]
            if use_numpy:
                node_val = np.empty(
                    shape=self.node_to_shape_map[node], dtype=np.float32)
            else:
                node_val = self.node_to_arr_map[node]
            t0 = time.perf_counter()
            node._backend = self.backend
            node.op.compute(node, input_vals, node_val, use_numpy)
            t1 = time.perf_counter()
            op_timings.append((node.name, type(node.op).__name__, (t1 - t0) * 1e3))
            node_to_val_map[node] = node_val
        total_ms = (time.perf_counter() - t_start) * 1e3

        mem_stats = {}
        if not use_numpy:
            mem_stats = ndarray.memory_stats(self.ctx)
        self.last_profile = {
            "total_ms": total_ms,
            "num_ops_executed": len(op_timings),
            "op_timings_ms": op_timings,
            "memory_plan_info": self.memory_plan_info,
            "allocator_stats": mem_stats,
        }

        # Collect node values.
        if not use_numpy and convert_to_numpy_ret_vals:
            return [node_to_val_map[n].asnumpy() for n in self.eval_node_list]
        return [node_to_val_map[n] for n in self.eval_node_list]


def gradients(output_node, node_list):
    """Take gradient of output node with respect to each node in node_list."""
    node_to_output_grads_list = {}
    node_to_output_grads_list[output_node] = [oneslike_op(output_node)]
    node_to_output_grad = {}
    skip_nodes = set()
    # Traverse forward graph in reverse topological order
    reverse_topo_order = list(reversed(find_topo_sort([output_node])))
    for node in reverse_topo_order:
        if node in skip_nodes:
            continue
        output_grad = sum_node_list(node_to_output_grads_list[node])
        node_to_output_grad[node] = output_grad
        if isinstance(node.op, PlaceholderOp):
            continue
        meta = getattr(node, "_fusion_pattern", None)
        if isinstance(meta, dict) and meta.get("kind") in ("layernorm", "adaln") and "internal_nodes" in meta:
            internal_list = meta["internal_nodes"]
            internal_set = set(internal_list)
            skip_nodes.update(internal_set)
            local_grads = {node: [output_grad]}
            for sub_node in reversed(internal_list):
                sub_grad = sum_node_list(local_grads[sub_node])
                node_to_output_grad[sub_node] = sub_grad
                sub_in_grads = sub_node.op.gradient(sub_node, sub_grad)
                for i, inp in enumerate(sub_node.inputs):
                    local_grads.setdefault(inp, []).append(sub_in_grads[i])
            if meta["kind"] == "layernorm":
                x_n, g_n, b_n, eps_val = meta["x"], meta["gamma"], meta["beta"], meta["eps"]
                dx_u = sum_node_list(local_grads[x_n])
                dg_u = sum_node_list(local_grads[g_n])
                db_u = sum_node_list(local_grads[b_n])
                dx_u._fusion_pattern = {"kind": "layernorm_grad", "which": "dx", "dy": output_grad, "x": x_n, "gamma": g_n, "eps": eps_val}
                dg_u._fusion_pattern = {"kind": "layernorm_grad", "which": "dgamma", "dy": output_grad, "x": x_n, "gamma": g_n, "eps": eps_val}
                db_u._fusion_pattern = {"kind": "layernorm_grad", "which": "dbeta", "dy": output_grad, "x": x_n, "gamma": g_n, "eps": eps_val}
                node_to_output_grads_list.setdefault(x_n, []).append(dx_u)
                node_to_output_grads_list.setdefault(g_n, []).append(dg_u)
                node_to_output_grads_list.setdefault(b_n, []).append(db_u)
            else:
                x_n, s_n, sh_n, eps_val = meta["x"], meta["scale"], meta["shift"], meta["eps"]
                dx_u = sum_node_list(local_grads[x_n])
                ds_u = sum_node_list(local_grads[s_n])
                dsh_u = sum_node_list(local_grads[sh_n])
                dx_u._fusion_pattern = {"kind": "adaln_grad", "which": "dx", "dy": output_grad, "x": x_n, "scale": s_n, "eps": eps_val}
                ds_u._fusion_pattern = {"kind": "adaln_grad", "which": "dscale", "dy": output_grad, "x": x_n, "scale": s_n, "eps": eps_val}
                dsh_u._fusion_pattern = {"kind": "adaln_grad", "which": "dshift", "dy": output_grad, "x": x_n, "scale": s_n, "eps": eps_val}
                node_to_output_grads_list.setdefault(x_n, []).append(dx_u)
                node_to_output_grads_list.setdefault(s_n, []).append(ds_u)
                node_to_output_grads_list.setdefault(sh_n, []).append(dsh_u)
            continue
        input_grads_list = node.op.gradient(node, output_grad)
        for i in range(len(node.inputs)):
            if node.inputs[i] not in node_to_output_grads_list:
                node_to_output_grads_list[node.inputs[i]] = []
            # Calculate partial adjoint for input nodes.
            node_to_output_grads_list[node.inputs[i]].append(
                input_grads_list[i])

    grad_node_list = [node_to_output_grad[node] for node in node_list]
    return grad_node_list

##################
# Helper Methods #
##################


def find_topo_sort(node_list):
    """Given a list of nodes, return a topo ordering of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a
    topological sort.

    """
    visited = set()
    topo_order = []
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)


def sum_node_list(node_list):
    """Custom sum func to avoid creating redundant nodes in Python sum func."""
    from operator import add
    from functools import reduce
    return reduce(add, node_list)


def broadcast_rule(shape_a, shape_b):
    """Return output shape of broadcast shape_a, shape_b.
    e.g. broadcast_rule((3,2), (4,3,2))
    returns output_shape = (4,3,2)

    Check out explanations and more examples at
    https://docs.scipy.org/doc/numpy-1.10.0/user/basics.broadcasting.html
    http://eli.thegreenplace.net/2015/broadcasting-arrays-in-numpy/
    """
    assert(isinstance(shape_a, tuple))
    assert(isinstance(shape_b, tuple))
    if len(shape_a) > len(shape_b):
        longer_shape, shorter_shape = shape_a, shape_b
    else:
        longer_shape, shorter_shape = shape_b, shape_a
    len_diff = len(longer_shape) - len(shorter_shape)
    for i in range(len_diff):
        # pad with leading 1s
        shorter_shape = (1,) + shorter_shape
    assert len(shorter_shape) == len(longer_shape)
    output_shape = list(longer_shape)
    for i in range(len(output_shape)):
        assert (shorter_shape[i] == longer_shape[i]) \
            or (shorter_shape[i] == 1) \
            or (longer_shape[i] == 1)
        output_shape[i] = max(shorter_shape[i], longer_shape[i])
    return tuple(output_shape)
