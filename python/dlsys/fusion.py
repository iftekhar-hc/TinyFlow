from __future__ import absolute_import


class GraphPass(object):
    """Base class for computation graph transformation passes."""

    def apply(self, eval_node_list):
        raise NotImplementedError()


class PassManager(object):
    """Runs a sequence of graph optimization passes on an evaluation node list."""

    def __init__(self, passes=None):
        self.passes = list(passes or [])

    def add_pass(self, graph_pass):
        self.passes.append(graph_pass)

    def run(self, eval_node_list):
        nodes = list(eval_node_list)
        for p in self.passes:
            nodes = p.apply(nodes)
        return nodes

    @classmethod
    def default(cls, backend="cuda"):
        return default_fusion_pass_manager(backend=backend)


def _collect_consumers(eval_node_list):
    """Return (all_nodes_postorder, consumers_map) for the reachable DAG."""
    visited = set()
    postorder = []
    consumers = {}

    def _dfs(node):
        if node in visited:
            return
        visited.add(node)
        if node not in consumers:
            consumers[node] = set()
        for inp in node.inputs:
            if inp not in consumers:
                consumers[inp] = set()
            consumers[inp].add(node)
            _dfs(inp)
        postorder.append(node)

    for n in eval_node_list:
        _dfs(n)
    return postorder, consumers


class DeadCodeEliminationPass(GraphPass):
    """Prunes unreachable dead branches and normalizes the reachable subgraph."""

    def apply(self, eval_node_list):
        postorder, _ = _collect_consumers(eval_node_list)
        reachable = set(postorder)
        return [n for n in eval_node_list if n in reachable]


def _clone_rewritten_dag(eval_node_list, replacements):
    """Rebuild DAG bottom-up applying `replacements` map (old_node -> new_node) without mutating original nodes."""
    import copy
    memo = {}

    def _visit(node):
        if node in memo:
            return memo[node]
        target = replacements.get(node, node)
        changed = False
        new_inputs = []
        for inp in target.inputs:
            new_inp = _visit(inp)
            if new_inp is not inp:
                changed = True
            new_inputs.append(new_inp)
        meta = getattr(target, "_fusion_pattern", None)
        new_meta = None
        if isinstance(meta, dict):
            new_meta = dict(meta)
            for k in ("x", "gamma", "beta", "scale", "shift", "dy", "logits", "labels", "out_grad"):
                if k in new_meta and new_meta[k] is not None:
                    visited_k = _visit(new_meta[k])
                    if visited_k is not new_meta[k]:
                        changed = True
                    new_meta[k] = visited_k
        if changed and target is node:
            target = copy.copy(node)
        if changed:
            target.inputs = new_inputs
            if new_meta is not None:
                target._fusion_pattern = new_meta
        memo[node] = target
        memo[target] = target
        return target

    return [_visit(n) for n in eval_node_list]


class FuseLinearBiasActPass(GraphPass):
    """Fuses MatMul(X, W, False, False) + BroadcastTo(b) [+ ReLU/GELU] into FusedLinearBiasActOp."""

    def __init__(self, backend="cuda", fuse_in_training=False):
        self.backend = backend
        self.fuse_in_training = fuse_in_training

    def apply(self, eval_node_list):
        from . import autodiff as ad

        postorder, consumers = _collect_consumers(eval_node_list)
        eval_set = set(eval_node_list)
        replacements = {}

        def _match_add_bias_matmul(add_node):
            if not isinstance(add_node.op, ad.AddOp):
                return None
            lhs, rhs = add_node.inputs
            if isinstance(lhs.op, ad.MatMulOp) and isinstance(rhs.op, ad.BroadcastToOp):
                mm_node, bcast_node = lhs, rhs
            elif isinstance(rhs.op, ad.MatMulOp) and isinstance(lhs.op, ad.BroadcastToOp):
                mm_node, bcast_node = rhs, lhs
            else:
                return None
            if getattr(mm_node, "matmul_attr_trans_A", False) or getattr(mm_node, "matmul_attr_trans_B", False):
                return None
            if bcast_node.inputs[1] is not mm_node:
                return None
            mm_cons = consumers.get(mm_node, set())
            if not mm_cons.issubset({add_node, bcast_node}) or mm_node in eval_set:
                return None
            if consumers.get(bcast_node, set()) != {add_node} or bcast_node in eval_set:
                return None
            x_node, w_node = mm_node.inputs
            bias_node = bcast_node.inputs[0]
            return (x_node, w_node, bias_node, mm_node, bcast_node)

        for node in postorder:
            if node in replacements:
                continue
            if isinstance(node.op, (ad.ReluOp, ad.GeluOp)):
                add_cand = node.inputs[0]
                if add_cand in replacements:
                    continue
                if consumers.get(add_cand, set()) == {node} and add_cand not in eval_set:
                    matched = _match_add_bias_matmul(add_cand)
                    if matched is not None:
                        x_node, w_node, bias_node, _, _ = matched
                        act_type = "relu" if isinstance(node.op, ad.ReluOp) else "gelu"
                        replacements[node] = ad.fused_linear_op(
                            x_node, w_node, bias_node, act_type=act_type, backend=self.backend)
                        replacements[add_cand] = replacements[node]

        for node in postorder:
            if node in replacements:
                continue
            matched = _match_add_bias_matmul(node)
            if matched is not None:
                node_cons = consumers.get(node, set())
                if any(c in replacements for c in node_cons):
                    continue
                x_node, w_node, bias_node, _, _ = matched
                replacements[node] = ad.fused_linear_op(
                    x_node, w_node, bias_node, act_type="none", backend=self.backend)

        if not replacements:
            return eval_node_list
        return _clone_rewritten_dag(eval_node_list, replacements)


class FuseLayerNormAdaLNPass(GraphPass):
    """Fuses decomposed LayerNorm / AdaLN subgraphs into single-kernel LayerNormOp / AdaLNOp."""

    def __init__(self, backend="cuda"):
        self.backend = backend

    def apply(self, eval_node_list):
        from . import autodiff as ad

        postorder, _ = _collect_consumers(eval_node_list)
        replacements = {}
        for node in postorder:
            fusion_meta = getattr(node, "_fusion_pattern", None)
            if fusion_meta is None:
                continue
            kind = fusion_meta.get("kind")
            if kind == "layernorm":
                replacements[node] = ad.layernorm_op(
                    fusion_meta["x"],
                    fusion_meta["gamma"],
                    fusion_meta["beta"],
                    eps=fusion_meta.get("eps", 1e-5),
                    backend=self.backend,
                )
            elif kind == "adaln":
                replacements[node] = ad.adaln_op(
                    fusion_meta["x"],
                    fusion_meta["scale"],
                    fusion_meta["shift"],
                    eps=fusion_meta.get("eps", 1e-5),
                    backend=self.backend,
                )
            elif kind == "layernorm_grad":
                replacements[node] = ad.layernorm_grad_op(
                    fusion_meta["dy"],
                    fusion_meta["x"],
                    fusion_meta["gamma"],
                    eps=fusion_meta.get("eps", 1e-5),
                    which=fusion_meta.get("which", "dx"),
                    backend=self.backend,
                )
            elif kind == "adaln_grad":
                replacements[node] = ad.adaln_grad_op(
                    fusion_meta["dy"],
                    fusion_meta["x"],
                    fusion_meta["scale"],
                    eps=fusion_meta.get("eps", 1e-5),
                    which=fusion_meta.get("which", "dx"),
                    backend=self.backend,
                )
        if not replacements:
            return eval_node_list
        return _clone_rewritten_dag(eval_node_list, replacements)


class FuseCrossEntropyGradPass(GraphPass):
    """Fuses decomposed SoftmaxCrossEntropy backward subgraph into SoftmaxCrossEntropyGradOp."""

    def __init__(self, backend="cuda"):
        self.backend = backend

    def apply(self, eval_node_list):
        from . import autodiff as ad

        postorder, _ = _collect_consumers(eval_node_list)
        replacements = {}
        for node in postorder:
            fusion_meta = getattr(node, "_fusion_pattern", None)
            if fusion_meta is not None and fusion_meta.get("kind") == "softmax_ce_grad":
                replacements[node] = ad.softmax_cross_entropy_grad_op(
                    fusion_meta["logits"],
                    fusion_meta["labels"],
                    fusion_meta["out_grad"],
                    backend=self.backend,
                )
        if not replacements:
            return eval_node_list
        return _clone_rewritten_dag(eval_node_list, replacements)


def default_fusion_pass_manager(backend="cuda"):
    """Construct the standard TinyFlow graph optimization PassManager."""
    return PassManager([
        FuseLinearBiasActPass(backend=backend),
        FuseLayerNormAdaLNPass(backend=backend),
        FuseCrossEntropyGradPass(backend=backend),
        DeadCodeEliminationPass(),
    ])

