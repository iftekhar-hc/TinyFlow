from __future__ import absolute_import

import numpy as np
from . import ndarray


def align_bytes(nbytes, alignment=256):
    """Align byte count up to a multiple of `alignment` (default 256 bytes)."""
    if nbytes <= 0:
        return 0
    return ((int(nbytes) + alignment - 1) // alignment) * alignment


def shape_to_bytes(shape, aligned=True):
    """Compute float32 tensor size in bytes from a shape tuple."""
    if shape is None:
        return 0
    numel = int(np.prod(shape)) if len(shape) > 0 else 1
    raw = numel * 4
    return align_bytes(raw) if aligned else raw


class LivenessAnalyzer(object):
    """Analyzes tensor lifetimes and simulates live memory across an execution schedule."""

    def analyze(self, topo_order, node_to_shape_map, feed_nodes=None, eval_nodes=None):
        feed_set = set(feed_nodes or [])
        eval_set = set(eval_nodes or [])

        first_def = {}
        last_use = {}
        for step, node in enumerate(topo_order):
            first_def[node] = step
            last_use[node] = step
            for inp in node.inputs:
                if inp in last_use:
                    last_use[inp] = max(last_use[inp], step)
                else:
                    first_def[inp] = -1
                    last_use[inp] = step

        final_step = len(topo_order)
        for node in eval_set:
            last_use[node] = max(last_use.get(node, final_step), final_step)
        for node in feed_set:
            last_use[node] = max(last_use.get(node, final_step), final_step)

        live_set = set(feed_set)
        live_bytes = sum(shape_to_bytes(node_to_shape_map.get(n), aligned=False) for n in live_set)
        live_aligned_bytes = sum(shape_to_bytes(node_to_shape_map.get(n), aligned=True) for n in live_set)
        peak_bytes = live_bytes
        peak_aligned_bytes = live_aligned_bytes
        timeline = []

        for step, node in enumerate(topo_order):
            if node not in live_set:
                live_set.add(node)
                live_bytes += shape_to_bytes(node_to_shape_map.get(node), aligned=False)
                live_aligned_bytes += shape_to_bytes(node_to_shape_map.get(node), aligned=True)

            if live_bytes > peak_bytes:
                peak_bytes = live_bytes
            if live_aligned_bytes > peak_aligned_bytes:
                peak_aligned_bytes = live_aligned_bytes

            timeline.append({
                "step": step,
                "node": node.name,
                "live_bytes": live_bytes,
                "live_aligned_bytes": live_aligned_bytes,
                "num_live_tensors": len(live_set),
            })

            to_free = []
            for live_node in list(live_set):
                if live_node in feed_set or live_node in eval_set:
                    continue
                if last_use.get(live_node, step) <= step:
                    to_free.append(live_node)
            for dead_node in to_free:
                live_set.remove(dead_node)
                live_bytes -= shape_to_bytes(node_to_shape_map.get(dead_node), aligned=False)
                live_aligned_bytes -= shape_to_bytes(node_to_shape_map.get(dead_node), aligned=True)

        static_bytes = sum(
            shape_to_bytes(node_to_shape_map.get(n), aligned=False)
            for n in topo_order
        )
        static_aligned_bytes = sum(
            shape_to_bytes(node_to_shape_map.get(n), aligned=True)
            for n in topo_order
        )

        return {
            "first_def": first_def,
            "last_use": last_use,
            "timeline": timeline,
            "peak_bytes": peak_bytes,
            "peak_aligned_bytes": peak_aligned_bytes,
            "static_bytes": static_bytes,
            "static_aligned_bytes": static_aligned_bytes,
        }


class MinPeakMemoryScheduler(object):
    """Greedy memory-aware topological scheduler that minimizes peak live tensor bytes."""

    def schedule(self, eval_node_list, node_to_shape_map=None, feed_shapes=None):
        shape_map = node_to_shape_map or {}

        visited = set()
        all_nodes = []

        def _dfs(node):
            if node in visited:
                return
            visited.add(node)
            for inp in node.inputs:
                _dfs(inp)
            all_nodes.append(node)

        for n in eval_node_list:
            _dfs(n)

        if len(all_nodes) <= 1:
            return all_nodes

        in_degree = {n: 0 for n in all_nodes}
        consumers = {n: [] for n in all_nodes}
        for n in all_nodes:
            unique_inputs = set(n.inputs)
            in_degree[n] = len(unique_inputs)
            for inp in unique_inputs:
                consumers[inp].append(n)

        eval_set = set(eval_node_list)
        remaining_uses = {n: len(consumers[n]) + (1 if n in eval_set else 0) for n in all_nodes}
        default_rank = {n: idx for idx, n in enumerate(all_nodes)}

        ready = [n for n in all_nodes if in_degree[n] == 0]
        schedule_order = []

        while ready:
            best_idx = 0
            best_key = None
            for idx, cand in enumerate(ready):
                out_size = shape_to_bytes(shape_map.get(cand), aligned=True)
                freed_size = 0
                for inp in set(cand.inputs):
                    if remaining_uses.get(inp, 0) == 1:
                        freed_size += shape_to_bytes(shape_map.get(inp), aligned=True)
                net_delta = out_size - freed_size
                is_non_placeholder = 0 if len(cand.inputs) == 0 else 1
                key = (is_non_placeholder, net_delta, -freed_size, default_rank[cand])
                if best_key is None or key < best_key:
                    best_key = key
                    best_idx = idx

            chosen = ready.pop(best_idx)
            schedule_order.append(chosen)

            for inp in set(chosen.inputs):
                remaining_uses[inp] -= 1

            for cons in consumers[chosen]:
                in_degree[cons] -= 1
                if in_degree[cons] == 0:
                    ready.append(cons)

        return schedule_order


def compute_schedule_peak_memory(topo_order, eval_node_list, node_to_shape_map, feed_shapes=None):
    """Compute peak live bytes across a topological execution order."""
    feed_nodes = list(feed_shapes.keys()) if isinstance(feed_shapes, dict) else list(feed_shapes or [])
    analyzer = LivenessAnalyzer()
    info = analyzer.analyze(topo_order, node_to_shape_map, feed_nodes=feed_nodes, eval_nodes=eval_node_list)
    return info["peak_bytes"]


class MemoryPlanner(object):
    """Liveness-based static buffer reuse planner for intermediate graph tensors."""

    def __init__(self):
        self.analyzer = LivenessAnalyzer()

    def plan(self, topo_order, eval_node_list_or_shape_map, node_to_shape_map_or_feed, feed_shapes_or_eval, ctx):
        if isinstance(eval_node_list_or_shape_map, dict):
            node_to_shape_map = eval_node_list_or_shape_map
            feed_nodes = node_to_shape_map_or_feed
            eval_nodes = feed_shapes_or_eval
            return_tuple = False
        else:
            eval_nodes = eval_node_list_or_shape_map
            node_to_shape_map = node_to_shape_map_or_feed
            feed_nodes = list(feed_shapes_or_eval.keys()) if isinstance(feed_shapes_or_eval, dict) else list(feed_shapes_or_eval or [])
            return_tuple = True

        feed_set = set(feed_nodes or [])
        eval_set = set(eval_nodes or [])
        liveness = self.analyzer.analyze(topo_order, node_to_shape_map, feed_set, eval_set)
        last_use = liveness["last_use"]

        # Pool of available NDArray buffers keyed by exact shape
        free_by_shape = {}
        node_to_arr = {}
        active_nodes = set()
        allocated_buffers = []

        for step, node in enumerate(topo_order):
            if node in feed_set:
                continue
            shape = tuple(node_to_shape_map[node])
            if node not in eval_set and free_by_shape.get(shape):
                arr = free_by_shape[shape].pop()
            else:
                arr = ndarray.empty(shape, ctx=ctx)
                allocated_buffers.append(arr)
            node_to_arr[node] = arr
            active_nodes.add(node)

            # Release any intermediate tensor whose last use is this step
            to_release = []
            for live_node in list(active_nodes):
                if live_node in feed_set or live_node in eval_set:
                    continue
                if last_use.get(live_node, step) <= step:
                    to_release.append(live_node)
            for dead_node in to_release:
                active_nodes.remove(dead_node)
                dead_shape = tuple(node_to_shape_map[dead_node])
                free_by_shape.setdefault(dead_shape, []).append(node_to_arr[dead_node])

        planned_bytes = sum(arr.nbytes for arr in allocated_buffers)
        naive_bytes = sum(
            shape_to_bytes(node_to_shape_map.get(n), aligned=False)
            for n in topo_order if n not in feed_set
        )
        num_inter = sum(1 for n in topo_order if n not in feed_set)
        info = {
            "node_to_arr_map": node_to_arr,
            "liveness": liveness,
            "num_unique_buffers": len(allocated_buffers),
            "num_physical_buffers": len(allocated_buffers),
            "num_shared_buffers": len(allocated_buffers),
            "num_intermediate_nodes": num_inter,
            "num_nodes": len(topo_order),
            "num_Order_nodes": len(topo_order),
            "planned_bytes": planned_bytes,
            "planned_pool_bytes": planned_bytes,
            "static_bytes": liveness["static_bytes"],
            "naive_total_bytes": naive_bytes,
            "peak_live_bytes": liveness["peak_bytes"],
            "schedule_peak_live_bytes": liveness["peak_bytes"],
            "peak_aligned_bytes": liveness["peak_aligned_bytes"],
            "memory_saved_bytes": max(0, naive_bytes - planned_bytes),
            "memory_reduction_ratio": (
                float(naive_bytes - planned_bytes) / float(naive_bytes)
                if naive_bytes > 0 else 0.0
            ),
        }
        if return_tuple:
            return node_to_arr, info
        return info

