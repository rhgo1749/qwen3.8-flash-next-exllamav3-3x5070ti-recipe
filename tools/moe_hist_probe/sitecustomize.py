"""Experimental routing-histogram probe for ExLlamaV3 1.5.1.

Load with Python's sitecustomize mechanism (for example by placing this file on
PYTHONPATH) and set EXL3_MOE_HIST_PROBE=1.

This hooks ExLlamaV3 internals. Expect to review it after upgrading ExLlamaV3.
"""

import json
import os
import statistics
from pathlib import Path

if os.environ.get("EXL3_MOE_HIST_PROBE"):
    try:
        import torch as _torch
        from exllamav3.modules import block_sparse_mlp_cpu as _m

        _out = Path(
            os.environ.get(
                "EXL3_MOE_HIST_OUT",
                "/tmp/qwen38-routing-stats.json",
            )
        )

        def _snapshot(reg):
            stats = {}
            rows = []
            total_mass = current_cpu_mass = ideal_cpu_mass = 0.0

            for mod in reg or []:
                hist_t = getattr(mod, "_split_hist", None)
                map_t = getattr(mod, "_split_map", None)
                first = getattr(mod, "cpu_split_first", None)
                key = str(getattr(mod, "key", ""))

                if not key or hist_t is None or map_t is None or first is None:
                    continue

                hist = hist_t.detach().float().cpu()
                mp = map_t.detach().cpu()
                mass = float(hist.sum().item())
                if mass <= 0.0:
                    continue

                vals = [float(x) for x in hist.tolist()]
                prev = stats.get(key)
                if prev is None:
                    stats[key] = vals
                else:
                    stats[key] = [a + b for a, b in zip(prev, vals)]

                gpu_mask = mp < int(first)
                gpu_slots = int(gpu_mask.sum().item())
                if gpu_slots <= 0:
                    continue

                cur_gpu = float(hist[gpu_mask].sum().item())
                ideal_gpu = float(
                    hist.topk(gpu_slots, largest=True).values.sum().item()
                )
                cur_cpu = max(0.0, mass - cur_gpu)
                ideal_cpu = max(0.0, mass - ideal_gpu)

                total_mass += mass
                current_cpu_mass += cur_cpu
                ideal_cpu_mass += ideal_cpu
                rows.append((cur_cpu / mass, ideal_cpu / mass))

            if not stats or total_mass <= 0.0:
                return None

            ideals = sorted(x[1] for x in rows)
            currents = sorted(x[0] for x in rows)
            wc = current_cpu_mass / total_mass
            wi = ideal_cpu_mass / total_mass

            return stats, {
                "keys": len(stats),
                "modules": len(rows),
                "mass": total_mass,
                "cpu_current": wc,
                "cpu_ideal": wi,
                "gpu_current": 1.0 - wc,
                "gpu_ideal": 1.0 - wi,
                "ideal_med": statistics.median(ideals),
                "current_med": statistics.median(currents),
            }

        def _merge_write(stats):
            """Atomically update sampled keys while preserving existing keys."""
            _out.parent.mkdir(parents=True, exist_ok=True)
            merged = {}

            if _out.exists():
                try:
                    loaded = json.loads(_out.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        merged.update(loaded)
                except Exception as exc:
                    print(
                        " -- MOE_HIST_PROBE "
                        f"merge_read_error={type(exc).__name__}:{exc}",
                        flush=True,
                    )

            merged.update(stats)
            tmp = _out.with_suffix(_out.suffix + ".tmp")
            tmp.write_text(
                json.dumps(merged, separators=(",", ":")),
                encoding="utf-8",
            )
            os.replace(tmp, _out)

        def _emit(reg, reason):
            snap = _snapshot(reg)
            if not snap:
                return

            stats, metrics = snap
            _merge_write(stats)

            print(
                " -- MOE_HIST_PROBE "
                f"reason={reason} "
                f"keys={metrics['keys']} "
                f"modules={metrics['modules']} "
                f"mass={metrics['mass']:.0f} "
                f"cpu_current={metrics['cpu_current'] * 100:.2f}% "
                f"cpu_ideal={metrics['cpu_ideal'] * 100:.2f}% "
                f"gpu_hit_current={metrics['gpu_current'] * 100:.2f}% "
                f"gpu_hit_ideal={metrics['gpu_ideal'] * 100:.2f}% "
                f"ideal_layer_med={metrics['ideal_med'] * 100:.2f}% "
                f"current_layer_med={metrics['current_med'] * 100:.2f}% "
                f"stats={_out}",
                flush=True,
            )

        # Static placement has no upstream _split_hist because CPU swap is off.
        # For a selected key prefix, collect a passive histogram directly from
        # selected expert IDs without changing placement.
        _passive_prefix = os.environ.get("EXL3_MOE_HIST_PREFIX", "mtp.")
        _passive_counts = {}
        _passive_calls = {}

        def _passive_sample(self, selected_experts):
            key = str(getattr(self, "key", ""))
            if not key.startswith(_passive_prefix):
                return
            if getattr(self, "_split_map", None) is not None:
                return

            n_experts = int(getattr(self, "num_experts", 0))
            first = getattr(self, "cpu_split_first", None)
            if not n_experts or first is None:
                return

            ids = selected_experts.reshape(-1).long()
            perm = getattr(self, "_split_perm", None)

            if perm is not None:
                perm_t = getattr(self, "_hist_probe_perm_t", None)
                if perm_t is None or perm_t.device != ids.device:
                    perm_t = _torch.tensor(
                        perm,
                        dtype=_torch.long,
                        device=ids.device,
                    )
                    self._hist_probe_perm_t = perm_t
                ids = perm_t[ids]

            hist = _passive_counts.get(key)
            if hist is None or hist.device != ids.device:
                hist = _torch.zeros(
                    n_experts,
                    dtype=_torch.float,
                    device=ids.device,
                )
                _passive_counts[key] = hist

            hist.add_(
                _torch.bincount(ids, minlength=n_experts).to(hist.dtype)
            )

            calls = _passive_calls.get(key, 0) + 1
            _passive_calls[key] = calls
            if calls % 128:
                return

            vals_t = hist.detach().float().cpu()
            vals = [float(x) for x in vals_t.tolist()]
            mass = float(vals_t.sum().item())
            if mass <= 0.0:
                return

            if perm is None:
                gpu_ids = list(range(int(first)))
            else:
                gpu_ids = list(perm[: int(first)])

            cur_gpu = float(vals_t[gpu_ids].sum().item())
            ideal_gpu = float(
                vals_t.topk(int(first), largest=True).values.sum().item()
            )

            cpu_current = max(0.0, mass - cur_gpu) / mass
            cpu_ideal = max(0.0, mass - ideal_gpu) / mass

            _merge_write({key: vals})

            print(
                " -- MOE_HIST_PROBE_STATIC "
                f"key={key} calls={calls} mass={mass:.0f} "
                f"cpu_current={cpu_current * 100:.2f}% "
                f"cpu_ideal={cpu_ideal * 100:.2f}% "
                f"stats={_out}",
                flush=True,
            )

        _orig_translate = _m.BlockSparseMLP_CPU._split_translate
        _orig_split_submit = _m.BlockSparseMLP_CPU.cpu_split_submit
        _counts = {"prefill": 0, "decode": 0}

        def _probe_split_submit(
            self,
            y,
            bsz,
            selected_experts,
            routing_weights,
        ):
            # Sampling here catches the fused issue path used by decode.
            try:
                _passive_sample(self, selected_experts)
            except Exception as exc:
                print(
                    " -- MOE_HIST_PROBE "
                    f"passive_error={type(exc).__name__}:{exc}",
                    flush=True,
                )

            return _orig_split_submit(
                self,
                y,
                bsz,
                selected_experts,
                routing_weights,
            )

        def _probe_translate(self, selected_experts):
            out = _orig_translate(self, selected_experts)
            try:
                ip = self.config.infer_params
                reg = getattr(ip, "moe_cpu_swap_modules", None) or []
                if reg and reg[0] is self:
                    mode = (
                        "prefill"
                        if int(selected_experts.numel()) > 128
                        else "decode"
                    )
                    _counts[mode] += 1
                    interval = 4 if mode == "prefill" else 128
                    if _counts[mode] % interval == 0:
                        _emit(reg, mode)
            except Exception as exc:
                print(
                    " -- MOE_HIST_PROBE "
                    f"error={type(exc).__name__}:{exc}",
                    flush=True,
                )
            return out

        _m.BlockSparseMLP_CPU.cpu_split_submit = _probe_split_submit
        _m.BlockSparseMLP_CPU._split_translate = _probe_translate

        print(
            f" -- MOE_HIST_PROBE enabled out={_out}",
            flush=True,
        )

    except Exception as exc:
        print(
            " -- MOE_HIST_PROBE "
            f"init_error={type(exc).__name__}:{exc}",
            flush=True,
        )
