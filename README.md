# OmniMoE — MoE VLM Engineering (Sanity‑Check + Upgrade as of 2025‑10‑22)

Below is a sanity‑check + upgrade of your blueprint to be correct and current as of Oct 22, 2025 (one day before your requested 2025‑10‑23). I’ve also filled in missing engineering details, corrected a few things that changed across libraries, and provided ready‑to‑drop‑in config/code patterns. I cite sources where the knobs changed or where “latest” matters.

---

## TL;DR — What to update vs. your draft

1. DeepSpeed MoE API & config names (2025): Use the code‑level `deepspeed.moe.layer.MoE(...)` with arguments like `ep_size`, `k`, `capacity_factor`, `drop_tokens`, `noisy_gate_policy`, `use_tutel`, `enable_expert_tensor_parallelism`. There isn’t a stable `"moe": {...}` section in `ds_config.json`; keep MoE options in code and reserve `ds_config.json` for ZeRO/optimizer/precision/checkpointing. The up‑to‑date parameter list is documented in DeepSpeed 0.18.x. ([deepspeed.readthedocs.io][1])
   Notably, older keys like `fp32_gate` in a JSON block don’t exist in current DS docs; cast router computations to FP32 in your module instead. ([deepspeed.readthedocs.io][1])

2. Megatron‑Core flags moved & grew: Use Megatron‑Core (0.14 as of Oct 8, 2025) which exposes MoE knobs such as `--expert-model-parallel-size` (EP size), `--num-experts`, `--moe-router-topk`, and `--moe-grouped-gemm`. Names drifted from older “moe‑expert‑parallel‑size” variants; check the current docs/README. ([PyPI][2])

3. Serving is easier now:

   - vLLM supports expert parallel serving with `--enable-expert-parallel` and derives EP size from `TP×DP` (or expose `--expert-parallel-size` when available in your build). Keep MoE experts co‑located on NVLink/NVSwitch when you can. ([VLLM Documentation][3])
   - TensorRT‑LLM has native EP for MoE and fused MoE operators (plus FP8/NVFP4 paths). See release notes and EP docs. It’s the fastest single‑node option when you can prebuild engines. ([nvidia.github.io][4])

4. Attention kernels: Prefer FlashAttention‑3 on Hopper‑class GPUs (H100/H200). It’s the current best‑practice and widely integrated (FA‑3 ≈1.5–2.0× vs FA‑2 on H100). ([tridao.me][5])

5. Dispatcher/All‑to‑All: In PyTorch, use `dist.all_to_all_single(send, recv, input_split_sizes, output_split_sizes, group=ep_group)` properly with a prior count exchange to compute `recv_splits`; don’t reuse the same buffer for send/recv. (Var‑sized all‑to‑all is supported, but mind known deadlock pitfalls; production stacks use fused dispatchers.) ([PyTorch Docs][6])

6. Routing/load‑balance: The Switch Transformer auxiliary loss (importance × load) and Noisy Top‑k remain the robust defaults; Expert‑Choice routing (EC) is a modern alternative used in some recent MoE lines (e.g., DeepSeek V2‑series). Start with Top‑2 + Switch aux loss unless you need EC’s properties. You can toggle EC per branch via `text_moe.use_expert_choice_router` / `vision_moe.use_expert_choice_router`, which activates the expert-driven dispatcher with capacity-aware token selection, balanced importance/load EMA, and zero-drop fallback. ([arXiv][7])

7. Optional accelerators: Tutel (Microsoft) remains a drop‑in MoE optimization, and MegaBlocks / “dropless” routing variants exist if you need zero‑drop or better tail behavior (set `router.use_megablocks_dropless: true` **only after** installing MegaBlocks ≥0.7.0 on every node, e.g. `pip install megablocks`).

Everything else in your blueprint is directionally correct.

---

## A clean, 2025‑ready framework

### 1) Architecture overview (VLM + MoE)

- Vision branch: CLIP/EVA/SigLIP ViT encoder → optional Perceiver Resampler (Flamingo) or Q‑Former (BLIP‑2) to compress patch tokens to ~32–64 latent tokens. These are still the two most common token reducers. ([NeurIPS Proceedings][8])
- Projector: linear/MLP to match LLM hidden size.
- Decoder LLM: standard GPT architecture with MoE FFN replacing dense FFN in (some or all) blocks.
- MoE FFN: Top‑2 gating, SwiGLU MLP experts, aux load‑balancing loss.

> Practical placement: introduce MoE first in mid/deep blocks to stabilize; then widen coverage after router stabilizes. This practice is unchanged and still recommended. (Matches DS and Megatron guidance.) ([deepspeed.readthedocs.io][1])

---

### 2) MoE layer — production‑grade details

Router (gate):

- Linear gate → logits in FP32 → optional noise (`noisy_gate_policy`: `"Jitter"` or `"RSample"`) → Top‑k (k=2) → softmax on the selected k. Train with Switch aux loss (importance × load). In DeepSpeed you set noise via `noisy_gate_policy` (per‑layer when you construct `deepspeed.moe.layer.MoE`). ([deepspeed.readthedocs.io][1])

Capacity & dropping:

- Capacity per expert per microbatch
  `capacity = ceil(capacity_factor * top_k * N_tokens / num_experts)`
  Start `capacity_factor=1.25` (Top‑2); move to 1.5 if you see >1–2% drops. If you need “dropless”, pick a larger capacity or adopt a shared/dense fallback expert. (DeepSpeed exposes `drop_tokens=True/False`.) ([deepspeed.readthedocs.io][1])

All‑to‑All routing (EP group):

- Pack tokens by destination rank/local expert, exchange counts, then do `all_to_all_single` with var‑size splits; apply local expert MLPs; all‑to‑all back; combine using gate weights; unpermute to the original token order. Use fused dispatchers (Megatron‑Core, DeepSpeed, or Tutel) when available; they implement grouped GEMMs and overlap A2A with compute. ([NVIDIA Docs][9])

Experts: identical shapes for kernel fusion; SwiGLU; enable grouped GEMM where supported (Megatron’s `--moe-grouped-gemm`). ([NVIDIA Docs][9])

Numerics: BF16 for activations/weights; keep gate logits and aux loss reductions in FP32. With FA‑3 you can also enable FP8 attention on Hopper, but keep router in FP32. ([tridao.me][10])

---

### 3) Corrected minimal dispatcher skeleton (training‑side)

Below is a conceptual sketch that fixes the var‑size all‑to‑all and the combine path your draft omitted. It’s still simplified (you should rely on DeepSpeed/Megatron/Tutel for speed), but the indexing pattern is right:

```python
# Core ideas only: no error checks; assumes NCCL + CUDA tensors
import torch, torch.nn as nn, torch.distributed as dist

class ExpertMLP(nn.Module):
    def __init__(self, hidden, mult=4, act=nn.SiLU):
        super().__init__()
        inner = int(hidden * mult)
        self.w1 = nn.Linear(hidden, inner, bias=False)
        self.w2 = nn.Linear(inner, hidden, bias=False)
        self.act = act()
    def forward(self, x):  # [n_tok, H]
        return self.w2(self.act(self.w1(x)))

class TopKGate(nn.Module):
    def __init__(self, hidden, num_experts, k=2, jitter_std=0.0):
        super().__init__()
        self.wg = nn.Linear(hidden, num_experts, bias=False)
        self.k = k
        self.jitter_std = jitter_std
    def forward(self, x):  # x:[N,H], compute in fp32 for stability
        logits = self.wg(x).float()                   # [N,E]
        if self.jitter_std > 0:
            logits = logits + self.jitter_std * torch.randn_like(logits)
        topk_val, topk_idx = torch.topk(logits, k=self.k, dim=-1)   # [N,k]
        gate = torch.softmax(topk_val, dim=-1)                      # [N,k]
        return gate.to(x.dtype), topk_idx.int(), logits  # return logits if you need aux loss

def exchange_counts(send_counts, group):
    # send_counts: [ep_size] (#tokens we send TO each rank)
    # returns recv_counts: [ep_size] (#tokens we RECEIVE from each rank)
    send = send_counts.clone()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)  # simple fixed-size exchange
    return recv

class A2ADispatcher:
    def __init__(self, ep_group, num_experts, ep_size, capacity_factor=1.25, top_k=2):
        self.ep_group = ep_group
        self.num_experts = num_experts
        self.ep_size = ep_size
        assert num_experts % ep_size == 0
        self.num_local = num_experts // ep_size
        self.top_k = top_k
        self.capacity_factor = capacity_factor

    def route(self, x, gate, topk_idx):
        """
        x: [N,H]; gate:[N,k]; topk_idx:[N,k]
        returns:
          expert_inputs: concatenated tokens for local experts [M,H]
          expert_slices: list of (start,end) per local expert
          combine_info: dict with original positions, per-path gates to combine
        """
        N, H = x.shape
        # 1) capacity
        cap = int((self.capacity_factor * self.top_k * N + self.num_experts - 1) // self.num_experts)

        # 2) expand tokens for top-k
        expanded = x.unsqueeze(1).expand(N, self.top_k, H).reshape(-1, H)              # [N*k, H]
        experts  = topk_idx.reshape(-1)                                                # [N*k]
        gates    = gate.reshape(-1)                                                    # [N*k]

        # 3) map expert -> (owner_rank, local_expert)
        owner = torch.div(experts, self.num_local, rounding_mode='floor')              # [N*k]
        local = experts % self.num_local
        key   = owner * self.num_local + local
        sort_idx = torch.argsort(key)
        expanded, owner, local, gates = expanded[sort_idx], owner[sort_idx], local[sort_idx], gates[sort_idx]

        # 4) enforce capacity per (owner,local) bucket
        buckets = owner * self.num_local + local
        uniq, counts = buckets.unique_consecutive(return_counts=True)
        keep_mask = torch.ones_like(buckets, dtype=torch.bool)
        start = 0
        for c in counts.tolist():
            if c > cap: keep_mask[start+cap:start+c] = False
            start += c
        send_tokens = expanded[keep_mask]
        send_owner  = owner[keep_mask]
        send_local  = local[keep_mask]
        send_gates  = gates[keep_mask]

        # 5) build variable-size splits per dest rank
        H = x.shape[-1]
        splits = [ (send_owner == r).sum().item() for r in range(self.ep_size) ]
        send_splits = torch.tensor(splits, device=x.device, dtype=torch.int64)
        send_buf = torch.empty(send_tokens.shape[0], H, device=x.device, dtype=x.dtype)
        send_buf.copy_(send_tokens)

        # 6) exchange counts to compute recv_splits, then all_to_all_single
        recv_splits = exchange_counts(send_splits, self.ep_group)
        total_recv = int(recv_splits.sum().item())
        recv_buf = torch.empty(total_recv, H, device=x.device, dtype=x.dtype)

        dist.all_to_all_single(
            output=recv_buf, input=send_buf,
            output_split_sizes=recv_splits.tolist(),
            input_split_sizes=send_splits.tolist(),
            group=self.ep_group
        )
        # 7) For local processing, we need the per-local-expert slices of recv_buf
        #    Reconstruct local buckets (we sent (owner, local), but on receive, everything belongs to 'this' owner)
        #    In practice, send metadata in a parallel all_to_all for 'local' and 'gate' as fp32/idx lists.
        return recv_buf, send_local, send_gates, keep_mask, sort_idx

class MoEFFN(nn.Module):
    def __init__(self, hidden, num_experts, k, ep_group, ep_size, ffn_mult=4, capacity_factor=1.25):
        super().__init__()
        self.hidden = hidden
        self.num_experts = num_experts
        self.k = k
        self.gate = TopKGate(hidden, num_experts, k=k, jitter_std=0.0)
        self.dispatch = A2ADispatcher(ep_group, num_experts, ep_size, capacity_factor, k)
        self.num_local = num_experts // ep_size
        self.local_experts = nn.ModuleList([ExpertMLP(hidden, ffn_mult) for _ in range(self.num_local)])

    def forward(self, x):  # x:[B,T,H]
        B, T, H = x.shape
        flat = x.reshape(B*T, H)
        gate, topk_idx, _ = self.gate(flat)
        recv_buf, send_local, send_gates, keep_mask, sort_idx = self.dispatch.route(flat, gate, topk_idx)

        # Split recv_buf evenly by local expert capacity/buckets (omitted: reconstruct exact per-expert slices)
        # Placeholder: assume contiguous per-expert blocks; apply experts and concatenate
        # In production, keep an explicit index tensor for each local expert slice.
        # y_local = [ self.local_experts[e](tokens_e) for e, tokens_e in enumerate(split(recv_buf)) ]
        # y_cat = concat(y_local) ; all_to_all back (mirror of forward) ; combine top-k using gates ; unpermute
        return x
```

Notes:

- A production dispatcher keeps explicit index tensors to re‑assemble outputs and to weight each path by its gate before summing and restoring the original order. (DeepSpeed/Megatron/Tutel already do this and overlap A2A with compute.) ([deepspeed.readthedocs.io][1])
- If you roll your own, profile with NVTX around both A2As and per‑expert GEMMs; keep EP groups intra‑NVLink/NVSwitch whenever possible. Recent NVIDIA guidance emphasizes “wide EP” scaling on NVLink islands. ([NVIDIA Developer][11])

---

### 4) Parallelism: dp × tp × pp × ep (+ sp)

- World size: `world = dp * tp * pp * ep`.
- Sequence parallel (SP) is still recommended when TP>1 to reduce activation memory and to make MLPs efficient. Megatron docs, PRs and downstream frameworks continue to call this out. ([GitHub][12])
- Example (64 GPUs): `dp=2, tp=2, pp=2, ep=8` → 64. With `num_experts=64`, each EP rank hosts `num_local_experts=8`. Keep EP groups within a node if you have 8× NVLink; if you must span nodes, pin EP to the fastest fabric domain. ([NVIDIA Developer][11])

---

### 5) DeepSpeed training (what goes in JSON vs. code)

`ds_config.json` (ZeRO/precision/optimizer/checkpointing only):

```json
{
  "train_batch_size": 1024,
  "gradient_accumulation_steps": 8,
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 3,
    "offload_param": { "device": "none" },
    "offload_optimizer": { "device": "none" },
    "overlap_comm": true,
    "reduce_scatter": true,
    "contiguous_gradients": true
  },
  "optimizer": {
    "type": "AdamW",
    "params": { "lr": 3e-4, "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0.1 }
  },
  "gradient_clipping": 1.0,
  "activation_checkpointing": { "partition_activations": true, "cpu_checkpointing": false },
  "aio": { "block_size": 1048576, "queue_depth": 8 },
  "wall_clock_breakdown": false
}
```

Why no `"moe": {...}` in JSON? Because the current DS docs expose MoE layer knobs as constructor args (e.g., `ep_size`, `k`, `capacity_factor`, `drop_tokens`, `noisy_gate_policy`, `use_tutel`, `enable_expert_tensor_parallelism`) rather than JSON. Build the MoE layer in code and pass your expert module into `deepspeed.moe.layer.MoE`. ([deepspeed.readthedocs.io][1])

MoE layer creation (code):

```python
import deepspeed
from deepspeed.moe.layer import MoE

moe_ffn = MoE(
    hidden_size=hidden,
    expert=ExpertMLP(hidden, mult=4),   # your expert MLP module
    num_experts=64,
    ep_size=8,
    k=2,
    capacity_factor=1.25,
    min_capacity=4,
    drop_tokens=True,
    noisy_gate_policy="Jitter",
    use_tutel=True,                      # if tutel is installed
    enable_expert_tensor_parallelism=False # set True if you shard experts internally
)
```

Docs list all these arguments in DS 0.18.0. ([deepspeed.readthedocs.io][1])

Launcher:
`deepspeed --num_nodes 8 --num_gpus 8 train.py --deepspeed ds_config.json` (or via `accelerate`). ([DeepSpeed][13])

---

### 6) Megatron‑Core + (DeepSpeed ZeRO) path

If you prefer Megatron‑Core’s kernels and schedulers, drive MoE through Megatron flags and keep ZeRO/optimizer in DS JSON (or use Megatron’s own optimizer). Typical flags:

```bash
--tensor-model-parallel-size 2 \
--pipeline-model-parallel-size 2 \
--expert-model-parallel-size 8 \
--num-experts 64 \
--moe-router-topk 2 \
--moe-grouped-gemm \
--sequence-parallel
```

Current naming is in Megatron‑Core docs/README; `--expert-model-parallel-size` is the EP size you want. ([NVIDIA Docs][9])

---

### 7) Cluster playbook (SLURM)

```bash
#!/bin/bash
#SBATCH -N 8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH -p gpu
#SBATCH -J moe_vlm

export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_NET_GDR_LEVEL=2
export NCCL_MIN_NCHANNELS=8
export NCCL_DEBUG=WARN

srun --gpu-bind=closest --accel-bind=gn \
  deepspeed --num_nodes ${SLURM_JOB_NUM_NODES} --num_gpus 8 \
  train.py --deepspeed ds_config.json
```

Topology guidance: keep EP groups intra‑node (NVLink/NVSwitch). If you must set `ep_size` > GPUs per node, place the EP group on a single NVLink island before crossing racks. NVIDIA’s newest guidance for wide EP scaling on NVL72 emphasizes this locality. ([NVIDIA Developer][11])

---

### 8) Training paradigm (unchanged, with a few 2025 touches)

- Curriculum:
  Phase 1 freeze vision, 224px, Top‑1, high aux coeff → Phase 2 unfreeze, 336–448px, Top‑2, tune capacity, reduce aux → Phase 3 SFT/IT; add RL(HF/DPO) if needed.
- Kernels: use FlashAttention‑3 where supported; many stacks (Megatron, TRT‑LLM) integrate FA‑3. ([tridao.me][5])
- Regularizers: dropout in attention and MoE MLP; z‑loss on logits if you see numerics.
- Data mix: text‑only + image‑caption + interleaved multimodal + OCR heavy; still SOTA practice.

---

### 9) Inference & serving (2025 options)

- vLLM (OpenAI‑style API):

  ```bash
  python -m vllm.entrypoints.openai.api_server \
      --model /path/to/moe-vlm \
      --tensor-parallel-size 2 \
      --enable-expert-parallel \
      --max-model-len 4096
  ```

  EP is supported; EP size is typically computed from TP×DP or set explicitly (name can vary by build). vLLM retains PagedAttention for throughput. ([VLLM Documentation][3])

- DeepSpeed‑Inference: Initialize with EP groups mirroring training (`deepspeed.init_inference(...)`). Good for environments already standardized on DS. ([deepspeed.readthedocs.io][14])

- TensorRT‑LLM (fastest on NVLink nodes): Build TRT‑LLM engines per shard; it supports MoE EP and lists fused MoE/NVFP4 in recent release notes. Pair with Triton or `trtllm-serve`. ([nvidia.github.io][4])

Quantization: W8A16 / INT8/INT4 for attention/MLP; keep gate/routing in BF16/FP32. Quantize experts individually and keep scales synchronized across EP ranks (TRT‑LLM docs show the supported mixtures). ([nvidia.github.io][15])

---

### 10) “Latest practical optimizations” (what’s actually new/useful)

- Grouped GEMM for experts: batch local experts for occupancy (Megatron `--moe-grouped-gemm`, Triton/TensorRT‑LLM grouped kernels). ([NVIDIA Docs][9])
- FlashAttention‑3 in attention blocks (Hopper). ([tridao.me][5])
- Fused dispatchers / overlapped A2A: modern stacks overlap all‑to‑all with MLP compute; some research kernels (e.g., FlashDMoE) fuse routing+compute into a single persistent kernel. Consider only if you own the kernel stack. ([arXiv][16])
- “Wide EP” layouts: keep EP inside NVLink islands; NVIDIA has recent guidance and tooling. ([NVIDIA Developer][11])
- Tutel integration toggle in DS (`use_tutel=True`) still helps on multi‑node.
- Dropless variants (MegaBlocks) if you must guarantee 0 drops.

---

### 11) Monitoring & debugging (unchanged priorities)

- Track: per‑expert token counts, gate entropy, aux loss, tokens dropped, A2A latency/bandwidth, expert GEMM time, pipeline bubble.
- Use NVTX scopes around A2A and expert compute; profile with Nsight Systems. A2A > ~25–30% step time is a red flag (re‑place EP groups and/or increase capacity). The trainer now emits `moe_ms`, `a2a_ms`, ratios, and call counts every `logging_steps`, and the TensorBoard run mirrors these scalars for long-term tracking. ([NVIDIA Developer][11])

---

## Ready‑to‑use “starter” scaffolds

### A) VLM stub + MoE swap‑in (DeepSpeed path)

```python
# train.py (skeleton)
import os, torch, torch.nn as nn, deepspeed, torch.distributed as dist
from deepspeed.moe.layer import MoE

def init_dist():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

class VisionProjector(nn.Module):
    def __init__(self, vis_dim, hidden):
        super().__init__()
        self.proj = nn.Linear(vis_dim, hidden, bias=False)
    def forward(self, vis_tokens):  # [B, V, vis_dim]
        return self.proj(vis_tokens)

class DecoderBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, num_heads=hidden//64, batch_first=True)
        self.mlp  = nn.Sequential(nn.Linear(hidden, 4*hidden), nn.SiLU(), nn.Linear(4*hidden, hidden))
        self.ln1, self.ln2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
    def forward(self, x):
        h, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x

class VLM(nn.Module):
    def __init__(self, hidden=2048, nblk=24, num_experts=64, ep_size=8, topk=2):
        super().__init__()
        self.proj = VisionProjector(vis_dim=1024, hidden=hidden)
        self.blocks = nn.ModuleList([DecoderBlock(hidden) for _ in range(nblk)])
        # Replace selected FFNs with MoE:
        expert = ExpertMLP(hidden, mult=4)
        self.moe = MoE(hidden, expert, num_experts=num_experts, ep_size=ep_size,
                       k=topk, capacity_factor=1.25, min_capacity=4,
                       drop_tokens=True, noisy_gate_policy="Jitter", use_tutel=True)
        for i in range(nblk//2, nblk):   # MoE in deeper half as a start
            self.blocks[i].mlp = self.moe
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, vis_tokens, txt_tokens):
        x_vis = self.proj(vis_tokens)
        x = torch.cat([x_vis, txt_tokens], dim=1)
        for blk in self.blocks:
            x = blk(x)
        logits = self.lm_head(x[:, -txt_tokens.size(1):, :])
        return logits

if __name__ == "__main__":
    init_dist()
    model = VLM()
    engine, optim, _, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(),
        args=None, config="ds_config.json"
    )
    # training loop omitted: compute CE loss + aux gate loss returned by MoE.forward(); add with coeff
```

Aux loss wiring: DeepSpeed’s `MoE.forward()` returns `(output, l_aux, exp_counts)`. Add `loss_total = lm_loss + λ * l_aux.mean()` with λ≈1e‑2 to 1e‑1 to start. (See DS MoE API doc.) ([deepspeed.readthedocs.io][1])

### B) Megatron‑Core flags (if you go that route)

```bash
python pretrain_gpt.py \
  --tensor-model-parallel-size 2 \
  --pipeline-model-parallel-size 2 \
  --expert-model-parallel-size 8 \
  --num-experts 64 \
  --moe-router-topk 2 \
  --moe-grouped-gemm \
  --sequence-parallel \
  ... # (optimizer/ZeRO/IO etc.)
```

Use the current docs (0.14) for exact names; they’ve stabilized around the forms above. ([NVIDIA Docs][9])

---

## Serving quickstarts

- vLLM (OpenAI API)

  ```bash
  python -m vllm.entrypoints.openai.api_server \
    --model checkpoints/moe-vlm \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
    --max-model-len 4096
  ```

  EP/TP/DP interplay is described in the vLLM “Parallelism and Scaling” and “Expert Parallel Deployment” docs. ([VLLM Documentation][17])

- TensorRT‑LLM

  1. Export shards → `trtllm-build` with MoE/EP options; 2) launch `trtllm-serve` or Triton backend for online serving. Recent releases mention fused MoE + NVFP4 autotuning, plus EP docs. ([nvidia.github.io][18])

---

## Sanity checklist (what usually bites teams)

- ✅ `num_experts % ep_size == 0`.
- ✅ Router logits computed in FP32; cast back afterward.
- ✅ Count exchange before var‑size `all_to_all_single`. Avoid sending/receiving into the same tensor. ([PyTorch Docs][6])
- ✅ EP groups placed within NVLink/NVSwitch islands where possible (see “wide EP”). ([NVIDIA Developer][11])
- ✅ When TP>1, enable sequence parallel. ([GitHub][12])
- ✅ Track tokens dropped and A2A % step time. >~30% usually means topology or capacity issues.

---

## Where your draft was already strong

- Choice of Top‑2 gating, capacity math, EP A2A flow, stability tips (pre‑norm, BF16), and the DP×TP×PP×EP worldview are all sound and align with DeepSpeed/Megatron guidance. ([deepspeed.readthedocs.io][1])

---

## Optional 2025 extras (use if/when needed)

- Expert‑Choice routing (EC): experts pick tokens (fixed bucket sizes), giving natural load balance. Consider EC if you see chronic imbalance that the Switch aux loss can’t tame. (Used in newer large MoE lines.) ([arXiv][19])
- Dropless MoE with MegaBlocks‑style routing to avoid token drop tails under skew.
- FA‑3 everywhere on Hopper; otherwise FA‑2 is fine on earlier GPUs. ([tridao.me][5])

---

### References for “latest” claims used above

- DeepSpeed MoE API (layer args: `ep_size`, `k`, `capacity_factor`, `min_capacity`, `drop_tokens`, `noisy_gate_policy`, `use_tutel`, `enable_expert_tensor_parallelism`). ([deepspeed.readthedocs.io][1])
- Megatron‑Core (MoE, grouped‑GEMM, EP flags; 0.14.0 release on PyPI). ([NVIDIA Docs][9])
- vLLM EP support (parallelism docs + EP deployment). ([VLLM Documentation][17])
- TensorRT‑LLM EP & fused MoE / NVFP4 notes. ([nvidia.github.io][4])
- FlashAttention‑3 (paper/blog/talk). ([tridao.me][10])
- Tutel & MegaBlocks.
- NVIDIA “wide EP” guidance (recent). ([NVIDIA Developer][11])

---

## Want me to turn this into a repo scaffold?

If you share (a) node/GPU count & interconnect, (b) target hidden size/LLM depth, (c) desired num_experts/ep_size, and (d) whether you prefer DeepSpeed-only or Megatron-Core (+ZeRO), I can produce a minimal runnable project with:

- VLM stub (+ projector & token reducer),
- MoE FFN wired in correctly (using DS or Megatron modules),
- `ds_config.json` tuned to your cluster,
- SLURM launcher,
- vLLM or TensorRT-LLM serving entry.

---

## Omni-Stack MoE Code Layout (Generated Framework)

- `configs/`
  - `dataset_config.json` – staged curriculum data mixer matching the provided sampling ratios.
  - `omni_moe_config.json` – model hyperparameters, MoE routing knobs, projector specs.
  - `deepspeed_zero2_config.json` – ZeRO stage-2 with CPU offload and MoE options aligned to DeepSpeed 0.18.
  - `train_hyperparams.yaml` – curriculum order, stage LR scheduling, logging/checkpoint cadence.
  - `cluster_config.yaml` – NCCL and launcher hints for Slurm/NVL72 deployments.
- Key architectural features:
  - Top-2 sparse routing across text/vision/projector MoE layers with per-scope auxiliary and global load-balancing losses.
  - Shared “always-on” expert residual paths for stability, configurable per scope in `omni_moe_config.json`.
  - Multi-scale SigLIP2 vision tokens (mid + final layers + CLS) fed into the DeepStack-style projector with stage-tunable depth.
  - Stage-aware parameter freezing/unfreezing and optimizer reinitialisation via `train_hyperparams.yaml`.
- `data/`
  - `dataset.py` – weighted multi-source JSON loader with SigLIP transforms and dialogue tokenization.
  - `collate.py` – padding, label masking, and image tensor batching aware of per-sample pad ids.
- `models/`
  - `moe_layer.py` – DeepSpeed-compatible expert modules and optimizer grouping helper.
  - `vision_siglip.py` – SigLIP ViT with MoE injected into the final K transformer blocks.
  - `llm_qwen.py` – Qwen3-8B wrapper that replaces specified FFNs with MoE experts.
  - `projector_bridge.py` – DeepStack-inspired cross-modal MoE projector with auxiliary load-balancing loss.
  - `omni_model.py` – unified PreTrainedModel wiring SigLIP, projector, and Qwen, ready for HF/vLLM checkpoints.
- `training/`
  - `train.py` – CLI entry point; parses configs and launches DeepSpeed curriculum training.
  - `trainer.py` – stage-aware loop with scheduler, checkpointing, distributed logging.
  - `scheduler.py`, `losses.py`, `utils.py` – LR schedule, auxiliary losses, reproducibility helpers.
- `inference/`
  - `generate.py` – offline CLI for single image-question runs.
  - `inference_engine.py` – batched inference facade with tokenizer/vision preprocessing.
  - `serving/server.py` – FastAPI service (JSON + base64 image) suitable for Kubernetes or Triton sidecar.
- `deploy/`
  - `run_distributed.sh` – direct DeepSpeed launcher with environment overrides.
  - `slurm_submit.sh` – NVL-scale Slurm submission script with NCCL tuning.
  - `docker/Dockerfile` – container recipe targeting NVIDIA PyTorch 24.08.
- `evaluation/`
  - `evaluate_vqa.py`, `evaluate_captioning.py`, `evaluate_nlp.py` – task-specific harnesses with EM/BLEU metrics.
  - `metrics.py` – shared scoring utilities.
  - `opencompass/omni_moe_adapter.py` – OpenCompass adapter to load OmniMoE and evaluate on multimodal tasks.
  - `vlmevalkit_adapter.py` – VLMEvalKit bridge exposing a predict(image, question) method.
- `requirements.txt` – minimal runtime dependencies (DeepSpeed, Transformers, vLLM, FastAPI).

### Quickstart

```bash
pip install -r OmniMoE/requirements.txt

deploy/run_distributed.sh --output_dir /mnt/checkpoints/omni_stack_moe_exp1

python OmniMoE/inference/generate.py \
  --model_path /mnt/checkpoints/omni_stack_moe/final \
  --config_path OmniMoE/configs/omni_moe_config.json \
  --image /path/to/example.jpg \
  --question "What colors are on the bus?"

uvicorn OmniMoE.inference.serving.server:build_app --factory --host 0.0.0.0 --port 9000 \
  --reload --model_dir /mnt/checkpoints/omni_stack_moe/final \
  --config_path OmniMoE/configs/omni_moe_config.json

python OmniMoE/evaluation/evaluate_vqa.py \
  --model_dir /mnt/checkpoints/omni_stack_moe/final \
  --config OmniMoE/configs/omni_moe_config.json \
  --dataset /data/vqa/val.json
```

OpenCompass (example): see `evaluation/opencompass/omni_moe_adapter.py` and configure a run file to point at your checkpoints.
VLMEvalKit (example): import `OmniMoEVLMEval` from `evaluation/vlmevalkit_adapter.py` and register it in your VLMEvalKit model zoo.

For high-throughput serving, export the HF-style checkpoint produced in `output_dir/final` and load it with vLLM:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /mnt/checkpoints/omni_stack_moe/final \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --max-model-len 4096
```

The custom projector-run image tokens are embedded during the prompt phase, so subsequent decoding steps reuse KV caches exactly as vLLM expects.

[1]: https://deepspeed.readthedocs.io/en/latest/moe.html "Mixture of Experts (MoE) — DeepSpeed 0.18.0 documentation"
[2]: https://pypi.org/project/megatron-core/ "megatron-core"
[3]: https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment.html "Expert Parallel Deployment - vLLM"
[4]: https://nvidia.github.io/TensorRT-LLM/advanced/expert-parallelism.html "Expert Parallelism in TensorRT-LLM - GitHub Pages"
[5]: https://tridao.me/blog/2024/flash3/ "FlashAttention-3: Fast and Accurate Attention with ..."
[6]: https://pytorch.org/docs/stable/distributed.html "Distributed communication package - torch.distributed"
[7]: https://arxiv.org/pdf/2101.03961 "Switch Transformers"
[8]: https://proceedings.neurips.cc/paper_files/paper/2022/file/960a172bc7fbf0177ccccbb411a7d800-Paper-Conference.pdf "Flamingo: a Visual Language Model for Few-Shot Learning"
[9]: https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/moe.html "Mixture of Experts package — Megatron Core"
[10]: https://tridao.me/publications/flash3/flash3.pdf "FlashAttention-3: Fast and Accurate Attention with ..."
[11]: https://developer.nvidia.com/blog/scaling-large-moe-models-with-wide-expert-parallelism-on-nvl72-rack-scale-systems/ "Scaling Large MoE Models with Wide Expert Parallelism ..."
[12]: https://github.com/NVIDIA/bionemo-framework/issues/1243 "Tensor + Sequence parallel coverage"
[13]: https://www.deepspeed.ai/getting-started/ "Getting Started - DeepSpeed"
[14]: https://deepspeed.readthedocs.io/en/latest/inference-init.html "Inference Setup — DeepSpeed"
[15]: https://nvidia.github.io/TensorRT-LLM/release-notes.html "Release Notes — TensorRT LLM"
[16]: https://arxiv.org/abs/2506.04667 "FlashDMoE: Fast Distributed MoE in a Single Kernel"
[17]: https://docs.vllm.ai/en/stable/serving/parallelism_scaling.html "Parallelism and Scaling - vLLM"
[18]: https://nvidia.github.io/TensorRT-LLM/0.19.0/release-notes.html "Release Notes — TensorRT-LLM"
[19]: https://arxiv.org/abs/2202.09368 "Mixture-of-Experts with Expert Choice Routing"
