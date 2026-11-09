"""qwen-say and qwen-animate on vLLM, with persona pushes.

A push on neuron n of layer L at dose d adds a * W_down[:, n] to that layer's MLP output,
a = sign * d * median(|h_L|) / |W_down[:, n]|, for each request on its own.

vLLM has no hook for that, and its speed comes from compiling the model and replaying CUDA
graphs, so a push must be a fixed tensor operation inside the graph. Every decoder layer that
holds a catalog neuron gets two buffers when it is built: `steer_cols`, the catalog's
W_down columns at that layer (K x hidden), and `steer_coef`, one coefficient per token and
column (max tokens x K). The layer adds coef[:T] @ cols to its MLP output. Before each forward,
outside the graph, `_fill` writes every token's coefficients from vLLM's batch: which request
the token belongs to, where it is, what that request has written so far.

When a push is on:
  saying     on every token (qwen-say);
  writing    on the prompt, and while qwen-animate chooses its attitude: the plan's mood label
             ("sad at 0.3", up to its colon) and the style line (rigctl/compact.py: the third
             line). The verb list and the beats are written unpushed, so the push cannot
             rewrite the task. The gate reads the text including the token being read, so it covers the
             forward that predicts the span's first token.

The model is the animation adapter merged into Qwen3.5-4B (merge_lora.py), served with FP8
weights (Marlin, weight-only on the 3090s: 1.5x faster decoding at the same quality,
infer_bench.py). qwen-say is that model under its `unmerge` adapter, which takes the merge back
out; pushes use the base checkpoint's W_down columns.

A read (score) can also trace the catalog neurons: every catalog layer's MLP writes its catalog
neurons' activations (the input to down_proj) into `steer_act` (max tokens x K), through a custom
op so the compiled graph keeps the write, and after a forward the rows of the traced requests are
copied out.

Prefix caching is off (a prompt read under a push is not the prompt read without it), and so
is async scheduling (each step must see the token it reads). The engine runs in this process
on the V1 model runner, which keeps each step's batch on itself where `_fill` can read it.
"""
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

import itertools  # noqa: E402
import types  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.lora.request import LoRARequest  # noqa: E402
from vllm.model_executor.models import qwen3_5 as Q  # noqa: E402
from vllm.model_executor.models.qwen3_next import Qwen3NextDecoderLayer  # noqa: E402
from vllm.sampling_params import StructuredOutputsParams  # noqa: E402
from vllm.utils.torch_utils import direct_register_custom_op  # noqa: E402

MODEL = "Qwen/Qwen3.5-4B"
CATALOG = {}                                          # layer -> [neuron, ...], set before the engine is built
_ids = itertools.count()

_init = Q.Qwen3_5DecoderLayer.__init__


def _layer_init(self, vllm_config, layer_type, prefix=""):
    _init(self, vllm_config, layer_type, prefix)
    last = prefix.rsplit(".", 1)[-1]
    L = int(last) if last.isdigit() else -1
    if L in CATALOG:
        hidden = vllm_config.model_config.hf_text_config.hidden_size
        T = vllm_config.scheduler_config.max_num_batched_tokens
        dtype = vllm_config.model_config.dtype
        self.steer_layer = L
        self.register_buffer("steer_cols", torch.zeros(len(CATALOG[L]), hidden, dtype=dtype), persistent=False)
        self.register_buffer("steer_coef", torch.zeros(T, len(CATALOG[L]), dtype=dtype), persistent=False)
        mlp = self.mlp
        assert getattr(mlp, "expert_gate", None) is None, "a dense MLP"
        mlp.register_buffer("steer_idx", torch.tensor(CATALOG[L], dtype=torch.long), persistent=False)
        mlp.register_buffer("steer_act", torch.zeros(T, len(CATALOG[L]), dtype=dtype), persistent=False)
        mlp.forward = types.MethodType(_mlp_forward, mlp)


def _record(x: torch.Tensor, idx: torch.Tensor, buf: torch.Tensor) -> None:
    buf[:x.shape[0]].copy_(x.index_select(1, idx))


def _record_fake(x: torch.Tensor, idx: torch.Tensor, buf: torch.Tensor) -> None:
    return None


direct_register_custom_op("steer_record", _record, mutates_args=["buf"], fake_impl=_record_fake)


def _mlp_forward(self, x):
    # Qwen2MoeMLP.forward (a dense one), keeping the catalog neurons' activations
    gate_up, _ = self.gate_up_proj(x)
    out = self.act_fn(gate_up)
    torch.ops.vllm.steer_record(out, self.steer_idx, self.steer_act)
    out, _ = self.down_proj(out)
    return out


def _layer_forward(self, hidden_states, residual, positions, **kw):
    hs, res = Qwen3NextDecoderLayer.forward(self, hidden_states, residual, positions, **kw)
    if hasattr(self, "steer_coef"):
        hs = hs + self.steer_coef[:hs.shape[0]] @ self.steer_cols
    return hs, res


Q.Qwen3_5DecoderLayer.__init__ = _layer_init
Q.Qwen3_5DecoderLayer.forward = _layer_forward


def _gate_writing(text):
    # the plan's first line up to its colon, and the style line: the text read so far ends on the
    # clip line's newline or inside the style line
    return ":" not in text or text.count("\n") == 2


def _base_down(layers):
    """The base checkpoint's W_down at these layers (a merged model's own W_down has the LoRA in it)."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    src = snapshot_download(MODEL, local_files_only=True)
    where = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for L in layers:
        key = f"model.language_model.layers.{L}.mlp.down_proj.weight"
        with safe_open(os.path.join(src, where[key]), "pt") as f: out[L] = f.get_tensor(key)
    return out


def _runner(llm):
    core = llm.llm_engine.engine_core
    core = getattr(core, "engine_core", core)
    w = core.model_executor.driver_worker
    w = getattr(w, "worker", w)
    return w.model_runner


# qwen-say samples: greedy, a character in an unchanged scene says the same thing every step.
# qwen-animate writes greedily (temperature 0), so the program stays exact.
SAY_TEMP = float(os.environ.get("RIG_SAY_TEMP", "0.8"))


class VSteered:
    def __init__(self, adapter, median, pushes, gpu_memory_utilization=0.5, max_model_len=3072, max_num_seqs=16):
        self.pushes, self.median = pushes, median
        CATALOG.clear()
        for L, n, _ in pushes.values():
            CATALOG.setdefault(L, [])
            if n not in CATALOG[L]: CATALOG[L].append(n)
        self.llm = LLM(adapter, tokenizer=MODEL, quantization="fp8", enable_lora=True, max_lora_rank=64,
                       max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
                       max_num_seqs=max_num_seqs, limit_mm_per_prompt={"image": 0, "video": 0},
                       enable_prefix_caching=False, async_scheduling=False)
        self.base_lora = LoRARequest("base", 2, os.path.join(adapter, "unmerge"))
        self.tok = self.llm.get_tokenizer()
        # the CUDA stream the engine was built and warmed up on: the server steps it from another
        # thread, and the FP8 (Marlin) kernels only have workspace on this stream
        self.cuda_stream = torch.cuda.current_stream()
        self.runner = _runner(self.llm)
        self.layers, self.norms = {}, {}
        base = _base_down(sorted(CATALOG))
        for name, mod in self.runner.model.named_modules():
            if hasattr(mod, "steer_cols"):
                L = mod.steer_layer; self.layers[L] = mod
                W = base[L]
                for k, n in enumerate(CATALOG[L]):
                    col = W[:, n].detach().to(mod.steer_cols.device)
                    mod.steer_cols[k].copy_(col.to(mod.steer_cols.dtype))
                    self.norms[L, n] = col.float().norm().item()
        assert set(self.layers) == set(CATALOG), (sorted(self.layers), sorted(CATALOG))
        self.coef = {L: np.zeros(m.steer_coef.shape, dtype=np.float32) for L, m in self.layers.items()}
        self.dirty = set()                                # layers whose buffer holds an earlier step's push
        self.text = {}                                    # request id -> (tokens seen, text)
        # the traced neurons: every push's, in the catalog's order, as (layer, column in steer_act),
        # and what turns an activation into the push's units (see score)
        self.trace_names = list(pushes)
        self.trace_cols = [(L, CATALOG[L].index(n)) for L, n, _ in pushes.values()]
        self.trace_scale = np.array([sign * self.norms[L, n] / median[L] for L, n, sign in pushes.values()], dtype=np.float32)
        self.traces = {}                                  # trace key -> {position: activations}
        fwd = self.runner._model_forward

        def model_forward(*args, **kw):
            plan = self._fill(kw.get("positions", args[1] if len(args) > 1 else None))
            out = fwd(*args, **kw)
            if plan: self._keep(plan)
            return out
        self.runner._model_forward = model_forward

    TOP = 20                                          # vLLM's most logprobs a token (its max_logprobs)

    def amount(self, push, dose):
        L, n, sign = self.pushes[push]
        return sign * dose * self.median[L] / self.norms[L, n]

    def _fill(self, positions):
        """Every token's push coefficients for this forward. Returns the traced requests' rows:
        [(trace key, first row, end row, position of the first)]."""
        T = positions.shape[-1] if positions is not None else 0
        r = self.runner
        batch = getattr(r, "input_batch", None)
        nreq = batch.num_reqs if batch is not None else 0
        touched, plan = set(), []
        if nreq:
            qsl = r.query_start_loc.np[:nreq + 1]
            done = batch.num_computed_tokens_cpu[:nreq]
            for i, rid in enumerate(batch.req_ids[:nreq]):
                st = r.requests.get(rid)
                extra = (st and st.sampling_params.extra_args) or {}
                a, b = int(qsl[i]), int(qsl[i + 1])
                if extra.get("trace"): plan.append((extra["trace"], a, b, int(done[i])))
                push = extra.get("push")
                if not push: continue
                L, n, amount, mode = push
                if mode == "all":
                    g = np.ones(b - a, dtype=np.float32)
                else:                                     # the prompt, and the attitude spans
                    pos = done[i] + np.arange(b - a)
                    g = (pos < st.num_prompt_tokens).astype(np.float32)
                    if self._gate(rid, st): g[:] = 1
                c = self.coef[L]
                if L not in touched: c[:T] = 0; touched.add(L)
                c[a:b, CATALOG[L].index(n)] = g * amount
        for L in touched:
            self.layers[L].steer_coef[:T].copy_(torch.from_numpy(self.coef[L][:T]))
        for L in self.dirty - touched:                    # clear what an earlier step left
            self.layers[L].steer_coef.zero_()
        self.dirty = touched
        return plan

    def _keep(self, plan):
        """The traced requests' catalog activations from the forward just run."""
        end = max(b for _, _, b, _ in plan)
        act = {L: m.mlp.steer_act[:end].float().cpu().numpy() for L, m in self.layers.items()}
        rows = np.stack([act[L][:, k] for L, k in self.trace_cols], axis=1)
        for key, a, b, p0 in plan:
            got = self.traces.setdefault(key, {})
            for j in range(b - a): got[p0 + j] = rows[a + j]

    def _gate(self, rid, st):
        seen, text = self.text.get(rid, (0, ""))
        out = st.output_token_ids
        n = len(out)
        while n > seen and out[n - 1] < 0: n -= 1
        if n > seen:
            text += self.tok.decode(out[seen:n]); self.text[rid] = (n, text)
        return _gate_writing(text)

    def _run(self, texts, params, lora, every=False):
        """Each request's text, or with every=True the list of all its n samples."""
        with torch.cuda.stream(self.cuda_stream):
            self.text = {}
            outs = self.llm.generate(texts, params, lora_request=lora, use_tqdm=False)
            return [[c.text.strip() for c in o.outputs] if every else o.outputs[0].text.strip() for o in outs]

    def _stream(self, texts, params, lora, done=None, cancelled=None):
        with torch.cuda.stream(self.cuda_stream):
            return self._stream_on(texts, params, lora, done, cancelled)

    def _stream_on(self, texts, params, lora, done=None, cancelled=None):
        """As _run, stepping the engine here: done(i, text) is called as each request finishes, so
        a caller can start on it while the rest are written. Stops early when cancelled()."""
        self.text = {}
        eng = self.llm.llm_engine
        ids = [f"w{next(_ids)}" for _ in texts]
        where = {r: i for i, r in enumerate(ids)}
        for rid, t, p in zip(ids, texts, params): eng.add_request(rid, t, p, lora_request=lora)
        out = [""] * len(texts)
        # profile: per request the seconds to its first token and to its end and its tokens; the
        # engine steps taken and the seconds inside them (the rest is this loop and done())
        t0, first, fin, ntok, steps, in_step, in_done = time.time(), {}, {}, {}, 0, 0.0, 0.0
        while eng.has_unfinished_requests():
            if cancelled and cancelled():
                eng.abort_request(ids); break
            ts = time.time(); outs = eng.step(); in_step += time.time() - ts; steps += 1
            for o in outs:
                rid = o.request_id
                if rid not in where: continue
                n = len(o.outputs[0].token_ids)
                if n and rid not in first: first[rid] = round(time.time() - t0, 3)
                ntok[rid] = n
                if o.finished:
                    fin[rid] = round(time.time() - t0, 3)
                    i = where[rid]; out[i] = o.outputs[0].text.strip()
                    td = time.time()
                    if done: done(i, out[i])
                    in_done += time.time() - td
        self.profile = {"wall_s": round(time.time() - t0, 3), "engine_steps": steps, "in_steps_s": round(in_step, 3),
                        "in_done_s": round(in_done, 3),
                        "requests": [{"first_s": first.get(r), "end_s": fin.get(r), "tokens": ntok.get(r)} for r in ids]}
        return out

    def _terms(self, m):
        if not m.get("push"): return None
        L, n, sign = self.pushes[m["push"]]
        return {"layer": L, "neuron": n, "sign": sign, "dose": m["dose"], "amount": round(self.amount(m["push"], m["dose"]), 3)}

    def _chat(self, system, user):
        return self.tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def say(self, texts, members, grammars, n=1, seed=None):
        """{row: record} -> {row: [n lines it continues with, held to grammars[row]]}: the base
        model (the adapter undone), each member's push on every token, no chat template and no
        system prompt. seed: one random seed for every row, so rows that differ only in their push
        differ by it alone."""
        rows = sorted(texts)
        if not rows: return {}
        params = []
        for r in rows:
            m = members[r]
            push = [*self.pushes[m["push"]][:2], self.amount(m["push"], m["dose"]), "all"] if m.get("push") else None
            params.append(SamplingParams(n=n, max_tokens=160, temperature=SAY_TEMP, top_p=0.95, repetition_penalty=1.05, seed=seed,
                                         structured_outputs=StructuredOutputsParams(grammar=grammars[r]),
                                         extra_args={"push": push}))
        return dict(zip(rows, self._run([texts[r] for r in rows], params, self.base_lora, every=True)))

    def score(self, rows):
        """What a member's push did to a move it wrote, and what the model wanted at each of its
        tokens. rows: [(context, continuation, member)]. Each continuation is read after its
        context with no push, and again with the member's push on every token if it has one, and
        the model's own probabilities taken (the raw distribution: no grammar, temperature or
        penalties). Per row: {"toks": [(start, end, logprob with the push, logprob without)] for
        its tokens, start and end being character offsets in the continuation, or None for a member
        with no push; "offs": each token's (start, end); "top": per token, the TOP tokens the model
        found likeliest in its place (and the token itself), as [(text, logprob as served, logprob without the push or
        None if not among the TOP)]}. As served is with the member's push if it has one. "trace": per token, each catalog push's
        neuron (trace_names) as the model fired it reading that token, as served and without the
        push, in the push's own units: the neuron's output over the layer's median norm, signed
        so that a push at dose d adds d (so 1 is the neuron firing as a dose-1 push does)."""
        from vllm.inputs import TokensPrompt
        prompts, params, meta = [], [], []
        for ctx, cont, m in rows:
            c_ids = self.tok(ctx, add_special_tokens=False)["input_ids"]
            enc = self.tok(cont, add_special_tokens=False, return_offsets_mapping=True)
            push = [*self.pushes[m["push"]][:2], self.amount(m["push"], m["dose"]), "all"] if m.get("push") else None
            for p in ((push, None) if push else (None,)):
                prompts.append(TokensPrompt(prompt_token_ids=c_ids + enc["input_ids"]))
                params.append(SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=self.TOP, detokenize=False,
                                             extra_args={"push": p, "trace": f"t{next(_ids)}"}))
            meta.append((bool(push), len(c_ids), enc["input_ids"], enc["offset_mapping"]))
        with torch.cuda.stream(self.cuda_stream):
            self.text, self.traces = {}, {}
            outs = self.llm.generate(prompts, params, lora_request=self.base_lora, use_tqdm=False)
        traced = [self.traces.get(p.extra_args["trace"], {}) for p in params]
        text = {}
        dec = lambda t: text[t] if t in text else text.setdefault(t, self.tok.decode([t]))
        res, k = [], 0
        for pushed, n0, ids, offs in meta:
            on = outs[k].prompt_logprobs; off = outs[k + 1].prompt_logprobs if pushed else on
            t_on = traced[k]; t_off = traced[k + 1] if pushed else t_on; k += 2 if pushed else 1
            trace = {w: [[round(float(v), 2) for v in t[n0 + i] * self.trace_scale] if n0 + i in t else None for i in range(len(ids))]
                     for w, t in (("on", t_on), ("off", t_off))}
            toks = [(o[0], o[1], on[n0 + i][t].logprob, off[n0 + i][t].logprob) for i, (t, o) in enumerate(zip(ids, offs))]
            top = []
            for i in range(len(ids)):
                a, b = on[n0 + i], off[n0 + i]
                best = sorted(a, key=lambda t: -a[t].logprob)  # the TOP, and the token read if it is not one of them
                top.append([(dec(t), a[t].logprob, b[t].logprob if t in b else None) for t in best])
            res.append({"toks": toks if pushed else None, "offs": [tuple(o) for o in offs], "top": top, "trace": trace})
        return res

    def write(self, prompts, members, system, grammar, cancelled=None, done=None):
        """One program per member; returns (raw programs, push terms). done(i, raw) is called as
        each member's program is finished. system is one prompt for all, or one per member."""
        params = []
        for m in members:
            push = [*self.pushes[m["push"]][:2], self.amount(m["push"], m["dose"]), "attitude"] if m.get("push") else None
            params.append(SamplingParams(max_tokens=900, temperature=0,
                                         structured_outputs=StructuredOutputsParams(grammar=grammar),
                                         extra_args={"push": push}))
        systems = [system] * len(prompts) if isinstance(system, str) else system
        texts = [self._chat(sy, p) for sy, p in zip(systems, prompts)]
        return self._stream(texts, params, None, done, cancelled), [self._terms(m) for m in members]
