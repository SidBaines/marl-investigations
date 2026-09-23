"""One frozen text model hosts isolated optimizers without retaining T-by-V logits.

LoRA targets every text-model ``nn.Linear``, including ``lm_head``: SDK 0.30.1
``tinker/types/lora_config.py`` defaults train_attn/train_mlp/train_unembed
to True. PEFT's ``all-linear`` shorthand excludes the head, so targets are
enumerated explicitly. Nonlinear hybrid attention components (convolutions,
norms) stay frozen. Fused expert parameters are unsupported rather than silently
omitting their MLP adapters. Dropout is zero and alpha defaults to rank;
LearnerSpec has no alpha field, so an explicit constructor kwarg supplies it.

Qwen3.5's multimodal architecture is loaded through Qwen3_5ForCausalLM.
Transformers 5.5.4's qwen3_5_text conversion mapping strips
``model.language_model`` prefixes and ignores visual weights. Training and
export therefore use text-only PEFT paths; M4-2 must serve the matching text
model. Sampling and init_from orchestration belong to that backend.
"""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar
from uuid import uuid4

from marli.model import ModelSpec
from marli.runlog import git_info
from marli.train.backends.base import SamplerSnapshot, StepResult
from marli.train.types import LearnerSpec, TrainDatum

if TYPE_CHECKING:
    import torch
    from peft import PeftModel
    from torch import Tensor, nn
    from transformers import PreTrainedModel

    from marli.policy.base import TokenPolicy

_T = TypeVar("_T")


def chunked_logprobs(
    hidden_states: Tensor, lm_head: nn.Module, targets: Tensor, *, chunk_size: int = 1024
) -> Tensor:
    """Gather next-token logprobs with O(chunk_size * vocabulary) logit storage.

    Non-reentrant checkpointing recomputes each projection during backward.
    Merely chunking forward would retain *all* log_softmax outputs for backward
    and still use O(sequence * vocabulary) fp32 memory. No causal context is
    split: chunking happens only after the decoder has produced hidden states.
    """
    import torch
    from torch.utils.checkpoint import checkpoint

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if hidden_states.shape[:-1] != targets.shape or targets.numel() == 0:
        raise ValueError("hidden states and non-empty targets must align")
    hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    target = targets.reshape(-1)

    def project(h: Tensor, t: Tensor) -> Tensor:
        logits = lm_head(h).float()
        return logits.log_softmax(-1).gather(-1, t.unsqueeze(-1)).squeeze(-1)

    chunks = []
    for start in range(0, target.numel(), chunk_size):
        h, t = hidden[start : start + chunk_size], target[start : start + chunk_size]
        chunks.append(
            checkpoint(project, h, t, use_reentrant=False)
            if torch.is_grad_enabled()
            else project(h, t)
        )
    return torch.cat(chunks).reshape(targets.shape)


def _device(device: str) -> torch.device:
    import torch

    resolved = torch.device(device)
    if str(resolved) != "cpu" and (resolved.type != "cuda" or not torch.cuda.is_available()):
        raise ValueError("local training requires CUDA, or explicit device='cpu' for tests")
    return resolved


class LocalLearnerPool:
    def __init__(
        self,
        model: ModelSpec,
        *,
        device: str,
        dtype: str = "bfloat16",
        gradient_checkpointing: bool = True,
        attn_implementation: str = "sdpa",
    ) -> None:
        import torch
        import transformers

        resolved = _device(device)
        if model.local == "no":
            raise ValueError(f"model {model.name!r} does not support local training")
        if dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError(f"unsupported local dtype {dtype!r}")
        architecture = model.architecture
        if architecture == "Qwen3_5ForConditionalGeneration":
            architecture = "Qwen3_5ForCausalLM"
        loader = (
            getattr(transformers, architecture)
            if architecture
            else (transformers.AutoModelForCausalLM)
        )
        base = loader.from_pretrained(
            model.hf_id, dtype=getattr(torch, dtype), attn_implementation=attn_implementation
        )
        self._initialize(base, model, resolved, gradient_checkpointing, tokenizer_sha=None)

    @classmethod
    def from_model(
        cls,
        model: PreTrainedModel,
        *,
        tokenizer_sha: str,
        device: str = "cpu",
        gradient_checkpointing: bool = True,
        model_spec: ModelSpec | None = None,
    ) -> LocalLearnerPool:
        """Use an already constructed text model without loading weights/tokenizers."""
        if model_spec is None:
            family = {"qwen3": "qwen3", "qwen3_5_text": "qwen3_5", "gpt_oss": "gpt_oss"}[
                model.config.model_type
            ]
            model_spec = ModelSpec(
                name="local-test",
                hf_id=model.config._name_or_path or "random-init",
                family=family,
                renderer="fake",
                architecture=type(model).__name__,
                max_ctx=model.config.max_position_embeddings,
                default_max_tokens=1,
                thinking=False,
                tool_format={"qwen3": "qwen3_json", "qwen3_5": "qwen3_5_xml", "gpt_oss": "harmony"}[
                    family
                ],
                local="yes",
            )
        pool = cls.__new__(cls)
        pool._initialize(model, model_spec, _device(device), gradient_checkpointing, tokenizer_sha)
        return pool

    def _initialize(
        self,
        base: PreTrainedModel,
        spec: ModelSpec,
        device: torch.device,
        gradient_checkpointing: bool,
        tokenizer_sha: str | None,
    ) -> None:
        import torch

        if any(
            parameter.ndim > 2 for name, parameter in base.named_parameters() if "experts" in name
        ):
            raise ValueError("fused expert MLP adapters are unsupported by this local learner")
        if base.get_decoder() is base:
            raise ValueError("local learner requires a separable text decoder and lm_head")
        self.model = spec
        self.device = device
        self.tokenizer_sha = tokenizer_sha
        self.lock = asyncio.Lock()
        self.base_model = base.to(device)
        self.base_model.requires_grad_(False)
        self.base_model.config.use_cache = False
        if gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.target_modules = tuple(
            name for name, module in base.named_modules() if isinstance(module, torch.nn.Linear)
        )
        if not self.target_modules:
            raise ValueError("local learner requires linear LoRA targets")
        self.peft_model: PeftModel | None = None

    async def _run_locked(self, operation: Callable[[], _T]) -> _T:
        async with self.lock:
            task = asyncio.create_task(asyncio.to_thread(operation))
            cancellation = None
            while True:
                try:
                    result = await asyncio.shield(task)
                    break
                except asyncio.CancelledError as exc:
                    # to_thread cannot stop torch, even on repeated cancellation.
                    # Keep the adapter locked until its worker actually finishes.
                    if task.cancelled():
                        raise
                    cancellation = exc
            if cancellation is not None:
                raise cancellation
            return result


def _microbatches(datums: Sequence[TrainDatum], budget: int) -> Iterator[list[TrainDatum]]:
    """Bound padded input tokens; a single over-budget context stays intact."""
    batch: list[TrainDatum] = []
    longest = 0
    for datum in datums:
        length = len(datum.tokens) - 1
        if batch and max(longest, length) * (len(batch) + 1) > budget:
            yield batch
            batch, longest = [], 0
        batch.append(datum)
        longest = max(longest, length)
    if batch:
        yield batch


class LocalLearner:
    def __init__(
        self,
        name: str,
        spec: LearnerSpec,
        pool: LocalLearnerPool,
        *,
        alpha: int | None = None,
        max_tokens_per_microbatch: int = 16384,
        logprob_chunk_size: int = 1024,
    ) -> None:
        if spec.backend != "local":
            raise ValueError("LocalLearner requires spec.backend == 'local'")
        if not name or "." in name or "/" in name or "\\" in name:
            raise ValueError("adapter name must be nonempty without dots or path separators")
        if max_tokens_per_microbatch <= 0 or logprob_chunk_size <= 0:
            raise ValueError("microbatch token budget and logprob chunk size must be positive")
        if alpha is not None and alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if pool.lock.locked():
            raise RuntimeError("create learners before starting concurrent pool operations")

        import torch
        from peft import LoraConfig, get_peft_model

        self.name, self.spec, self.pool = name, spec, pool
        self.model = pool.model
        self.version = 0
        self.step = 0
        self.alpha = spec.rank if alpha is None else alpha
        self.max_tokens_per_microbatch = max_tokens_per_microbatch
        self.logprob_chunk_size = logprob_chunk_size
        config = LoraConfig(
            r=spec.rank,
            lora_alpha=self.alpha,
            lora_dropout=0.0,
            target_modules=list(pool.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        if pool.peft_model is None:
            pool.peft_model = get_peft_model(pool.base_model, config, adapter_name=name)
        else:
            pool.peft_model.add_adapter(name, config)
        config.base_model_name_or_path = self.model.hf_id
        pool.peft_model.set_adapter(name)
        self.parameters = tuple(p for p in pool.peft_model.parameters() if p.requires_grad)
        self.optimizer = torch.optim.AdamW(
            self.parameters,
            lr=spec.learning_rate,
            betas=(spec.beta1, spec.beta2),
            eps=spec.eps,
            weight_decay=spec.weight_decay,
        )

    async def train_step(
        self, datums: Sequence[TrainDatum], *, learning_rate: float | None = None
    ) -> StepResult:
        if not datums:
            raise ValueError("train_step requires non-empty datums")
        datums = tuple(datums)
        for datum in datums:
            if datum.learner != self.name:
                raise ValueError(f"datum for {datum.learner!r} routed to learner {self.name!r}")
            if len(datum.tokens) > self.model.max_ctx:
                raise ValueError(f"datum exceeds model max ctx {self.model.max_ctx}")
        if {datum.policy_version for datum in datums} != {self.version}:
            raise ValueError(f"datums must use current policy version {self.version}")
        if learning_rate is not None and learning_rate < 0:
            raise ValueError("learning_rate must be non-negative")
        return await self.pool._run_locked(lambda: self._train_step(datums, learning_rate))

    def _train_step(self, datums: Sequence[TrainDatum], learning_rate: float | None) -> StepResult:
        import torch

        from marli.train.backends.local import losses

        model = self.pool.peft_model
        assert model is not None
        model.set_adapter(self.name)
        model.train()
        for group in self.optimizer.param_groups:
            group["lr"] = self.spec.learning_rate if learning_rate is None else learning_rate
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        metric_sums: dict[str, float] = {}
        ratio_max = 0.0
        n_actions = sum(d.n_action_tokens for d in datums)
        objective = getattr(losses, self.spec.loss)
        try:
            for batch in _microbatches(datums, self.max_tokens_per_microbatch):
                width = max(len(d.tokens) - 1 for d in batch)
                shape = (len(batch), width)
                inputs = torch.zeros(shape, dtype=torch.long, device=self.pool.device)
                targets = torch.zeros_like(inputs)
                attention = torch.zeros_like(inputs)
                sample, advantages, mask = (
                    torch.zeros(shape, dtype=torch.float32, device=self.pool.device)
                    for _ in range(3)
                )
                for row, datum in enumerate(batch):
                    length = len(datum.tokens) - 1
                    tokens = torch.tensor(datum.tokens, device=self.pool.device)
                    inputs[row, :length], targets[row, :length] = tokens[:-1], tokens[1:]
                    attention[row, :length] = 1
                    for dest, values in (
                        (sample, datum.logprobs),
                        (advantages, datum.advantages),
                        (mask, datum.mask),
                    ):
                        dest[row, :length] = torch.tensor(values, device=self.pool.device)
                hidden = self.pool.base_model.get_decoder()(
                    input_ids=inputs, attention_mask=attention, use_cache=False, return_dict=True
                ).last_hidden_state
                lp = chunked_logprobs(
                    hidden,
                    self.pool.base_model.get_output_embeddings(),
                    targets,
                    chunk_size=self.logprob_chunk_size,
                )
                loss, metrics = objective(
                    lp.flatten(), sample.flatten(), advantages.flatten(), mask.flatten()
                )
                loss.backward()
                loss_sum += loss.detach().item()
                count = sum(d.n_action_tokens for d in batch)
                ratio_max = max(ratio_max, metrics.pop("ratio_max").item())
                for key, value in metrics.items():
                    metric_sums[key] = metric_sums.get(key, 0.0) + value.item() * count
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters,
                self.spec.grad_clip if self.spec.grad_clip > 0 else float("inf"),
                error_if_nonfinite=True,
            ).item()
            self.optimizer.step()
            self.step += 1
        finally:
            self.optimizer.zero_grad(set_to_none=True)
        metrics_out = {key: value / max(n_actions, 1) for key, value in metric_sums.items()}
        metrics_out.update({"loss:sum": loss_sum, "ratio_max": ratio_max})
        return StepResult(
            learner=self.name,
            n_datums=len(datums),
            n_tokens=sum(len(d.tokens) - 1 for d in datums),
            n_action_tokens=n_actions,
            loss=loss_sum,
            grad_norm=grad_norm,
            kl_sample_train=metrics_out["kl_sample_train"],
            metrics=metrics_out,
        )

    async def save_adapter(self, directory: str | Path) -> str:
        """Export a standalone PEFT directory, including a LoRA unembedding head."""
        return await self.pool._run_locked(lambda: self._save_adapter(Path(directory)))

    def _save_adapter(self, directory: Path) -> str:
        from peft import get_peft_model_state_dict
        from safetensors.torch import save_file

        model = self.pool.peft_model
        assert model is not None
        directory.mkdir(parents=True, exist_ok=True)
        # Explicit False avoids saving the frozen vocabulary matrix or fetching
        # base config from the Hub, both triggered by PEFT's embedding auto mode.
        weights = get_peft_model_state_dict(
            model, adapter_name=self.name, save_embedding_layers=False
        )
        save_file(
            {key: value.detach().cpu().contiguous().clone() for key, value in weights.items()},
            directory / "adapter_model.safetensors",
            metadata={"format": "pt"},
        )
        config = copy.deepcopy(model.peft_config[self.name])
        config.base_model_name_or_path = self.model.hf_id
        config.inference_mode = True
        config.save_pretrained(directory)
        return str(directory.resolve())

    async def save_state(self, name: str) -> str:
        """Save to a unique step-versioned child of ``name``; publish manifest last."""
        return await self.pool._run_locked(lambda: self._save_state(Path(name)))

    def _save_state(self, directory: Path) -> str:
        import torch

        directory = directory / f"step-{self.step:08d}-{uuid4().hex[:12]}"
        directory.mkdir(parents=True, exist_ok=False)
        self._save_adapter(directory / "adapter")
        torch.save(
            {"optimizer": self.optimizer.state_dict(), "step": self.step},
            directory / "optimizer.pt",
        )
        manifest = {
            "schema_version": 1,
            "base_model": self.model.hf_id,
            "rank": self.spec.rank,
            "alpha": self.alpha,
            "target_modules": list(self.pool.target_modules),
            "tokenizer_sha": self.pool.tokenizer_sha,
            "step": self.step,
            "adapter": "adapter",
            "optimizer": "optimizer.pt",
            "git_commit": git_info(Path(__file__).resolve().parents[5]).commit,
        }
        temporary = directory / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(directory / "manifest.json")
        return str(directory.resolve())

    async def load_state(self, path: str, *, with_optimizer: bool = True) -> None:
        """Restore this adapter; weights-only loads reset optimizer and step to zero."""
        await self.pool._run_locked(lambda: self._load_state(Path(path), with_optimizer))

    def _load_state(self, directory: Path, with_optimizer: bool) -> None:
        import torch
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        expected = {
            "schema_version": 1,
            "base_model": self.model.hf_id,
            "rank": self.spec.rank,
            "alpha": self.alpha,
            "target_modules": list(self.pool.target_modules),
            "tokenizer_sha": self.pool.tokenizer_sha,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"local checkpoint {key} mismatch")
        weights = load_file(directory / manifest["adapter"] / "adapter_model.safetensors")
        state = (
            torch.load(
                directory / manifest["optimizer"], map_location=self.pool.device, weights_only=True
            )
            if with_optimizer
            else None
        )
        model = self.pool.peft_model
        assert model is not None
        model.set_adapter(self.name)
        set_peft_model_state_dict(model, weights, adapter_name=self.name)
        if state is not None:
            self.optimizer.load_state_dict(state["optimizer"])
            self.step = state["step"]
        else:
            self.optimizer.state.clear()
            self.step = 0
        self.optimizer.zero_grad(set_to_none=True)

    def policy(self, *, policy_id: str | None = None) -> TokenPolicy:
        raise NotImplementedError("local sampling is implemented by the M4-2 backend")

    async def sync_sampler(self, name: str) -> SamplerSnapshot:
        raise NotImplementedError("local sampler synchronization belongs to the M4-2 backend")

    async def close(self) -> None:
        # Learners share the base model; its lifetime belongs to the backend/pool.
        pass
