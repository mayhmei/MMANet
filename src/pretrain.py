# -*- coding: utf-8 -*-
"""
Pretrain script (optimized)
Key optimizations (without changing MLM objective):
1) Dynamic padding per batch via collate_fn (no full-dataset pad -> huge RAM save)
2) Faster mask_tokens (no labels.tolist() / no python loops)
3) each_batch_eval -> eval every N optimizer steps (configurable), while keeping old flags compatible
4) Fix: missing import re
5) Fix: cache dir mismatch (ensure directory exists for cached_features_file)
6) Fix: evaluation should NOT disturb training RNG state (save/restore RNG around eval)
"""

import yaml
import argparse
from pathlib import Path
import os
import re
import torch
import logging
import random
import numpy as np
import pickle
import time
from typing import Dict, List, Tuple
import copy
import glob
from tqdm.auto import tqdm

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW

from transformers import (
    AutoConfig,
    BertTokenizer,
    AutoTokenizer,
    AutoModelWithLMHead,
    AutoModel,
    PreTrainedTokenizer,
    PreTrainedModel,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)


# ========= Fake SummaryWriter =========
class SummaryWriter:
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


# ========= Datasets =========
class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        self.examples = tokenizer.batch_encode_plus(
            lines, add_special_tokens=True, max_length=block_size, truncation=True
        )["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


class TextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        filename_wo_ext = os.path.splitext(filename)[0]

        cached_features_file = os.path.join(
            "../../TrafficData/DataCache/",
            f"{filename_wo_ext}_{args.model_type}_mlm_{block_size}_cached",
        )

        block_size = block_size - (tokenizer.model_max_length - tokenizer.max_len_single_sentence)

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)
            self.examples = []

            with open(file_path, encoding="utf-8") as f:
                text = f.readlines()

            tokenized_text = []
            line_count = 0
            valid_line_count = 0
            for line in text:
                line_count += 1
                tokens = line.strip().split(" ")
                if len(tokens) > block_size:
                    tokens = tokens[:block_size]
                if len(tokens) == 1:
                    continue
                token_ids = tokenizer.convert_tokens_to_ids(tokens)
                if len(token_ids) > 0:
                    valid_line_count += 1
                    tokenized_text.append(token_ids)

            for i in range(len(tokenized_text)):
                self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i]))

            logger.info("Processed %d/%d valid lines", valid_line_count, line_count)
            logger.info("Saving features into cached file %s", cached_features_file)

            os.makedirs(os.path.dirname(cached_features_file), exist_ok=True)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

            if len(self.examples) < 10:
                logger.warning("WARNING: Only %d examples were processed! This is abnormally low.", len(self.examples))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


# ========= Config / Args =========
def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_pretrain_args():
    config_parser = argparse.ArgumentParser(description="Pretrain model for network traffic analysis")
    config_parser.add_argument(
        "--config", type=str, default="../../Config/intra_packet_pretrain.yaml", help="path to config file"
    )
    config_args, _ = config_parser.parse_known_args()

    config = load_config(config_args.config)
    model_config = config["model"]
    data_config = config["data"]
    training_config = config["training"]
    gpu_config = config["gpu"]

    data_config["train_file"] = data_config["train_file"].format(dataset=data_config["dataset"])
    data_config["eval_file"] = data_config["eval_file"].format(dataset=data_config["dataset"])
    data_config["output_dir"] = data_config["output_dir"].format(dataset=data_config["dataset"])

    def _resolve_path(p):
        if not p:
            return None
        if os.path.isabs(p):
            return p
        base = os.path.dirname(__file__)
        cand = os.path.normpath(os.path.join(base, p))
        if os.path.exists(cand):
            return cand
        return p

    def _derive_architecture(dc, mc):
        cfg_path = _resolve_path(dc.get("config_file"))
        try:
            if cfg_path and os.path.exists(cfg_path):
                cfg = AutoConfig.for_model(mc["type"]).from_json_file(cfg_path)
                h = getattr(cfg, "hidden_size", None)
                ah = getattr(cfg, "num_attention_heads", None)
                nl = getattr(cfg, "num_hidden_layers", None)
                if h and ah and nl:
                    return f"{h}d_{ah}h_{nl}l"
        except Exception:
            pass
        return "auto"

    def _unique_name(base_dir, nm):
        if not os.path.isdir(base_dir):
            return nm
        if not os.path.exists(os.path.join(base_dir, nm)):
            return nm
        i = 1
        while True:
            cand = f"{nm}-{i}"
            if not os.path.exists(os.path.join(base_dir, cand)):
                return cand
            i += 1

    ds = data_config["dataset"]
    raw_name = model_config.get("name")
    if raw_name is None or str(raw_name).strip().lower() in ("auto",):
        arch = _derive_architecture(data_config, model_config)
        date_tag = time.strftime("%Y%m%d")
        base_nm = f"{ds}_model_{arch}_{date_tag}"
        model_config["name"] = _unique_name(data_config["output_dir"], base_nm)
    else:
        model_config["name"] = raw_name.format(dataset=ds)

    parser = argparse.ArgumentParser(description="Pretrain BERT model")

    parser.add_argument("--train_data_file", default=data_config["train_file"], type=str)
    parser.add_argument("--each_epoch_eval", default=training_config["each_epoch_eval"], type=bool)
    parser.add_argument("--each_batch_eval", default=training_config["each_batch_eval"], type=bool)
    parser.add_argument("--each_checkpoint_eval", default=training_config["each_checkpoint_eval"], type=bool)

    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(data_config["output_dir"], model_config["name"]),
        help="output directory",
    )
    parser.add_argument("--model_type", type=str, default=model_config["type"])
    parser.add_argument("--model_name", type=str, default=model_config["name"])
    parser.add_argument("--gpu_start", default=gpu_config["start"], type=int)
    parser.add_argument("--gpu_num", default=1, type=int)

    parser.add_argument("--eval_data_file", default=data_config["eval_file"], type=str)
    parser.add_argument("--line_by_line", default=training_config["line_by_line"], action="store_true")
    parser.add_argument("--should_continue", action="store_true")
    parser.add_argument("--model_name_or_path", default=model_config["name_or_path"], type=str)

    parser.add_argument("--mlm", default=training_config["mlm"], action="store_true")
    parser.add_argument("--mlm_probability", type=float, default=0.15)

    parser.add_argument("--config_name", default=data_config["config_file"], type=str)
    parser.add_argument("--tokenizer_name", default=data_config["tokenizer_file"], type=str)
    parser.add_argument("--cache_dir", default=None, type=str)

    parser.add_argument("--block_size", default=training_config["max_seq_length"], type=int)

    parser.add_argument("--do_train", default=training_config["do_train"], action="store_true")
    parser.add_argument("--do_eval", default=training_config["do_eval"], action="store_true")
    parser.add_argument("--do_fune_tune", default=training_config["do_fine_tune"], action="store_true")
    parser.add_argument(
        "--evaluate_during_training",
        default=training_config["evaluate_during_training"],
        action="store_true",
    )

    parser.add_argument("--per_gpu_train_batch_size", default=training_config["batch_size"], type=int)
    parser.add_argument("--per_gpu_eval_batch_size", default=training_config["eval_batch_size"], type=int)

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", default=training_config["learning_rate"], type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=training_config["num_epochs"], type=float)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--warmup_proportion", default=training_config["warmup_proportion"], type=float)

    parser.add_argument("--logging_steps", type=int, default=training_config.get("logging_steps", 100))
    parser.add_argument("--save_steps", type=int, default=training_config.get("save_steps", 1000))
    parser.add_argument("--save_total_limit", type=int, default=None)
    parser.add_argument("--eval_all_checkpoints", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--overwrite_output_dir", default=training_config["overwrite_output_dir"], action="store_true")
    parser.add_argument("--overwrite_cache", default=training_config["overwrite_cache"], action="store_true")
    parser.add_argument("--seed_flag", default=training_config["seed_flag"], type=bool)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--fp16_opt_level", type=str, default="O1")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--server_ip", type=str, default="")
    parser.add_argument("--server_port", type=str, default="")

    parser.add_argument("--device", default=gpu_config.get("device", "auto"), type=str)

    parser.add_argument(
        "--eval_steps",
        type=int,
        default=int(training_config.get("eval_steps", 0)),
        help="Evaluate every N optimizer steps (after grad accumulation). 0 means disabled.",
    )

    parser.add_argument("--config", type=str, default=config_args.config)

    return parser.parse_args()


def resolve_device(device_str: str) -> torch.device:
    normalized = (device_str or "").strip().lower()
    if normalized in ("", "auto", "none"):
        return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if normalized.startswith("cpu"):
        return torch.device("cpu")
    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            return torch.device("cpu")
        try:
            dev = torch.device(normalized)
        except Exception:
            dev = torch.device("cuda:0")
        if torch.cuda.device_count() > 0:
            idx = dev.index if dev.index is not None else 0
            if idx >= torch.cuda.device_count():
                return torch.device("cuda:0")
        return dev
    try:
        dev = torch.device(device_str)
    except Exception:
        return torch.device("cpu")
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return dev


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, f"{checkpoint_prefix}-*"))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(rf".*{checkpoint_prefix}-([0-9]+)", path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    return [checkpoint[1] for checkpoint in checkpoints_sorted]


def validate_args(args):
    if args.model_type in ["bert", "roberta", "distilbert", "camembert"] and not args.mlm:
        raise ValueError("BERT-like models must be run with --mlm flag")

    if args.eval_data_file is None and args.do_eval:
        raise ValueError("Cannot do evaluation without an evaluation data file")

    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        args.model_name_or_path = sorted_checkpoints[-1]

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty. "
            f"Use --overwrite_output_dir to overcome."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()


def setup_logging(args):
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "training.log")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    _logger = logging.getLogger(__name__)
    _logger.warning(
        "Training env: rank=%s, device=%s, n_gpu=%s, distributed=%s, fp16=%s",
        args.local_rank,
        args.device,
        getattr(args, "n_gpu", None),
        bool(args.local_rank != -1),
        getattr(args, "fp16", False),
    )
    return _logger


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if getattr(args, "n_gpu", 0) > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def load_model_config(args):
    if args.config_name:
        config = AutoConfig.for_model(args.model_type).from_json_file(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError("config_name or model_name_or_path must be provided.")
    return config


def load_tokenizer(args):
    if args.tokenizer_name:
        Tokenizer = BertTokenizer
        tokenizer = Tokenizer(
            vocab_file=args.tokenizer_name,
            max_seq_length=args.block_size - 2,
            model_max_length=args.block_size,
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError("tokenizer_name or model_name_or_path must be provided.")

    if args.block_size <= 0:
        args.block_size = tokenizer.model_max_length
    else:
        args.block_size = min(args.block_size, tokenizer.model_max_length)

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    return tokenizer


def initialize_model(args, config, _logger):
    if args.model_name_or_path:
        model = AutoModelWithLMHead.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
    else:
        _logger.info("Training from scratch")
        if args.do_fune_tune:
            model = AutoModel.from_pretrained(args.output_dir)
        else:
            model = AutoModelWithLMHead.from_config(config)

    model.to(args.device)

    def analyze_model_computation(model, config, args, _logger):
        if args.local_rank not in [-1, 0]:
            return

        import math
        import torch
        import torch.nn as nn
        from contextlib import nullcontext

        def _unwrap(m):
            return m.module if hasattr(m, "module") else m

        def _count_params(m: nn.Module):
            return sum(p.numel() for p in m.parameters())

        def _bytes_of_params(m: nn.Module):
            return sum(p.numel() * p.element_size() for p in m.parameters())

        def _format_mb(x_bytes):
            return x_bytes / (1024 ** 2)

        def _safe_get_gpu_mem(device_index=0):
            try:
                free_b, total_b = torch.cuda.mem_get_info(device_index)
                return free_b, total_b
            except Exception:
                return None, None

        n_gpu = int(getattr(args, "n_gpu", 0) or 0)
        per_gpu_bs = int(getattr(args, "per_gpu_train_batch_size", 1) or 1)
        effective_bs = int(getattr(args, "train_batch_size", per_gpu_bs * max(1, n_gpu if n_gpu > 0 else 1)))
        seq_len = int(min(int(getattr(args, "block_size", 512) or 512), int(getattr(config, "max_position_embeddings", 512) or 512)))

        m = _unwrap(model)
        total_params = _count_params(m)
        trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)

        module_params = {}
        if hasattr(m, "bert"):
            module_params["bert.embeddings"] = _count_params(m.bert.embeddings)
            module_params["bert.encoder"] = _count_params(m.bert.encoder)
            module_params["bert.pooler"] = 0 if (getattr(m.bert, "pooler", None) is None) else _count_params(m.bert.pooler)
        if hasattr(m, "cls"):
            module_params["mlm.cls_head"] = _count_params(m.cls)
        if hasattr(m, "lm_head"):
            module_params["mlm.lm_head"] = _count_params(m.lm_head)

        tied = None
        try:
            inp = m.get_input_embeddings()
            outp = m.get_output_embeddings()
            if inp is not None and outp is not None:
                tied = (inp.weight.data_ptr() == outp.weight.data_ptr())
        except Exception:
            tied = None

        L = int(getattr(config, "num_hidden_layers", 0) or 0)
        H = int(getattr(config, "hidden_size", 0) or 0)
        A = int(getattr(config, "num_attention_heads", 0) or 0)
        I = int(getattr(config, "intermediate_size", 4 * H) or (4 * H))

        prof_forward_flops = None
        prof_train_flops = None
        prof_ok = False

        class _FwdWrapper(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base
            def forward(self, input_ids, attention_mask, labels=None):
                if labels is None:
                    out = self.base(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    out = self.base(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                return out

        wrapper = _FwdWrapper(m).to(args.device)

        if args.device.type == "cuda":
            try:
                from torch.profiler import profile, ProfilerActivity
                vocab = int(getattr(config, "vocab_size", 30522) or 30522)
                input_ids = torch.randint(0, vocab, (effective_bs, seq_len), device=args.device, dtype=torch.long)
                attention_mask = torch.ones((effective_bs, seq_len), device=args.device, dtype=torch.long)

                prob = 0.15
                labels = input_ids.clone()
                mask = (torch.rand_like(labels.float()) < prob)
                labels[~mask] = -100

                torch.cuda.synchronize()
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=False,
                    with_flops=True,
                    profile_memory=False
                ) as prof:
                    _ = wrapper(input_ids, attention_mask, labels=None)
                    torch.cuda.synchronize()

                fwd_flops = 0
                for e in prof.key_averages():
                    if hasattr(e, "flops") and e.flops is not None:
                        fwd_flops += e.flops
                if fwd_flops > 0:
                    prof_forward_flops = float(fwd_flops)
                    prof_ok = True

                wrapper.train()
                optim_ctx = nullcontext()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=False,
                    with_flops=True,
                    profile_memory=False
                ) as prof2:
                    out = wrapper(input_ids, attention_mask, labels=labels)
                    loss = out[0] if isinstance(out, (tuple, list)) else getattr(out, "loss", None)
                    if loss is None:
                        logits = out[0] if isinstance(out, (tuple, list)) else getattr(out, "logits")
                        loss = logits.float().mean()
                    loss.backward()
                    torch.cuda.synchronize()

                train_flops = 0
                for e in prof2.key_averages():
                    if hasattr(e, "flops") and e.flops is not None:
                        train_flops += e.flops
                if train_flops > 0:
                    prof_train_flops = float(train_flops)
                    prof_ok = True

            except Exception as e:
                _logger.warning(f"torch.profiler FLOPs failed, falling back to formula: {e}")
                prof_ok = False

        approx_forward_flops = effective_bs * seq_len * (12 * L * H * H + 2 * L * H * I) if (L and H and I) else None
        approx_train_flops = approx_forward_flops * 3 if approx_forward_flops is not None else None

        params_bytes = _bytes_of_params(m)

        if getattr(args, "fp16", False):
            numel = total_params
            param_fp16_bytes = numel * 2
            master_fp32_bytes = numel * 4
            grad_fp16_bytes = numel * 2
            adam_m_bytes = numel * 4
            adam_v_bytes = numel * 4
            param_related_train_bytes = param_fp16_bytes + master_fp32_bytes + grad_fp16_bytes + adam_m_bytes + adam_v_bytes
        else:
            param_related_train_bytes = params_bytes * 4

        bytes_per_act = 2 if getattr(args, "fp16", False) else 4
        act_factor = 8
        activation_bytes_rough = effective_bs * seq_len * H * L * bytes_per_act * act_factor if (L and H) else 0
        total_train_bytes_rough = param_related_train_bytes + activation_bytes_rough
        recommended_bytes_30 = int(total_train_bytes_rough * 1.3)

        step_time_ms = None
        throughput_sps = None
        peak_mem_mb = None

        if args.device.type == "cuda":
            try:
                vocab = int(getattr(config, "vocab_size", 30522) or 30522)
                input_ids = torch.randint(0, vocab, (effective_bs, seq_len), device=args.device, dtype=torch.long)
                attention_mask = torch.ones((effective_bs, seq_len), device=args.device, dtype=torch.long)
                labels = input_ids.clone()
                mask = (torch.rand_like(labels.float()) < 0.15)
                labels[~mask] = -100

                wrapper.train()
                warmup = 2
                iters = 5
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

                for _ in range(warmup):
                    out = wrapper(input_ids, attention_mask, labels=labels)
                    loss = out[0] if isinstance(out, (tuple, list)) else getattr(out, "loss", None)
                    if loss is None:
                        logits = out[0] if isinstance(out, (tuple, list)) else getattr(out, "logits")
                        loss = logits.float().mean()
                    loss.backward()
                    wrapper.zero_grad(set_to_none=True)

                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(iters):
                    out = wrapper(input_ids, attention_mask, labels=labels)
                    loss = out[0] if isinstance(out, (tuple, list)) else getattr(out, "loss", None)
                    if loss is None:
                        logits = out[0] if isinstance(out, (tuple, list)) else getattr(out, "logits")
                        loss = logits.float().mean()
                    loss.backward()
                    wrapper.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                t1 = time.time()

                step_time_ms = (t1 - t0) / iters * 1000.0
                throughput_sps = effective_bs / ((t1 - t0) / iters)
                peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

            except Exception as e:
                _logger.warning(f"micro-benchmark failed: {e}")

        _logger.info("=" * 80)
        _logger.info("Model Analysis Report (improved)")
        _logger.info("=" * 80)
        _logger.info("Architecture Info:")
        _logger.info(f"  • Type: {m.__class__.__name__}")
        _logger.info(f"  • Layers(L): {L}")
        _logger.info(f"  • Hidden Dim(H): {H}")
        _logger.info(f"  • Heads(A): {A}")
        _logger.info(f"  • FFN Dim(I): {I}")
        _logger.info(f"  • Vocab: {getattr(config, 'vocab_size', None)}")
        if tied is not None:
            _logger.info(f"  • Weight tying(emb<->decoder): {tied}")

        _logger.info("\nParameters:")
        _logger.info(f"  • Total: {total_params:,} ({total_params/1e6:.3f}M)")
        _logger.info(f"  • Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        if module_params:
            _logger.info("  • Modules:")
            for k, v in module_params.items():
                _logger.info(f"    - {k}: {v:,} ({v/1e6:.3f}M)")

        _logger.info(f"\nFLOPs (batch={effective_bs}, seq={seq_len})")
        if prof_forward_flops is not None:
            _logger.info(f"  • profiler forward FLOPs: {prof_forward_flops:.3e}")
        if prof_train_flops is not None:
            _logger.info(f"  • profiler train FLOPs (fwd+bwd): {prof_train_flops:.3e}")
        if not prof_ok and approx_forward_flops is not None:
            _logger.info(f"  • approx forward FLOPs (simplified): {approx_forward_flops:.3e}")
            _logger.info(f"  • approx train FLOPs (≈x3): {approx_train_flops:.3e}")

        if args.device.type == "cuda":
            free_b, total_b = _safe_get_gpu_mem(device_index=(args.device.index or 0))
            _logger.info("\nMemory Estimation:")
            _logger.info(f"  • Weights: {_format_mb(params_bytes):.1f} MB")
            _logger.info(f"  • Training params overhead: {_format_mb(param_related_train_bytes):.1f} MB")
            _logger.info(f"  • Activations: {_format_mb(activation_bytes_rough):.1f} MB")
            _logger.info(f"  • Total training: {_format_mb(total_train_bytes_rough):.1f} MB ({total_train_bytes_rough/(1024**3):.2f} GiB)")
            _logger.info(f"  • Recommended GPU (30% buffer): {_format_mb(recommended_bytes_30):.1f} MB ({recommended_bytes_30/(1024**3):.2f} GiB)")
            if free_b is not None:
                _logger.info(f"  • GPU free/total: {_format_mb(free_b):.1f} / {_format_mb(total_b):.1f} MB")
            if peak_mem_mb is not None:
                _logger.info(f"  • Benchmark peak allocated: {peak_mem_mb:.1f} MB")

            if step_time_ms is not None:
                _logger.info("\nPerformance (dummy batch):")
                _logger.info(f"  • Step time: {step_time_ms:.2f} ms")
                _logger.info(f"  • Throughput: {throughput_sps:.2f} samples/sec")

        _logger.info("=" * 80)

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "module_params": module_params,
            "weight_tying": tied,
            "prof_forward_flops": prof_forward_flops,
            "prof_train_flops": prof_train_flops,
            "approx_forward_flops": approx_forward_flops,
            "approx_train_flops": approx_train_flops,
            "params_mb": _format_mb(params_bytes),
            "param_related_train_mb_rough": _format_mb(param_related_train_bytes),
            "activation_mb_rough": _format_mb(activation_bytes_rough),
            "step_time_ms": step_time_ms,
            "throughput_sps": throughput_sps,
            "peak_allocated_mb": peak_mem_mb,
        }


    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    try:
        _logger.warning("[ANALYZE] start model analysis...")
        analysis_result = analyze_model_computation(model, config, args, _logger)
        _logger.warning("[ANALYZE] done.")
    except Exception as e:
        _logger.exception("[ANALYZE] failed: %s", e)
    return model


def load_and_cache_examples(args, tokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        return LineByLineTextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)
    return TextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)


# ========= Dynamic padding collate =========
def make_collate_fn(tokenizer: PreTrainedTokenizer):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("tokenizer.pad_token_id is None. Please set pad_token for tokenizer.")

    def collate_fn(examples: List[torch.Tensor]) -> torch.Tensor:
        return pad_sequence(examples, batch_first=True, padding_value=pad_id)

    return collate_fn


# ========= Fast MLM masking =========
def _torch_isin(elements: torch.Tensor, test_elements: torch.Tensor) -> torch.Tensor:
    if hasattr(torch, "isin"):
        return torch.isin(elements, test_elements)
    out = torch.zeros_like(elements, dtype=torch.bool)
    for v in test_elements.view(-1).tolist():
        out |= (elements == v)
    return out


def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:
    if tokenizer.mask_token is None:
        raise ValueError("Tokenizer has no mask token; MLM requires mask_token.")

    labels = inputs.clone()

    probability_matrix = torch.full(labels.shape, args.mlm_probability, device=labels.device)

    special_ids = torch.tensor(tokenizer.all_special_ids, device=labels.device, dtype=labels.dtype)
    special_mask = _torch_isin(labels, special_ids)
    probability_matrix.masked_fill_(special_mask, value=0.0)

    if tokenizer.pad_token_id is not None:
        pad_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(pad_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100

    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=labels.device)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5, device=labels.device)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long, device=labels.device)
    inputs[indices_random] = random_words[indices_random]

    return inputs, labels


# ========= RNG isolation for eval =========
class _RNGSnapshot:
    def __init__(self):
        self.py_state = random.getstate()
        self.np_state = np.random.get_state()
        self.torch_state = torch.get_rng_state()
        self.cuda_states = None
        if torch.cuda.is_available():
            try:
                self.cuda_states = torch.cuda.get_rng_state_all()
            except Exception:
                self.cuda_states = None

    def restore(self):
        random.setstate(self.py_state)
        np.random.set_state(self.np_state)
        torch.set_rng_state(self.torch_state)
        if self.cuda_states is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(self.cuda_states)
            except Exception:
                pass


# ========= Evaluate / Train =========
def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix="") -> Dict:
    eval_output_dir = args.output_dir
    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)
    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    collate_fn = make_collate_fn(tokenizer)
    eval_sampler = (
        RandomSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset, shuffle=False)
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=collate_fn,
        drop_last=False,
    )

    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    snap = _RNGSnapshot()
    try:
        if getattr(args, "seed_flag", False):
            set_seed(args)

        model.eval()
        eval_loss = 0.0
        nb_eval_steps = 0

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = batch.to(args.device)
            attention_mask = (batch != tokenizer.pad_token_id).long()

            if args.mlm:
                inputs, labels = mask_tokens(batch.clone(), tokenizer, args)
            else:
                inputs, labels = batch, batch

            with torch.no_grad():
                outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            eval_loss += loss.item()
            nb_eval_steps += 1

        eval_loss = eval_loss / max(1, nb_eval_steps)
        perplexity = torch.exp(torch.tensor(eval_loss))

        return {"perplexity": perplexity, "eval_loss": eval_loss}
    finally:
        snap.restore()


def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, int, float, float]:
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    collate_fn = make_collate_fn(tokenizer)
    train_sampler = (
        RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset, shuffle=True)
    )
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=collate_fn,
        drop_last=False,
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_proportion * t_total),
        num_training_steps=int(t_total),
    )

    if (
        args.model_name_or_path
        and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
        and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if args.n_gpu > 1:
        device_ids = list(range(args.gpu_start, args.gpu_start + args.n_gpu))
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", int(args.num_train_epochs))
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", int(t_total))

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
            logger.info("  Continuing training from checkpoint, global_step=%d, epoch=%d", global_step, epochs_trained)
        except ValueError:
            logger.info("  Starting training (no global_step in checkpoint name).")

    model_to_resize = model.module if hasattr(model, "module") else model
    model_to_resize.resize_token_embeddings(len(tokenizer))

    model.zero_grad()

    set_seed(args)

    eval_steps = 0
    if getattr(args, "each_batch_eval", False):
        eval_steps = 1
    else:
        eval_steps = int(getattr(args, "eval_steps", 0) or 0)

    best_eval_loss = 9e8
    best_train_loss = 9e8
    best_epoch = 0
    best_model = None

    train_iterator = range(epochs_trained, int(args.num_train_epochs))
    for e in train_iterator:
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(e)

        nb_tr_steps = 0
        tr_loss = 0.0

        epoch_iterator = tqdm(
            train_dataloader,
            desc=f"Epoch {e+1}/{int(args.num_train_epochs)}",
            miniters=100,
        )

        for step, batch in enumerate(epoch_iterator):
            start_time = time.time()

            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            batch = batch.to(args.device)
            attention_mask = (batch != tokenizer.pad_token_id).long()

            if args.mlm:
                inputs, labels = mask_tokens(batch.clone(), tokenizer, args)
            else:
                inputs, labels = batch, batch

            model.train()
            outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                from apex import amp
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            nb_tr_steps += 1

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    from apex import amp
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                epoch_iterator.set_postfix(
                    loss=f"{tr_loss/nb_tr_steps:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.7f}",
                )

                if eval_steps > 0 and (global_step % eval_steps == 0):
                    eval_result = evaluate(args, model, tokenizer, prefix=f"step-{global_step}")
                    epoch_iterator.set_postfix(
                        loss=f"{tr_loss/nb_tr_steps:.4f}",
                        eval_loss=f"{eval_result['eval_loss']:.4f}",
                        lr=f"{scheduler.get_last_lr()[0]:.7f}",
                    )

            if args.max_steps > 0 and global_step > args.max_steps:
                break

        eval_result = None
        train2eval_loss = tr_loss / max(1, nb_tr_steps)
        if args.each_epoch_eval:
            eval_result = evaluate(args, model, tokenizer, prefix=f"epoch-{e+1}")

            loss_txt = os.path.join(args.output_dir, "train_eval_loss.txt")
            mode = "w" if e == 0 else "a"
            with open(loss_txt, mode, encoding="utf-8") as f:
                f.write(f"-epoch {e} -train_loss {train2eval_loss:.4f} -eval_loss {eval_result['eval_loss']:.4f}\n")

            logger.info(
                "------------epoch %d -train_loss %.4f -eval_loss %.4f----------------",
                e,
                train2eval_loss,
                eval_result["eval_loss"],
            )
            tqdm.write(
                f"Epoch {e+1}/{int(args.num_train_epochs)} done: "
                f"loss={train2eval_loss:.4f}, eval_loss={eval_result['eval_loss']:.4f}"
            )

        if eval_result is not None and best_eval_loss > eval_result["eval_loss"]:
            best_eval_loss = eval_result["eval_loss"]
            best_train_loss = train2eval_loss
            best_epoch = e
            best_model = copy.deepcopy(model)

            print(".................saving best model..............")
            os.makedirs(args.output_dir, exist_ok=True)
            model_to_save = best_model.module if hasattr(best_model, "module") else best_model
            model_to_save.save_pretrained(args.output_dir)
            torch.save(best_model.state_dict(), os.path.join(args.output_dir, args.model_name + ".pth"))

        if args.max_steps > 0 and global_step > args.max_steps:
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    os.makedirs(args.output_dir, exist_ok=True)
    if best_model is None:
        best_model = model
        best_epoch = int(args.num_train_epochs) - 1
        best_train_loss = tr_loss / max(1, nb_tr_steps)
        if args.do_eval:
            end_eval = evaluate(args, model, tokenizer, prefix="final")
            best_eval_loss = end_eval["eval_loss"]

    model_to_save = best_model.module if hasattr(best_model, "module") else best_model
    model_to_save.save_pretrained(args.output_dir)
    torch.save(best_model.state_dict(), os.path.join(args.output_dir, args.model_name + ".pth"))
    logger.info("Saving epoch-%d-best_eval_loss-%.4f model to %s", best_epoch, best_eval_loss, args.output_dir)

    return global_step, best_epoch, best_eval_loss, best_train_loss


def train_and_evaluate(args, model, tokenizer, config, _logger):
    if args.local_rank == 0:
        torch.distributed.barrier()

    _logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()

        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, best_epoch, best_eval_loss, best_train_loss = train(args, train_dataset, model, tokenizer)
        _logger.info(
            "In all, %d epoch were trained, global steps were %d. best epoch=%d, best train loss=%f, best eval loss=%f.",
            int(args.num_train_epochs),
            global_step,
            best_epoch,
            best_train_loss,
            best_eval_loss,
        )

    if args.do_eval:
        model_eval = AutoModelWithLMHead.from_config(config)
        model_eval.to(args.device)
        model_eval.load_state_dict(torch.load(os.path.join(args.output_dir, args.model_name + ".pth"), map_location=args.device))

        eval_result = evaluate(args, model_eval, tokenizer, prefix="final-eval")
        _logger.info("Final eval: eval_loss=%.4f, ppl=%s", eval_result["eval_loss"], str(eval_result["perplexity"]))

        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
        collate_fn = make_collate_fn(tokenizer)
        train_sampler = (
            RandomSampler(train_dataset)
            if args.local_rank == -1
            else DistributedSampler(train_dataset, shuffle=False)
        )
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=args.train_batch_size,
            collate_fn=collate_fn,
            drop_last=False,
        )

        model_eval.eval()
        train2eval_loss = 0.0
        nb_train_eval_steps = 0
        for batch in tqdm(train_dataloader, desc="TrainSet Evaluating"):
            batch = batch.to(args.device)
            attention_mask = (batch != tokenizer.pad_token_id).long()
            if args.mlm:
                inputs, labels = mask_tokens(batch.clone(), tokenizer, args)
            else:
                inputs, labels = batch, batch

            with torch.no_grad():
                outputs = model_eval(input_ids=inputs, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            train2eval_loss += loss.item()
            nb_train_eval_steps += 1

        train2eval_loss = train2eval_loss / max(1, nb_train_eval_steps)
        _logger.info("---------------- Eval directly (trainset), train_loss %.4f | eval_loss %.4f ----------------",
                     train2eval_loss, eval_result["eval_loss"])


def main():
    args = get_pretrain_args()
    requested_device = args.device
    args.device = resolve_device(args.device)

    if args.device.type == "cuda" and torch.cuda.is_available():
        avail = torch.cuda.device_count()
        usable = max(0, avail - int(args.gpu_start))
        args.n_gpu = max(1, min(int(args.gpu_num), usable)) if usable > 0 else 0
    else:
        args.n_gpu = 0

    _logger = setup_logging(args)
    _logger.info(
        "PyTorch: %s, torch.version.cuda=%s, torch.version.hip=%s, cuda_available=%s, cuda_device_count=%s",
        torch.__version__,
        torch.version.cuda,
        getattr(torch.version, "hip", None),
        torch.cuda.is_available(),
        torch.cuda.device_count(),
    )
    if args.device.type != "cuda" and str(requested_device).lower().startswith("cuda"):
        _logger.warning(
            "No available GPU detected, falling back to CPU: requested=%s, using=%s",
            requested_device,
            args.device,
        )

    validate_args(args)

    if args.seed_flag:
        set_seed(args)

    model_config = load_model_config(args)
    tokenizer = load_tokenizer(args)
    model = initialize_model(args, model_config, _logger)

    train_and_evaluate(args, model, tokenizer, model_config, _logger)


if __name__ == "__main__":
    main()
