import yaml
import argparse
from pathlib import Path
import os
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
import re

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW

logger = logging.getLogger(__name__)

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

class ConfigObj:  # Configuration wrapper to convert dicts into objects
    def __init__(self, d):  # Initializer
        for k, v in d.items():  # Iterate config dictionary
            if isinstance(v, dict):  # If the value is a dict
                v = ConfigObj(v)  # Recursively wrap into ConfigObj
            setattr(self, k, v)  # Assign as attribute

# 创建假的SummaryWriter类
class SummaryWriter:
    def __init__(self, *args, **kwargs):
        pass
    
    def add_scalar(self, *args, **kwargs):
        pass
    
    def close(self):
        pass

class TextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, config, file_path: str, block_size=512):
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        filename_wo_ext = os.path.splitext(filename)[0]  # Strip file extension

        cached_features_file = os.path.join(
            '../../TrafficData/DataCache/', filename_wo_ext + "_" + config.model.type + "_mlm_" + str(block_size) + "_cached"
        )

        block_size = block_size - (tokenizer.model_max_length - tokenizer.max_len_single_sentence)

        if os.path.exists(cached_features_file) and not config.training.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)
            #CLS, SEP = '[CLS]', '[SEP]'
            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.readlines()
                tokenized_text = []
                line_count = 0
                valid_line_count = 0
                for line in text:
                    line_count += 1
                    line = line.strip().split(' ')
                    if len(line) > block_size:
                        line = line[:block_size]
                    if len(line)==1:  # Skip empty token
                        continue
                    tokenized_line = tokenizer.convert_tokens_to_ids(line)
                    if len(tokenized_line) > 0:
                        valid_line_count += 1
                        tokenized_text.append(tokenized_line)
            """
            tokenized_text : list, [[id1, id2, ..., ], [id1, id2, ..., ], ..., [id1, id2, ..., ]], 
            self.examples : list, [[cls_id, id1, id2, ..., seq_id], [cls_id, id1, id2, ..., seq_id], ...], 
            """
            for i in range(len(tokenized_text)):  # Truncate in block of block_size
                self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i]))

            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should look for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info("Processed %d/%d valid lines", valid_line_count, line_count)

            logger.info("Saving features into cached file %s", cached_features_file)
            if not os.path.exists('../../TrafficData/DataCache/'):
                os.makedirs('../../TrafficData/DataCache/', exist_ok=True)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

            if len(self.examples) < 10:  # Too few examples; may indicate a problem
                logger.warning("WARNING: Only %d examples were processed! This is abnormally low.", len(self.examples))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)

class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, config, file_path: str, block_size=512):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)

def get_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:  # Open config file
        config = ConfigObj(yaml.safe_load(f))
        config.data.train_file = config.data.train_file.format(dataset=config.data.dataset)
        config.data.eval_file = config.data.eval_file.format(dataset=config.data.dataset)
        config.data.output_dir = config.data.output_dir.format(dataset=config.data.dataset)
        config.model.name = config.model.name.format(dataset=config.data.dataset)

    return config
    
def setup_logging(config):
    """Set up the logging system."""
    # Ensure output directory exists
    os.makedirs(config.data.output_dir, exist_ok=True)

    # Create log file path
    log_file = os.path.join(config.data.output_dir, "training.log")

    # Basic logging configuration
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if config.log.local_rank in [-1, 0] else logging.WARN,
        handlers=[
            logging.FileHandler(log_file),  # File handler
            logging.StreamHandler()  # Console handler
        ]
    )
    # Get logger
    logger = logging.getLogger(__name__)
    # Record environment info
    logger.warning(
        "训练环境: 进程排名=%s, 设备=%s, GPU数量=%s, 分布式=%s, 16位精度=%s",
        config.log.local_rank, config.gpu.device, config.gpu.n_gpu,
        bool(config.log.local_rank != -1), getattr(config.training, 'fp16', False)
    )
    return logger

def _sorted_checkpoints(config, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    import re  # Ensure regex module available
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(config.data.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted

def validate_args(config):
    """Validate the correctness of configuration arguments."""
    # BERT-like models must use MLM
    if config.model.type in ["bert", "roberta", "distilbert", "camembert"] and not config.training.mlm:
        raise ValueError("BERT-like models must be run with --mlm flag")
    
    # 评估时必须要有评估的数据集
    if config.data.eval_file is None and config.training.do_eval:
        raise ValueError("Cannot do evaluation without an evaluation data file")
    
    # 输出目录存在且不为空时的处理
    if (os.path.exists(config.data.output_dir) and os.listdir(config.data.output_dir) 
            and config.training.do_train and not config.training.overwrite_output_dir):
        raise ValueError(f"Output directory ({config.data.output_dir}) already exists and is not empty") 
    
    if config.training.should_continue:
        sorted_checkpoints = _sorted_checkpoints(config)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            config.model.name_or_path = sorted_checkpoints[-1]
    
   
    if ( 
            os.path.exists(config.data.output_dir)
            and os.listdir(config.data.output_dir)
            and config.training.do_train
            and not config.training.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                config.data.output_dir  
            )
        )
    
    
    if not os.path.exists(config.data.output_dir):
        os.makedirs(config.data.output_dir, exist_ok=True)
    
    # Load pretrained model and tokenizer
    if config.log.local_rank not in [-1, 0]: # 。。。。。。。。。。
        torch.distributed.barrier() 

def load_model_config(config):
    """加载配置和分词器"""
    if config.data.config_file:
        model_config = AutoConfig.for_model(config.model.type).from_json_file(config.data.config_file)
    elif config.model.name_or_path:
        model_config = AutoConfig.from_pretrained(config.model.name_or_path, cache_dir=config.model.cache_dir)
    else:
        # When we release a pip version exposing CONFIG_MAPPING,
        # we can do `config = CONFIG_MAPPING[args.model_type]()`.
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --config_name"
        )
    return model_config

def load_tokenizer(config):
    if config.data.tokenizer_file:
        Tokenizer = BertTokenizer  # Class alias for transformers tokenizer
        tokenizer = Tokenizer(vocab_file=config.data.tokenizer_file, 
                              max_seq_length=config.training.max_seq_length - 2,
                              model_max_length=config.training.max_seq_length) 
    elif config.model.name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path, cache_dir=config.model.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )
    
    # Limit tokenizer maximum length
    if config.training.max_seq_length <= 0:
        config.training.max_seq_length = tokenizer.model_max_length
        # Our input block size will be the max possible for the model
    else:
        config.training.max_seq_length = min(config.training.max_seq_length, tokenizer.model_max_length)
        
    return tokenizer

def initialize_model(config, model_config, logger):
    
    if config.model.name_or_path: 
        model = AutoModelWithLMHead.from_pretrained(
            config.model.name_or_path,
            from_tf=bool(".ckpt" in config.model.name_or_path),
            config=model_config,
            cache_dir=config.model.cache_dir,
        )
    else:
        logger.info("from scratch")
        if config.training.do_fine_tune:       
            model = AutoModel.from_pretrained(config.data.output_dir)
        else: 
            model = AutoModelWithLMHead.from_config(model_config)
            
    model.to(config.gpu.device) 
    config.training.train_batch_size = config.training.batch_size * max(1, config.gpu.n_gpu) 
    return model

def load_and_cache_examples(config, tokenizer, evaluate=False):
    file_path = config.data.eval_file if evaluate else config.data.train_file
    if config.training.line_by_line:
        return LineByLineTextDataset(tokenizer, config, file_path=file_path, block_size=config.training.max_seq_length)
    else:
        return TextDataset(tokenizer, config, file_path=file_path, block_size=config.training.max_seq_length)

def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, config) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. "
            "Remove the mlm flag if you want to use this tokenizer."
        )
    labels = inputs.clone()
    mlm_prob = float(getattr(config.training, "mlm_probability", 0.15))
    probability_matrix = torch.full(labels.shape, mlm_prob)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer.pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]
    return inputs, labels

def train_and_evaluate(config, model, tokenizer, model_config, logger):
    if config.log.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", config)
    
    # Training
    if config.training.do_train: 
        if config.log.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = load_and_cache_examples(config, tokenizer, evaluate=False)

        if config.log.local_rank == 0:
            torch.distributed.barrier()

        global_step, best_epoch, best_eval_loss, best_train_loss = train(config, train_dataset, model, tokenizer)
        logger.info(
            "In all, %d epoch were trained, global steps were %d. best epoch is %d, best train loss is %f, best eval loss is %f.",
            config.training.num_epochs, global_step, best_epoch, best_train_loss, best_eval_loss)
    
    if config.training.do_eval:
        train_dataset = load_and_cache_examples(config, tokenizer, evaluate=False)
        # Convert Dataset to a list of sequences
        train_sequences = [train_dataset[i] for i in range(len(train_dataset))]
        train_dataset_pad = pad_sequence(train_sequences, batch_first=True, padding_value=tokenizer.pad_token_id)
        train_sampler = RandomSampler(train_dataset_pad) if config.log.local_rank == -1 else DistributedSampler(
            train_dataset_pad)
        train_dataloader = DataLoader(
            train_dataset_pad, sampler=train_sampler, batch_size=config.training.batch_size
        )
        # Evaluate on the entire train set
        model = AutoModelWithLMHead.from_config(model_config)
        model.to(config.gpu.device)
        model.load_state_dict(torch.load(os.path.join(config.data.output_dir, config.model.name + ".pth")))

        eval_result = evaluate(config, model, tokenizer)
        train2eval_loss = 0
        nb_train_eval_steps = 0
        for batch in tqdm(train_dataloader, desc="TrainSet Evaluating"):
            inputs, labels = mask_tokens(batch, tokenizer, config) if config.training.mlm else (batch, batch)
            inputs = inputs.to(config.gpu.device)
            labels = labels.to(config.gpu.device)

            with torch.no_grad():
                outputs = model(inputs, labels=labels)
                loss = outputs[0]

            # Add missing accumulation logic
            train2eval_loss += loss.item()
            nb_train_eval_steps += 1

        train2eval_loss = train2eval_loss / nb_train_eval_steps
        logger.info("---------------- Eval directly, -train_loss %.4f -eval_loss %.4f----------------\n" %
                    (train2eval_loss, eval_result['eval_loss']))

def set_seed(config):
    
    seed = 42
    if hasattr(config, "training") and hasattr(config.training, "seed"):
        try:
            seed = int(config.training.seed)
        except Exception:
            pass
    elif hasattr(config, "seed"):
        try:
            seed = int(config.seed)
        except Exception:
            pass

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_gpu = 0
    if hasattr(config, "gpu") and hasattr(config.gpu, "n_gpu"):
        try:
            n_gpu = int(config.gpu.n_gpu)
        except Exception:
            n_gpu = 0
    if n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate(config, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix="") -> Dict:
    eval_output_dir = config.data.output_dir
    eval_dataset = load_and_cache_examples(config, tokenizer, evaluate=True)
    if config.log.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    eval_batch_size = int(config.training.eval_batch_size) * max(1, int(config.gpu.n_gpu))

    # Convert Dataset to a list of sequences
    eval_sequences = [eval_dataset[i] for i in range(len(eval_dataset))]
    eval_dataset_pad = pad_sequence(eval_sequences, batch_first=True, padding_value=tokenizer.pad_token_id)
    eval_sampler = RandomSampler(eval_dataset_pad) if config.log.local_rank == -1 else DistributedSampler(eval_dataset_pad)
    eval_dataloader = DataLoader(
        eval_dataset_pad, sampler=eval_sampler, batch_size=eval_batch_size
    )
    set_seed(config)

    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", eval_batch_size)

    model.eval()
    eval_loss = 0
    nb_eval_steps = 0

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = mask_tokens(batch, tokenizer, config) if config.training.mlm else (batch, batch)
        inputs = inputs.to(config.gpu.device)
        labels = labels.to(config.gpu.device)

        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            loss = outputs[0]

        if config.gpu.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if config.training.gradient_accumulation_steps > 1:
            loss = loss / config.training.gradient_accumulation_steps

        eval_loss += loss.item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity, "eval_loss": eval_loss}

    return result

def train(config, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, int, float, float]:
    # Log to TensorBoard (replace with your actual logger if needed)
    if config.log.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    # Convert Dataset to a list of sequences and pad
    train_sequences = [train_dataset[i] for i in range(len(train_dataset))]
    train_dataset_pad = pad_sequence(train_sequences, batch_first=True, padding_value=tokenizer.pad_token_id)
    train_sampler = RandomSampler(train_dataset_pad) if config.log.local_rank == -1 else DistributedSampler(train_dataset_pad)
    train_dataloader = DataLoader(
        train_dataset_pad, sampler=train_sampler, batch_size=config.training.train_batch_size
    )

    # Compute total steps and epochs
    if config.training.max_steps > 0:
        t_total = int(config.training.max_steps)
        config.training.num_epochs = int(config.training.max_steps // max(1, (len(train_dataloader) // int(config.training.gradient_accumulation_steps))) + 1)
    else:
        t_total = (len(train_dataloader) // int(config.training.gradient_accumulation_steps)) * int(config.training.num_epochs)

    # Optimizer and scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": float(config.training.weight_decay),
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=float(config.training.learning_rate),
        eps=float(config.training.adam_epsilon),
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(float(config.training.warmup_proportion) * t_total),
        num_training_steps=t_total,
    )

    # Restore optimizer/scheduler if available
    if (
        getattr(config.model, "name_or_path", None)
        and os.path.isdir(config.model.name_or_path)
        and os.path.isfile(os.path.join(config.model.name_or_path, "optimizer.pt"))
        and os.path.isfile(os.path.join(config.model.name_or_path, "scheduler.pt"))
    ):
        optimizer.load_state_dict(torch.load(os.path.join(config.model.name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(config.model.name_or_path, "scheduler.pt")))

    # fp16
    if getattr(config.training, "fp16", False):
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex to enable fp16 training: https://www.github.com/nvidia/apex")
        model, optimizer = amp.initialize(model, optimizer, opt_level=getattr(config.training, "fp16_opt_level", "O1"))

    # Multi-GPU / Distributed
    if int(config.gpu.n_gpu) > 1:
        model = torch.nn.DataParallel(model)

    if int(config.log.local_rank) != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[int(config.log.local_rank)], output_device=int(config.log.local_rank), find_unused_parameters=True
        )

    # Training logs
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", int(config.training.num_epochs))
    logger.info("  Instantaneous batch size per GPU = %d", int(config.training.batch_size))
    world_size = torch.distributed.get_world_size() if int(config.log.local_rank) != -1 else 1
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        int(config.training.train_batch_size) * int(config.training.gradient_accumulation_steps) * world_size,
    )
    logger.info("  Gradient Accumulation steps = %d", int(config.training.gradient_accumulation_steps))
    logger.info("  Total optimization steps = %d", int(t_total))

    # Resume step inference from checkpoint
    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    if getattr(config.model, "name_or_path", None) and os.path.exists(config.model.name_or_path):
        try:
            checkpoint_suffix = config.model.name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // max(1, (len(train_dataloader) // int(config.training.gradient_accumulation_steps)))
            steps_trained_in_current_epoch = global_step % max(1, (len(train_dataloader) // int(config.training.gradient_accumulation_steps)))

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    # Token embedding resize (tokenizer may be extended)
    model_to_resize = model.module if hasattr(model, "module") else model
    model_to_resize.resize_token_embeddings(len(tokenizer))

    model.zero_grad()

    train_iterator = range(epochs_trained, int(config.training.num_epochs))
    set_seed(config)

    best_eval_loss = 9e8
    best_train_loss = 9e8
    best_epoch = -1
    best_model = None

    for e in train_iterator:
        nb_tr_steps = 0
        tr_loss = 0.0
        steps_trained_left = steps_trained_in_current_epoch

        # Progress bar (update every 100 batches)
        epoch_iterator = tqdm(
            train_dataloader,
            desc=f"Epoch {e + 1}/{int(config.training.num_epochs)}",
            miniters=100
        )
        for step, batch in enumerate(epoch_iterator):
            if steps_trained_left > 0:
                steps_trained_left -= 1
                continue

            inputs, labels = mask_tokens(batch, tokenizer, config) if getattr(config.training, "mlm", False) else (batch, batch)
            inputs = inputs.to(config.gpu.device)
            labels = labels.to(config.gpu.device)

            model.train()
            outputs = model(inputs, labels=labels)
            loss = outputs[0]

            if int(config.gpu.n_gpu) > 1:
                loss = loss.mean()
            if int(config.training.gradient_accumulation_steps) > 1:
                loss = loss / int(config.training.gradient_accumulation_steps)

            if getattr(config.training, "fp16", False):
                from apex import amp
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            nb_tr_steps += 1

            if (step + 1) % int(config.training.gradient_accumulation_steps) == 0:
                if getattr(config.training, "fp16", False):
                    from apex import amp
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), float(config.training.max_grad_norm))
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.training.max_grad_norm))
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                # Update progress bar info
                epoch_iterator.set_postfix(
                    loss=f"{tr_loss / max(1, nb_tr_steps):.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.7f}"
                )

                # Optional: per-batch evaluation
                if getattr(config.training, "each_batch_eval", False):
                    eval_result = evaluate(config, model, tokenizer)
                    epoch_iterator.set_postfix(
                        loss=f"{tr_loss / max(1, nb_tr_steps):.4f}",
                        eval_loss=f"{eval_result['eval_loss']:.4f}"
                    )

            if int(config.training.max_steps) > 0 and global_step > int(config.training.max_steps):
                break

        # Per-epoch evaluation (recommended: enabled in pretrain.yaml)
        if getattr(config.training, "each_epoch_eval", True):
            eval_result = evaluate(config, model, tokenizer)
            train2eval_loss = tr_loss / max(1, nb_tr_steps)

            # Write to file
            output_dir = config.data.output_dir
            os.makedirs(output_dir, exist_ok=True)
            fpath = os.path.join(output_dir, "train_eval_loss.txt")
            mode = "w" if e == 0 else "a"
            with open(fpath, mode, encoding="utf-8") as f:
                f.write("-epoch %d -train_loss %.4f -eval_loss %.4f\n" % (e, train2eval_loss, eval_result['eval_loss']))

            logger.info("------------epoch %d -train_loss %.4f -eval_loss %.4f----------------", e, train2eval_loss, eval_result['eval_loss'])
            tqdm.write(f"Epoch {e+1}/{int(config.training.num_epochs)} completed: loss={train2eval_loss:.4f}, eval_loss={eval_result['eval_loss']:.4f}")

            # Save best
            if best_eval_loss > eval_result['eval_loss']:
                best_eval_loss = eval_result['eval_loss']
                best_train_loss = train2eval_loss
                best_epoch = e
                best_model = copy.deepcopy(model)
                print(".................saving model..............")
                model_to_save = best_model.module if hasattr(best_model, "module") else best_model
                model_to_save.save_pretrained(output_dir)
                torch.save(best_model.state_dict(), os.path.join(output_dir, config.model.name + ".pth"))

        if int(config.training.max_steps) > 0 and global_step > int(config.training.max_steps):
            break

    if config.log.local_rank in [-1, 0]:
        tb_writer.close()

    # Finalize: save best model again
    output_dir = config.data.output_dir
    os.makedirs(output_dir, exist_ok=True)
    if best_model is not None:
        model_to_save = best_model.module if hasattr(best_model, "module") else best_model
        model_to_save.save_pretrained(output_dir)
        torch.save(best_model.state_dict(), os.path.join(output_dir, config.model.name + ".pth"))
        logger.info("Saving epoch-%d-loss-%.4f-model to %s", best_epoch, best_eval_loss, output_dir)

    return global_step, best_epoch, best_eval_loss, best_train_loss

def normalize_config_types(config):
    try:
        # Training numeric values
        if hasattr(config, "training"):
            # float
            if hasattr(config.training, "learning_rate"):
                config.training.learning_rate = float(config.training.learning_rate)
            if hasattr(config.training, "adam_epsilon"):
                config.training.adam_epsilon = float(config.training.adam_epsilon)
            if hasattr(config.training, "warmup_proportion"):
                config.training.warmup_proportion = float(config.training.warmup_proportion)
            if hasattr(config.training, "weight_decay"):
                config.training.weight_decay = float(config.training.weight_decay)
            # int
            for k in ["batch_size", "eval_batch_size", "test_batch_size",
                      "max_seq_length", "num_epochs", "max_steps",
                      "gradient_accumulation_steps", "seed"]:
                if hasattr(config.training, k):
                    setattr(config.training, k, int(getattr(config.training, k)))
            # defaults
            if not hasattr(config.training, "max_grad_norm"):
                config.training.max_grad_norm = 1.0
            if not hasattr(config.training, "mlm_probability"):
                config.training.mlm_probability = 0.15
        # GPU values
        if hasattr(config, "gpu") and hasattr(config.gpu, "n_gpu"):
            config.gpu.n_gpu = int(config.gpu.n_gpu)
        # Logger process rank
        if hasattr(config, "log") and hasattr(config.log, "local_rank"):
            try:
                config.log.local_rank = int(config.log.local_rank)
            except Exception:
                pass
    except Exception as e:
        logging.getLogger(__name__).warning(f"Exception during config type normalization: {e}")

def main():
    config = get_config('../../Config/pretrain.yaml')
    # Normalize numeric types in config to avoid calculation/comparison errors
    normalize_config_types(config)
    logger = setup_logging(config) # Define logging format
    validate_args(config) # Validate arguments

    if config.training.seed_flag: # Set random seed for reproducibility
        set_seed(config)

    model_config = load_model_config(config) # Load model config
    tokenizer = load_tokenizer(config) # Load tokenizer

    model = initialize_model(config, model_config, logger) # Initialize model

    results = train_and_evaluate(config, model, tokenizer, model_config, logger) # Train and evaluate; model contains internal head so no external classifier needed

if __name__ == '__main__':
    main()