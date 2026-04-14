import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
import yaml
import time
from transformers import BertTokenizer
from data_utils import TrafficDataset, stratified_shuffle_preserve_blocks
import sys
import sklearn.metrics as metrics
from collections import Counter
import swanlab
sys.path.append('..')
from prepare_data import prepare_data
from sklearn.model_selection import train_test_split
from utils import Logger
from torch.profiler import profile, ProfilerActivity

from model_mmnet import MMNet

class ConfigObj:
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                v = ConfigObj(v)
            setattr(self, k, v)

def get_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = ConfigObj(yaml.safe_load(f))
        
        config.path.train_path = config.path.train_path.format(dataset=config.data.dataset)
        config.path.test_path = config.path.test_path.format(dataset=config.data.dataset)
        config.path.vocab_path = config.path.vocab_path.format(dataset=config.data.dataset)
        config.path.pretrain_path = config.path.pretrain_path.format(dataset=config.data.dataset)
        config.path.label_path = config.path.label_path.format(dataset=config.data.dataset)
        config.path.finetune_train_path = config.path.finetune_train_path.format(dataset=config.data.dataset)
        config.path.finetune_test_path = config.path.finetune_test_path.format(dataset=config.data.dataset)
        config.path.save_path = config.path.save_path.format(dataset=config.data.dataset)
        config.model.name = config.model.name.format(dataset=config.data.dataset)
        with open(config.path.label_path, 'r', encoding='utf-8') as f:
            config.data.class_list = [line.strip() for line in f if line.strip()]

    config.device = torch.device(config.device.device if torch.cuda.is_available() else 'cpu')
    
    config.tokenizer = BertTokenizer(
        vocab_file=config.path.vocab_path,
        max_seq_length=config.data.pad_len - 2,
        model_max_length=config.data.pad_len
    )
    
    run_name = f"{config.data.dataset}_{config.model.mode}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    
    run_output_dir = os.path.join(config.path.save_path, run_name)
    config.path.run_output_dir = run_output_dir
    
    config.path.print_path = os.path.join(run_output_dir, "console_output.txt")
    config.path.loss_path = os.path.join(run_output_dir, "loss_record.txt")
    config.path.model_save_dir = os.path.join(run_output_dir, "checkpoints")
    config.path.log_path = os.path.join(run_output_dir, "tensorboard_logs")

    os.makedirs(config.path.run_output_dir, exist_ok=True)
    os.makedirs(config.path.model_save_dir, exist_ok=True)
    os.makedirs(config.path.log_path, exist_ok=True)
    
    return config


def evaluate(config, model, data_loader, test=False):
    model.eval()
    loss_total = 0
    
    if config.model.mode == 'pretrain':
        with torch.no_grad():
            for inputs, labels in data_loader:
                contrastive_loss = model.contrastive_step(inputs, alpha=getattr(config.model, 'alpha', 0.4), update=False)
                loss_total += contrastive_loss.item()
        
        avg_loss = loss_total / len(data_loader)
        return avg_loss
    
    elif config.model.mode == 'finetune':
        labels_all = torch.tensor([], dtype=torch.long, device='cpu')
        predict_all = torch.tensor([], dtype=torch.long, device='cpu')
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                labels = labels.to(next(model.parameters()).device)
                logits = model(inputs)
                
                if isinstance(logits, tuple):
                    logits = logits[0]
                
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits, labels)
                loss_total += loss.item()
                
                predic = torch.max(logits.detach(), 1)[1].cpu()
                true_labels = labels.squeeze().cpu()
                
                labels_all = torch.cat([labels_all, true_labels])
                predict_all = torch.cat([predict_all, predic])
        
        acc = metrics.accuracy_score(labels_all, predict_all)
        
        if test:
            f1 = metrics.f1_score(labels_all, predict_all, average='weighted', zero_division=0)
            precision = metrics.precision_score(labels_all, predict_all, average='weighted', zero_division=0)
            recall = metrics.recall_score(labels_all, predict_all, average='weighted', zero_division=0)
            report = metrics.classification_report(
                labels_all, predict_all,
                labels=list(range(len(config.data.class_list))),
                target_names=config.data.class_list,
                digits=4,
                zero_division=0
            )
            confusion = metrics.confusion_matrix(labels_all, predict_all)
            y_true = labels_all.numpy()
            y_pred = predict_all.numpy()
            return acc, loss_total / len(data_loader), f1, precision, recall, confusion, report, y_true, y_pred
        
        return acc, loss_total / len(data_loader)

def loss_contrastive(view1, view2, temperature=0.7):
    batch_size = view1.shape[0]
    
    view1 = F.normalize(view1, p=2, dim=1)
    view2 = F.normalize(view2, p=2, dim=1)
    
    similarity_matrix = torch.matmul(view1, view2.T) / temperature
    
    labels = torch.arange(batch_size, device=view1.device)
    
    loss_1 = F.cross_entropy(similarity_matrix, labels, reduction='mean')
    loss_2 = F.cross_entropy(similarity_matrix.T, labels, reduction='mean')
    
    loss = (loss_1 + loss_2) / 2
    return loss

def analyze_mmnet_model(model, config, train_loader, logger=None,
                        buffer_ratio=0.3, do_backward=True, warmup=2, iters=5):
    log = logger.info if logger else print
    device = next(model.parameters()).device
    mode = config.model.mode

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    module_params = {}
    for name, child in model.named_children():
        module_params[name] = sum(p.numel() for p in child.parameters())

    params_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    is_fp16_like = any(p.dtype in (torch.float16, torch.bfloat16) for p in model.parameters())
    bytes_per_param_train_upper = 20 if is_fp16_like else 16
    param_related_train_bytes_upper = total_params * bytes_per_param_train_upper

    dummy_batch = next(iter(train_loader))
    inputs, labels = dummy_batch

    def to_device(x, dev):
        if torch.is_tensor(x):
            return x.to(dev)
        if isinstance(x, (list, tuple)):
            return type(x)(to_device(i, dev) for i in x)
        if isinstance(x, dict):
            return {k: to_device(v, dev) for k, v in x.items()}
        return x

    inputs = to_device(inputs, device)
    labels = to_device(labels, device)

    def run_forward():
        if mode == "pretrain":
            loss = model.contrastive_step(inputs, alpha=getattr(config.model, "alpha", 0.4), update=False)
            return loss
        else:
            logits = model(inputs)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = F.cross_entropy(logits, labels.long().view(-1))
            return loss

    prof_fwd_flops = None
    prof_train_flops = None
    if device.type == "cuda":
        try:
            from torch.profiler import profile, ProfilerActivity

            torch.cuda.synchronize()
            model.eval()
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
                with_flops=True,
                profile_memory=False,
            ) as prof:
                with torch.no_grad():
                    _ = run_forward()
                torch.cuda.synchronize()

            fwd = 0
            for e in prof.key_averages():
                if hasattr(e, "flops") and e.flops is not None:
                    fwd += e.flops
            prof_fwd_flops = float(fwd) if fwd > 0 else None

            if do_backward:
                torch.cuda.synchronize()
                model.train()
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=False,
                    with_flops=True,
                    profile_memory=False,
                ) as prof2:
                    loss = run_forward()
                    loss.backward()
                    model.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()

                tr = 0
                for e in prof2.key_averages():
                    if hasattr(e, "flops") and e.flops is not None:
                        tr += e.flops
                prof_train_flops = float(tr) if tr > 0 else None

        except Exception as e:
            log(f"[analyze_mmnet_model] Profiler unavailable, skipping FLOPs: {e}")

    step_time_ms = None
    peak_alloc_mb = None
    if device.type == "cuda":
        try:
            model.train()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            for _ in range(warmup):
                loss = run_forward()
                loss.backward()
                model.zero_grad(set_to_none=True)

            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(iters):
                loss = run_forward()
                loss.backward()
                model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            t1 = time.time()

            step_time_ms = (t1 - t0) / iters * 1000.0
            peak_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        except Exception as e:
            log(f"[analyze_mmnet_model] Micro-benchmark failed: {e}")

    def mb(x): return x / (1024 ** 2)
    def gib(x): return x / (1024 ** 3)

    if peak_alloc_mb is not None:
        recommended_gib = (peak_alloc_mb / 1024.0) * (1.0 + buffer_ratio)
    else:
        recommended_gib = gib(param_related_train_bytes_upper) * (1.0 + buffer_ratio)

    log("=" * 80)
    log("MMNet Model Resource Analysis")
    log("=" * 80)
    log(f"• mode: {mode}")
    log(f"• device: {device}")

    log("\nParameters")
    log(f"  • Total: {total_params:,} ({total_params/1e6:.3f} M)")
    log(f"  • Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    log("  • Modules:")
    for k, v in module_params.items():
        log(f"    - {k}: {v:,} ({v/1e6:.3f} M)")

    log("\nMemory (Reference)")
    log(f"  • Model Weights: {mb(params_bytes):.1f} MB ({gib(params_bytes):.2f} GiB)")
    log(f"  • Training Upper Bound: {mb(param_related_train_bytes_upper):.1f} MB ({gib(param_related_train_bytes_upper):.2f} GiB)")

    log("\nFLOPs (Profiler)")
    log(f"  • Forward FLOPs: {prof_fwd_flops:.3e}" if prof_fwd_flops else "  • Forward FLOPs: N/A")
    log(f"  • Train Step FLOPs: {prof_train_flops:.3e}" if prof_train_flops else "  • Train Step FLOPs: N/A")

    if peak_alloc_mb is not None:
        log("\nMicro-benchmark")
        log(f"  • Step Time: {step_time_ms:.2f} ms")
        log(f"  • Peak Allocated: {peak_alloc_mb:.1f} MB ({peak_alloc_mb/1024:.2f} GiB)")

    log(f"\nRecommended GPU Memory: ≈ {recommended_gib:.2f} GiB")
    log("=" * 80)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "module_params": module_params,
        "params_gib": gib(params_bytes),
        "param_related_train_gib_upper": gib(param_related_train_bytes_upper),
        "prof_fwd_flops": prof_fwd_flops,
        "prof_train_flops": prof_train_flops,
        "peak_alloc_gib": (peak_alloc_mb / 1024.0) if peak_alloc_mb is not None else None,
        "recommended_gib": recommended_gib,
    }


def train(config, model, train_loader, dev_loader, logger):
    model.train()
    
    if config.model.mode == 'pretrain':
        logging.info("Starting Pretraining - Contrastive Learning")
    elif config.model.mode == 'finetune':
        logging.info("Starting Finetuning - Classification Task")
        if hasattr(config.training, 'freeze_feature_extractors') and config.training.freeze_feature_extractors:
            logging.info("Feature extractors frozen")
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=0.001
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.992)
    
    total_batch = 0
    last_improve = 0
    flag = False
    dev_best_loss = float('inf')
    dev_best_acc = 0.0
    
    if config.model.mode == 'pretrain':
        best_model_path = os.path.join(config.path.model_save_dir, "best_pretrain_model.pth")
    elif config.model.mode == 'finetune':
        best_model_path = os.path.join(config.path.model_save_dir, "best_finetune_model.pth")
    
    for epoch in range(config.training.epoch):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        train_acc = 0
        total_train = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            model.zero_grad()
            labels = labels.to(next(model.parameters()).device)
            
            if config.model.mode == 'pretrain':
                loss = model.contrastive_step(inputs, alpha=getattr(config.model, 'alpha', 0.4), update=True)
                batch_acc = 0
            elif config.model.mode == 'finetune':
                logits = model(inputs)
                loss_fn = nn.CrossEntropyLoss()
                labels = labels.to(logits.device)
                if labels.dim() > 1:
                    labels = labels.squeeze(-1)
                labels = labels.long()
                classification_loss = loss_fn(logits, labels)
                loss = classification_loss
                
                true_labels = labels.squeeze().detach().cpu()
                predic = torch.max(logits.detach(), 1)[1].cpu()
                batch_acc = metrics.accuracy_score(true_labels, predic)
                train_acc += batch_acc * len(true_labels)
                total_train += len(true_labels)
                
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if i % 10 == 0:
                logger.log_batch(i, len(train_loader), loss.item(), batch_acc if config.model.mode == 'finetune' else None)
            
            if total_batch % 100 == 0:
                if config.model.mode == 'pretrain':
                    dev_loss = evaluate(config, model, dev_loader)
                    if dev_loss < dev_best_loss:
                        dev_best_loss = dev_loss
                        torch.save(model.state_dict(), best_model_path)
                        last_improve = total_batch
                        logger.log_model_save(best_model_path, f"Loss: {dev_loss:.4f}")
                elif config.model.mode == 'finetune':
                    dev_acc, dev_loss, dev_f1, dev_precision, dev_recall, _, _, _, _ = evaluate(
                        config, model, dev_loader, test=True
                    )
                    if dev_acc > dev_best_acc:
                        dev_best_acc = dev_acc
                        dev_best_loss = dev_loss
                        torch.save(model.state_dict(), best_model_path)
                        last_improve = total_batch
                        logger.log_model_save(best_model_path, f"Acc: {dev_acc:.4f}")
                model.train()
            
            total_batch += 1
            if total_batch - last_improve > config.training.require_improvement:
                logger.log_early_stopping(config.training.require_improvement, total_batch - last_improve)
                flag = True
                break
        
        epoch_time = time.time() - epoch_start_time
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / total_train if config.model.mode == 'finetune' and total_train > 0 else 0
        
        if config.model.mode == 'pretrain':
            if total_batch % 100 != 0:
                dev_loss = evaluate(config, model, dev_loader)
            logger.log_epoch(epoch, avg_train_loss, 0, dev_loss, 0, 
                           optimizer.param_groups[0]['lr'], epoch_time)
            
            swanlab.log({
                "loss/avg_train_loss": avg_train_loss,
                "loss/avg_dev_loss": dev_loss,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch_time": epoch_time
            })
            
        elif config.model.mode == 'finetune':
            if total_batch % 100 != 0:
                dev_acc, dev_loss, dev_f1, dev_precision, dev_recall, _, _, _, _ = evaluate(
                    config, model, dev_loader, test=True
                )
            logger.log_epoch(epoch, avg_train_loss, avg_train_acc, dev_loss, dev_acc, 
                           optimizer.param_groups[0]['lr'], epoch_time)
            
            swanlab.log({
                "loss/avg_train_loss": avg_train_loss,
                "accuracy/avg_train_accuracy": avg_train_acc,
                "loss/avg_dev_loss": dev_loss,
                "accuracy/avg_dev_accuracy": dev_acc,
                "f1/avg_dev_f1": dev_f1,
                "precision/avg_dev_precision": dev_precision,
                "recall/avg_dev_recall": dev_recall,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch_time": epoch_time
            })
        
        if optimizer.param_groups[0]['lr'] > 1e-5:
            scheduler.step()
        
        if flag:
            break
    
    model.load_state_dict(torch.load(best_model_path))
    return model

def split_train_dev(data, dev_ratio=0.1, random_state=42):
    labels = [item[-1] for item in data]
    
    train_data, dev_data = train_test_split(
        data, 
        test_size=dev_ratio, 
        random_state=random_state,
        stratify=labels
    )
    
    train_labels = [item[-1] for item in train_data]
    dev_labels = [item[-1] for item in dev_data]
    
    train_dist = Counter(train_labels)
    dev_dist = Counter(dev_labels)
    
    logging.info(f"Data Split Complete:")
    logging.info(f"Train Size: {len(train_data)} ({len(train_data)/len(data)*100:.1f}%)")
    logging.info(f"Dev Size: {len(dev_data)} ({len(dev_data)/len(data)*100:.1f}%)")
    logging.info(f"Train Distribution: {dict(train_dist)}")
    logging.info(f"Dev Distribution: {dict(dev_dist)}")
    
    return train_data, dev_data

def main():
    config = get_config('../../Config/inter_packet_pretrain.yaml')
    
    swanlab.init(
        project="MMNet-constrative-ALBEF",
        experiment_name=f"ustc_{config.model.mode}_training_{time.strftime('%Y%m%d_%H%M%S')}",
        config={
            "mode": config.model.mode,
            "batch_size": config.training.batch_size,
            "learning_rate": config.training.learning_rate,
            "epochs": config.training.epoch,
            "temperature": config.model.temperature if hasattr(config.model, 'temperature') else 0.7,
            "model_name": "MMNet"
        }
    )
    
    logger = Logger(config)
    logger.log_config()
    
    if config.model.mode == 'finetune':
        train_data = prepare_data(config, config.path.finetune_train_path)
        test_data = prepare_data(config, config.path.finetune_test_path)
    elif config.model.mode == 'pretrain':
        train_data = prepare_data(config, config.path.train_path)
    
    if config.model.shuffle_data:
        train_data = stratified_shuffle_preserve_blocks(train_data)
        if config.model.mode == 'finetune':
            test_data = stratified_shuffle_preserve_blocks(test_data)
    
    train_data, dev_data = split_train_dev(train_data, dev_ratio=0.1)

    train_dataset = TrafficDataset(train_data)
    dev_dataset = TrafficDataset(dev_data)
    if config.model.mode == 'finetune':
        test_dataset = TrafficDataset(test_data)
        test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False, drop_last=(config.model.mode == 'pretrain'))
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, drop_last=(config.model.mode == 'pretrain'))
    dev_loader = DataLoader(dev_dataset, batch_size=config.training.batch_size, shuffle=False, drop_last=(config.model.mode == 'pretrain'))
    
    
    model = MMNet(config).to(config.device)
    
    if config.model.mode == 'finetune' and hasattr(config.path, 'pretrained_model_path'):
        logging.info(f"Loading pretrained weights from {config.path.pretrained_model_path}")
        model.load_pretrained_weights(config.path.pretrained_model_path)
        
        if hasattr(config.training, 'freeze_feature_extractors') and config.training.freeze_feature_extractors:
            model.freeze_feature_extractors()
            logging.info("Feature extractors frozen")
    
    analyze_mmnet_model(
        model=model,
        config=config,
        train_loader=train_loader,
        logger=logging.getLogger(__name__),
        buffer_ratio=0.3,
        do_backward=True,
        warmup=2,
        iters=5,
    )

    model = train(config, model, train_loader, dev_loader, logger)
    
    if config.model.mode == 'finetune':
        test_acc, test_loss, test_f1, test_precision, test_recall, test_confusion, test_report, y_true, y_pred = evaluate(
            config, model, test_loader, test=True
        )
        logger.log_test_results(test_acc, test_loss, test_f1, test_precision, test_recall, test_confusion, test_report)

        swanlab.log({
                "accuracy/final_test_accuracy": test_acc,
                "loss/final_test_loss": test_loss,
                "f1/final_test_f1": test_f1,
                "precision/final_test_precision": test_precision,
                "recall/final_test_recall": test_recall
            })
        
        cm = test_confusion if test_confusion is not None else metrics.confusion_matrix(
            y_true, y_pred, labels=list(range(len(config.data.class_list)))
        )
        swanlab.log({
            "confusion_matrix": cm.tolist()
        })
    
    logger.log_final_results(config.model.mode)
    logger.close()
    swanlab.finish()

if __name__ == '__main__':
    main()
