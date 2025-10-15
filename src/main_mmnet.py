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
import swanlab  # Add SwanLab import
sys.path.append('..')
from prepare_data import prepare_data
from sklearn.model_selection import train_test_split
from utils import Logger  # Import Logger class

from model_mmnet import MMNet

class ConfigObj:
    """Configuration wrapper to access dict values as attributes."""
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                v = ConfigObj(v)
            setattr(self, k, v)

def get_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:  # Open config file
        config = ConfigObj(yaml.safe_load(f))  # Load YAML and create config object
        
        config.path.train_path = config.path.train_path.format(dataset=config.data.dataset)
        config.path.test_path = config.path.test_path.format(dataset=config.data.dataset)
        config.path.vocab_path = config.path.vocab_path.format(dataset=config.data.dataset)
        config.path.pretrain_path = config.path.pretrain_path.format(dataset=config.data.dataset)
        config.path.label_path = config.path.label_path.format(dataset=config.data.dataset)
        config.path.finetune_train_path = config.path.finetune_train_path.format(dataset=config.data.dataset)
        config.path.finetune_test_path = config.path.finetune_test_path.format(dataset=config.data.dataset)
        config.path.test_mode_path = config.path.test_mode_path.format(dataset=config.data.dataset)
        config.model.name = config.model.name.format(dataset=config.data.dataset)
        with open(config.path.label_path, 'r', encoding='utf-8') as f:  # Open label file
            config.data.class_list = [line.strip() for line in f if line.strip()]  # Read class list

    # Device handling
    config.device = torch.device(config.device.device if torch.cuda.is_available() else 'cpu')  # Set compute device
    # Initialize BERT tokenizer
    config.tokenizer = BertTokenizer(
        vocab_file=config.path.vocab_path,  # Path to vocab file
        max_seq_length=config.data.pad_len - 2,  # Max sequence length
        model_max_length=config.data.pad_len  # Model max length
    )
    # 1) Create a unique run name to avoid overwrites
    run_name = f"{config.data.dataset}_{config.model.mode}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    # 2) Create a root output directory under 'save_path' to store this run
    run_output_dir = os.path.join(config.path.save_path, run_name)
    config.path.run_output_dir = run_output_dir  # Persist for later use
    # 3) Update all output paths to this unique run directory
    config.path.print_path = os.path.join(run_output_dir, "console_output.txt")
    config.path.loss_path = os.path.join(run_output_dir, "loss_record.txt")
    config.path.model_save_dir = os.path.join(run_output_dir, "checkpoints")  # directory of checkpoints
    config.path.log_path = os.path.join(run_output_dir, "tensorboard_logs")  # TensorBoard logs path
    # 4) Create directories if missing
    os.makedirs(config.path.run_output_dir, exist_ok=True)
    os.makedirs(config.path.model_save_dir, exist_ok=True)
    os.makedirs(config.path.log_path, exist_ok=True)
    return config  # Return config object


def evaluate(config, model, data_loader, test=False):
    """Evaluation function — supports pretrain and finetune modes."""
    model.eval()
    loss_total = 0
    
    if config.model.mode == 'pretrain':
        # Pretrain mode: compute contrastive loss only
        with torch.no_grad():
            for inputs, labels in data_loader:
                # Use ALBEF-style contrastive step (updates EMA and queues)
                contrastive_loss = model.contrastive_step(inputs, alpha=getattr(config.model, 'alpha', 0.4), update=False)
                loss_total += contrastive_loss.item()
        
        avg_loss = loss_total / len(data_loader)
        return avg_loss  # Return average contrastive loss
    elif config.model.mode in ['finetune', 'test']:
        # Finetune/Test mode: compute classification metrics
        labels_all = torch.tensor([], dtype=torch.long, device='cpu')
        predict_all = torch.tensor([], dtype=torch.long, device='cpu')
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                # Move labels to correct device
                labels = labels.to(next(model.parameters()).device)
                logits = model(inputs)
                
                # Ensure output dimension correct
                if isinstance(logits, tuple):
                    logits = logits[0]
                
                # Compute loss
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits, labels)
                loss_total += loss.item()
                
                # Get prediction result
                predic = torch.max(logits.detach(), 1)[1].cpu()
                true_labels = labels.squeeze().cpu()
                
                # Statistics all samples labels and prediction values
                labels_all = torch.cat([labels_all, true_labels])
                predict_all = torch.cat([predict_all, predic])
        
        acc = metrics.accuracy_score(labels_all, predict_all)
        
        if test:  # On test or explicitly requested, return all metrics
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
            # Added: return y_true/y_pred for SwanLab confusion matrix
            y_true = labels_all.numpy()
            y_pred = predict_all.numpy()
            return acc, loss_total / len(data_loader), f1, precision, recall, confusion, report, y_true, y_pred
        
        return acc, loss_total / len(data_loader)  # Non-test returns accuracy and loss

def loss_contrastive(view1, view2, temperature=0.7):
    batch_size = view1.shape[0]
    
    # Feature normalization
    view1 = F.normalize(view1, p=2, dim=1)
    view2 = F.normalize(view2, p=2, dim=1)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(view1, view2.T) / temperature
    
    # Positive sample labels
    labels = torch.arange(batch_size, device=view1.device)
    
    # Compute contrastive loss
    loss_1 = F.cross_entropy(similarity_matrix, labels, reduction='mean')
    loss_2 = F.cross_entropy(similarity_matrix.T, labels, reduction='mean')
    
    # Debug info
    # print(f"loss_1: {loss_1.item()}, loss_2: {loss_2.item()}")
    
    # Bidirectional average
    loss = (loss_1 + loss_2) / 2
    # print(f"averaged loss: {loss.item()}")
    
    # Force scalar return
    # loss = loss.squeeze()
    # while loss.dim() > 0:
    #     loss = loss.mean()
    
    # print(f"final loss: {loss.item()}")
    return loss

def train(config, model, train_loader, dev_loader, logger):
    """Training function."""
    model.train()
    
    # Log training info based on mode
    if config.model.mode == 'pretrain':
        logging.info("Starting pretrain mode - contrastive learning")
    elif config.model.mode == 'finetune':
        logging.info("Starting finetune mode - classification task")
        if hasattr(config.training, 'freeze_feature_extractors') and config.training.freeze_feature_extractors:
            logging.info("Feature extractors are frozen; training classifier only")
    
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
    dev_best_acc = 0.0  # Track best accuracy
    
    # Set different model save paths based on mode
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
            
            # Forward pass
            if config.model.mode == 'pretrain':
                # Use model's ALBEF-style contrastive step (updates EMA and queues)
                loss = model.contrastive_step(inputs, alpha=getattr(config.model, 'alpha', 0.4), update=True)
                # No accuracy metric in pretrain mode
                batch_acc = 0
            elif config.model.mode == 'finetune':
                logits = model(inputs)
                loss_fn = nn.CrossEntropyLoss()
                # Ensure labels are on the correct device and properly shaped
                labels = labels.to(logits.device)
                if labels.dim() > 1:
                    labels = labels.squeeze(-1)
                labels = labels.long()
                classification_loss = loss_fn(logits, labels)
                loss = classification_loss
                
                # Compute training metrics
                import sklearn.metrics as metrics
                true_labels = labels.squeeze().detach().cpu()
                predic = torch.max(logits.detach(), 1)[1].cpu()
                batch_acc = metrics.accuracy_score(true_labels, predic)
                train_acc += batch_acc * len(true_labels)
                total_train += len(true_labels)
                
            loss.backward()
            # Add gradient clipping to prevent gradient explosion
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()  # Update model parameters
            
            # Accumulate training metrics
            train_loss += loss.item()
            
            # Log batch info
            if i % 10 == 0:
                logger.log_batch(i, len(train_loader), loss.item(), batch_acc if config.model.mode == 'finetune' else None)
            
            if total_batch % 100 == 0:
                if config.model.mode == 'pretrain':
                    dev_loss = evaluate(config, model, dev_loader)
                    if dev_loss < dev_best_loss:
                        dev_best_loss = dev_loss
                        torch.save(model.state_dict(), best_model_path)
                        last_improve = total_batch
                        logger.log_model_save(best_model_path, f"Contrastive loss: {dev_loss:.4f}")
                elif config.model.mode == 'finetune':
                    # Retrieve dev F1 / Precision / Recall during evaluation
                    dev_acc, dev_loss, dev_f1, dev_precision, dev_recall, _, _, _, _ = evaluate(
                        config, model, dev_loader, test=True
                    )
                    if dev_acc > dev_best_acc:  # Use accuracy as criterion in finetune mode
                        dev_best_acc = dev_acc
                        dev_best_loss = dev_loss
                        torch.save(model.state_dict(), best_model_path)
                        last_improve = total_batch
                        logger.log_model_save(best_model_path, f"Accuracy: {dev_acc:.4f}")
                model.train()
            
            total_batch += 1
            if total_batch - last_improve > config.training.require_improvement:
                logger.log_early_stopping(config.training.require_improvement, total_batch - last_improve)
                flag = True
                break
        
        # Compute epoch-level averages
        epoch_time = time.time() - epoch_start_time
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / total_train if config.model.mode == 'finetune' and total_train > 0 else 0
        
        # Evaluate at epoch end (avoid repeated calls)
        if config.model.mode == 'pretrain':
            # Re-evaluate only if not evaluated in the last 100 batches of the current epoch
            if total_batch % 100 != 0:
                dev_loss = evaluate(config, model, dev_loader)
            logger.log_epoch(epoch, avg_train_loss, 0, dev_loss, 0, 
                           optimizer.param_groups[0]['lr'], epoch_time)
            
            # Log pretrain epoch metrics to SwanLab
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
            
            # Log dev F1 / Precision / Recall to SwanLab
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
    
    # Load best model weights after training
    model.load_state_dict(torch.load(best_model_path))
    return model

def run_test(config, logger):
    """Standalone test evaluation function to avoid changing main training flow."""
    test_data = prepare_data(config, config.path.test_mode_path)
    if config.model.shuffle_data:
        test_data = stratified_shuffle_preserve_blocks(test_data)
    test_loader = DataLoader(
        TrafficDataset(test_data),
        batch_size=config.training.batch_size,
        shuffle=True,           # Shuffle during test as well
        drop_last=False
    )
    model = MMNet(config).to(config.device)
    model.mode = 'finetune'  # Use classification forward during test; avoid Unknown mode: test
    if not (hasattr(config.path, 'finetune_model_path') and os.path.exists(config.path.finetune_model_path)):
        raise FileNotFoundError(f"Finetuned weights to evaluate not found: {getattr(config.path, 'finetune_model_path', None)}")
    model.load_state_dict(torch.load(config.path.finetune_model_path, map_location=config.device))

    test_acc, test_loss, test_f1, test_precision, test_recall, test_confusion, test_report, y_true, y_pred = evaluate(
        config, model, test_loader, test=True
    )
    logger.log_test_results(test_acc, test_loss, test_f1, test_precision, test_recall, test_confusion, test_report)

    # Log final test metrics to SwanLab
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
    swanlab.log({"confusion_matrix": cm.tolist()})

def split_train_dev(data, dev_ratio=0.1, random_state=42):
    """
    Split the training data into train and dev sets.

    Args:
        data: original training data list
        dev_ratio: validation ratio (default 0.1, 10%)
        random_state: random seed for reproducibility

    Returns:
        train_data: training subset
        dev_data: validation subset
    """
    # Extract labels for stratified sampling
    labels = [item[-1] for item in data]
    
    # Use stratified sampling to preserve class distribution
    train_data, dev_data = train_test_split(
        data, 
        test_size=dev_ratio, 
        random_state=random_state,
        stratify=labels  # Stratified sampling to preserve class distribution
    )
    
    # Print dataset info after split
    train_labels = [item[-1] for item in train_data]
    dev_labels = [item[-1] for item in dev_data]
    
    train_dist = Counter(train_labels)
    dev_dist = Counter(dev_labels)
    
    logging.info(f"Data split completed:")
    logging.info(f"Train set size: {len(train_data)} ({len(train_data)/len(data)*100:.1f}%)")
    logging.info(f"Dev set size: {len(dev_data)} ({len(dev_data)/len(data)*100:.1f}%)")
    logging.info(f"Train class distribution: {dict(train_dist)}")
    logging.info(f"Dev class distribution: {dict(dev_dist)}")
    
    return train_data, dev_data

def main():
    # Load configuration
    config = get_config('../../Config/contrastive_train.yaml')
    
    # Initialize SwanLab
    swanlab.init(
        project="MMNet-constrative-ALBEF",
        experiment_name=f"vpn_pcz_multi_shift_2_to_1_{config.model.mode}_training_{time.strftime('%Y%m%d_%H%M%S')}",
        config={
            "mode": config.model.mode,
            "batch_size": config.training.batch_size,
            "learning_rate": config.training.learning_rate,
            "epochs": config.training.epoch,
            "temperature": config.model.temperature if hasattr(config.model, 'temperature') else 0.7,
            "model_name": "MMNet"
        }
    )
    
    # Initialize logger
    logger = Logger(config)
    logger.log_config()
    
    # Test mode: call standalone test function and exit early
    if config.model.mode == 'test':
        run_test(config, logger)
        logger.log_final_results('test')
        logger.close()
        swanlab.finish()
        return

    if config.model.mode == 'finetune':
        train_data = prepare_data(config, config.path.finetune_train_path)
        test_data = prepare_data(config, config.path.finetune_test_path)
    elif config.model.mode == 'pretrain':
        train_data = prepare_data(config, config.path.train_path)
        # test_data = prepare_data(config, config.path.test_path)

    # Shuffle data
    if config.model.shuffle_data:
        train_data = stratified_shuffle_preserve_blocks(train_data)
        if config.model.mode == 'finetune':
            test_data = stratified_shuffle_preserve_blocks(test_data)
    
    # Split training data into train and dev (9:1)
    train_data, dev_data = split_train_dev(train_data, dev_ratio=0.1)

    train_dataset = TrafficDataset(train_data)
    dev_dataset = TrafficDataset(dev_data)
    if config.model.mode == 'finetune':
        test_dataset = TrafficDataset(test_data)
        test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False, drop_last=(config.model.mode == 'pretrain'))
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, drop_last=(config.model.mode == 'pretrain'))
    dev_loader = DataLoader(dev_dataset, batch_size=config.training.batch_size, shuffle=False, drop_last=(config.model.mode == 'pretrain'))
    
    
    # Create model
    model = MMNet(config).to(config.device)
    
    # If finetune mode, load pretrained weights
    if config.model.mode == 'finetune' and hasattr(config.path, 'pretrained_model_path'):
        logging.info(f"Loading pretrained weights for finetuning from {config.path.pretrained_model_path}")
        model.load_pretrained_weights(config.path.pretrained_model_path)
        
        # Freeze feature extractors if configured
        if hasattr(config.training, 'freeze_feature_extractors') and config.training.freeze_feature_extractors:
            model.freeze_feature_extractors()
            logging.info("Feature extractors frozen, only training classifier")

    # Train model
    model = train(config, model, train_loader, dev_loader, logger)
    
    # Test model
    if config.model.mode == 'finetune':
        # New signature: receive precision, recall, y_true, y_pred
        test_acc, test_loss, test_f1, test_precision, test_recall, test_confusion, test_report, y_true, y_pred = evaluate(
            config, model, test_loader, test=True
        )
        # Update Logger invocation
        logger.log_test_results(test_acc, test_loss, test_f1, test_precision, test_recall, test_confusion, test_report)

        # Log final test metrics to SwanLab (including precision/recall)
        swanlab.log({
                "accuracy/final_test_accuracy": test_acc,
                "loss/final_test_loss": test_loss,
                "f1/final_test_f1": test_f1,
                "precision/final_test_precision": test_precision,
                "recall/final_test_recall": test_recall
            })
        
        # Upload confusion matrix to SwanLab (no longer calling nonexistent swanlab.confusion_matrix)
        # Prefer evaluate's test_confusion; otherwise compute with sklearn
        cm = test_confusion if test_confusion is not None else metrics.confusion_matrix(
            y_true, y_pred, labels=list(range(len(config.data.class_list)))
        )
        swanlab.log({
            "confusion_matrix": cm.tolist()
        })
    # Log final results
    logger.log_final_results(config.model.mode)
    logger.close()
    swanlab.finish()  # Finish SwanLab logging

if __name__ == '__main__':
    main()