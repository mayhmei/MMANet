import logging
import os
import time
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn import metrics

class Logger:
    def __init__(self, config):
        self.config = config
        self.start_time = time.time()  # Record start time
        print("Initializing logging system...")
        self.setup_logging()
        print(f"Log path: {config.path.log_path}")
        self.writer = SummaryWriter(log_dir=config.path.log_path)
        print("Logging system initialized!")
        
    def setup_logging(self):
        """Configure the logger."""
        # Create log directory
        print(f"Creating log directory: {self.config.path.print_path}")
        os.makedirs(os.path.dirname(self.config.path.print_path), exist_ok=True)
        
        # Clear previous logging handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Configure logging format and handlers
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.path.print_path, encoding='utf-8'),
                logging.StreamHandler()
            ],
            force=True
        )
        
        # Smoke test for logging
        logging.info("Logging system test — if you see this message, logging is working correctly!")
        
    def log_config(self):
        """Record configuration information."""
        logging.info("=" * 60)
        logging.info("🚀 MMNet experiment started")
        logging.info("=" * 60)
        logging.info("📊 Experiment configuration:")
        logging.info(f"   Dataset: {self.config.data.dataset}")
        logging.info(f"   Mode: {self.config.model.mode}")
        logging.info(f"   Batch size: {self.config.training.batch_size}")
        logging.info(f"   Learning rate: {self.config.training.learning_rate}")
        logging.info(f"   Epochs: {self.config.training.epoch}")
        logging.info(f"   Device: {self.config.device}")
        logging.info(f"   Packet count: {self.config.data.pad_num}")
        logging.info(f"   Packet length: {self.config.data.pad_len}")
        if hasattr(self.config.model, 'num_classes'):
            logging.info(f"   Number of classes: {self.config.model.num_classes}")
        logging.info("=" * 60)
        
    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, learning_rate, epoch_time=None):
        """Record per-epoch training information."""
        time_str = f", elapsed: {epoch_time:.2f}s" if epoch_time else ""
        logging.info(f"📈 Epoch {epoch + 1}/{self.config.training.epoch}{time_str}")
        
        if self.config.model.mode == 'pretrain':
            logging.info(f"   Train - Contrastive loss: {train_loss:.4f}")
            logging.info(f"   Val - Contrastive loss: {val_loss:.4f}")
        elif self.config.model.mode == 'finetune':
            logging.info(f"   Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            logging.info(f"   Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        logging.info(f"   Learning rate: {learning_rate:.6f}")
        
        # Write to TensorBoard
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/val', val_loss, epoch)
        if self.config.model.mode == 'finetune':
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
        self.writer.add_scalar('Learning_rate', learning_rate, epoch)
        
    def log_batch(self, batch_idx, total_batches, loss, acc=None):
        """Record per-batch training information."""
        if batch_idx % 20 == 0:  # Log once every 20 batches
            progress = (batch_idx + 1) / total_batches * 100
            progress_bar = "█" * int(progress // 5) + "░" * (20 - int(progress // 5))
            
            if self.config.model.mode == 'pretrain':
                logging.info(f"   [{progress_bar}] {progress:.1f}% - Batch {batch_idx + 1}/{total_batches} - Contrastive loss: {loss:.4f}")
            elif self.config.model.mode == 'finetune' and acc is not None:
                logging.info(f"   [{progress_bar}] {progress:.1f}% - Batch {batch_idx + 1}/{total_batches} - Loss: {loss:.4f}, Acc: {acc:.4f}")
            
    def log_model_save(self, path, improvement_info=""):
        """Record model save information."""
        logging.info(f"💾 Model saved: {path} {improvement_info}")
        
    def log_early_stopping(self, patience, no_improve_batches):
        """Record early stopping information."""
        logging.info(f"⏹️  Early stopping triggered: {no_improve_batches} batches without improvement (patience: {patience})")
        
    def log_test_results(self, test_acc, test_loss, test_f1, test_precision, test_recall, confusion_matrix, classification_report):
        """Record test results."""
        logging.info("\n" + "=" * 50)
        logging.info("🎯 Test results:")
        logging.info(f"   Accuracy: {test_acc:.4f}")
        logging.info(f"   Loss: {test_loss:.4f}")
        logging.info(f"   F1 score: {test_f1:.4f}")
        logging.info(f"   Precision: {test_precision:.4f}")
        logging.info(f"   Recall: {test_recall:.4f}")
        logging.info("\n📊 Confusion matrix:")
        logging.info(confusion_matrix)
        logging.info("\n📋 Classification report:")
        logging.info(classification_report)
        logging.info("=" * 50)
        
        # Also write to TensorBoard (record once with global_step=0)
        self.writer.add_scalar('Accuracy/test', test_acc, 0)
        self.writer.add_scalar('Loss/test', test_loss, 0)
        self.writer.add_scalar('F1/test', test_f1, 0)
        self.writer.add_scalar('Precision/test', test_precision, 0)
        self.writer.add_scalar('Recall/test', test_recall, 0)
        
    def log_final_results(self, mode):
        """Record final results."""
        total_time = time.time() - self.start_time
        logging.info("\n" + "=" * 60)
        if mode == 'pretrain':
            logging.info("🏆 Contrastive pretraining complete!")
        elif mode == 'finetune':
            logging.info("🏆 Finetuning for classification complete!")
        logging.info(f"   Total training time: {total_time/60:.2f} minutes")
        logging.info("🎉 Experiment finished!")
        logging.info("=" * 60)
        
    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()