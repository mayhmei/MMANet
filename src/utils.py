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
        self.start_time = time.time()  # 记录开始时间
        print("正在初始化日志系统...")
        self.setup_logging()
        print(f"日志路径: {config.path.log_path}")
        self.writer = SummaryWriter(log_dir=config.path.log_path)
        print("日志系统初始化完成!")
        
    def setup_logging(self):
        """设置日志记录器"""
        # 创建日志目录
        print(f"创建日志目录: {self.config.path.print_path}")
        os.makedirs(os.path.dirname(self.config.path.print_path), exist_ok=True)
        
        # 清除之前的日志配置
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # 配置日志格式
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.path.print_path, encoding='utf-8'),
                logging.StreamHandler()
            ],
            force=True
        )
        
        # 测试日志是否工作
        logging.info("日志系统测试 - 如果您看到这条消息，说明日志系统工作正常!")
        
    def log_config(self):
        """记录配置信息"""
        logging.info("=" * 60)
        logging.info("🚀 MMNet 实验开始")
        logging.info("=" * 60)
        logging.info("📊 实验配置信息:")
        logging.info(f"   数据集: {self.config.data.dataset}")
        logging.info(f"   模式: {self.config.model.mode}")
        logging.info(f"   批次大小: {self.config.training.batch_size}")
        logging.info(f"   学习率: {self.config.training.learning_rate}")
        logging.info(f"   训练轮数: {self.config.training.epoch}")
        logging.info(f"   设备: {self.config.device}")
        logging.info(f"   数据包数量: {self.config.data.pad_num}")
        logging.info(f"   数据包长度: {self.config.data.pad_len}")
        if hasattr(self.config.model, 'num_classes'):
            logging.info(f"   类别数量: {self.config.model.num_classes}")
        logging.info("=" * 60)
        
    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, learning_rate, epoch_time=None):
        """记录每个epoch的训练信息"""
        time_str = f", 用时: {epoch_time:.2f}s" if epoch_time else ""
        logging.info(f"📈 Epoch {epoch + 1}/{self.config.training.epoch}{time_str}")
        
        if self.config.model.mode == 'pretrain':
            logging.info(f"   训练 - 对比损失: {train_loss:.4f}")
            logging.info(f"   验证 - 对比损失: {val_loss:.4f}")
        elif self.config.model.mode == 'finetune':
            logging.info(f"   训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            logging.info(f"   验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        logging.info(f"   学习率: {learning_rate:.6f}")
        
        # 记录到TensorBoard
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/val', val_loss, epoch)
        if self.config.model.mode == 'finetune':
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
        self.writer.add_scalar('Learning_rate', learning_rate, epoch)
        
    def log_batch(self, batch_idx, total_batches, loss, acc=None):
        """记录每个batch的训练信息"""
        if batch_idx % 20 == 0:  # 每20个batch记录一次
            progress = (batch_idx + 1) / total_batches * 100
            progress_bar = "█" * int(progress // 5) + "░" * (20 - int(progress // 5))
            
            if self.config.model.mode == 'pretrain':
                logging.info(f"   [{progress_bar}] {progress:.1f}% - Batch {batch_idx + 1}/{total_batches} - 对比损失: {loss:.4f}")
            elif self.config.model.mode == 'finetune' and acc is not None:
                logging.info(f"   [{progress_bar}] {progress:.1f}% - Batch {batch_idx + 1}/{total_batches} - Loss: {loss:.4f}, Acc: {acc:.4f}")
            
    def log_model_save(self, path, improvement_info=""):
        """记录模型保存信息"""
        logging.info(f"💾 模型已保存: {path} {improvement_info}")
        
    def log_early_stopping(self, patience, no_improve_batches):
        """记录早停信息"""
        logging.info(f"⏹️  早停触发: {no_improve_batches} 个batch无改善 (耐心值: {patience})")
        
    def log_test_results(self, test_acc, test_loss, test_f1, test_precision, test_recall, confusion_matrix, classification_report):
        """记录测试结果"""
        logging.info("\n" + "=" * 50)
        logging.info("🎯 测试结果:")
        logging.info(f"   准确率: {test_acc:.4f}")
        logging.info(f"   损失值: {test_loss:.4f}")
        logging.info(f"   F1分数: {test_f1:.4f}")
        logging.info(f"   精确率(Precision): {test_precision:.4f}")
        logging.info(f"   召回率(Recall): {test_recall:.4f}")
        logging.info("\n📊 混淆矩阵:")
        logging.info(confusion_matrix)
        logging.info("\n📋 分类报告:")
        logging.info(classification_report)
        logging.info("=" * 50)
        
        # 同时写入 TensorBoard（仅记录一次，global_step=0）
        self.writer.add_scalar('Accuracy/test', test_acc, 0)
        self.writer.add_scalar('Loss/test', test_loss, 0)
        self.writer.add_scalar('F1/test', test_f1, 0)
        self.writer.add_scalar('Precision/test', test_precision, 0)
        self.writer.add_scalar('Recall/test', test_recall, 0)
        
    def log_final_results(self, mode):
        """记录最终结果"""
        total_time = time.time() - self.start_time
        logging.info("\n" + "=" * 60)
        if mode == 'pretrain':
            logging.info("🏆 对比学习预训练完成!")
        elif mode == 'finetune':
            logging.info("🏆 分类任务微调完成!")
        logging.info(f"   总训练时间: {total_time/60:.2f} 分钟")
        logging.info("🎉 实验完成!")
        logging.info("=" * 60)
        
    def close(self):
        """关闭TensorBoard写入器"""
        self.writer.close()