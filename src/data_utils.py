import os
import torch
from torch.utils.data import Dataset
import numpy as np
import sys
import pickle
from sklearn.model_selection import train_test_split
import random

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prepare_data import prepare_data

def load_labels(label_path):
    """Load the label list."""
    with open(label_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def stratified_shuffle_preserve_blocks(data):
    """Shuffle by class while preserving sample order within each class."""
    # Group by class
    class_groups = {}
    for item in data:
        label = item[-1]
        if label not in class_groups:
            class_groups[label] = []
        class_groups[label].append(item)
    
    # Shuffle samples within each class
    for label in class_groups:
        random.shuffle(class_groups[label])
    
    # Reassemble data
    shuffled_data = []
    max_samples = max(len(samples) for samples in class_groups.values())
    
    for i in range(max_samples):
        for label in sorted(class_groups.keys()):
            samples = class_groups[label]
            if i < len(samples):
                shuffled_data.append(samples[i])
    
    return shuffled_data

class TrafficDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, index):
        (packet_time_interval, 
        packet_payload_length,
        packet_direction,
        packet_sequence_burst_indices,
        packet_sequence_mask, 
        packet_sequence_attention_mask, 
        
        burst_time_interval,
        burst_duration,
        burst_length,
        burst_packet_num,
        burst_sequence_mask,
        
        flow_bytes_id, 
        byte_flow_mask, 
        byte_flow_attention_mask, 
        byte_bert_attention_mask,
        byte_flow_burst_indices, 
        label) = self.data[index]
        
        # Convert to tensor if not already
        packet_time_interval = packet_time_interval.detach().clone() if isinstance(packet_time_interval, torch.Tensor) else torch.tensor(packet_time_interval, dtype=torch.float32)
        packet_payload_length = packet_payload_length.detach().clone() if isinstance(packet_payload_length, torch.Tensor) else torch.tensor(packet_payload_length, dtype=torch.float32)
        packet_direction = packet_direction.detach().clone() if isinstance(packet_direction, torch.Tensor) else torch.tensor(packet_direction, dtype=torch.long)
        packet_sequence_burst_indices = packet_sequence_burst_indices.detach().clone() if isinstance(packet_sequence_burst_indices, torch.Tensor) else torch.tensor(packet_sequence_burst_indices, dtype=torch.long)
        packet_sequence_mask = packet_sequence_mask.detach().clone() if isinstance(packet_sequence_mask, torch.Tensor) else torch.tensor(packet_sequence_mask, dtype=torch.long)
        packet_sequence_attention_mask = packet_sequence_attention_mask.detach().clone() if isinstance(packet_sequence_attention_mask, torch.Tensor) else torch.tensor(packet_sequence_attention_mask, dtype=torch.long)
        
        burst_time_interval = torch.tensor(burst_time_interval, dtype=torch.float32)
        burst_duration = burst_duration.detach().clone() if isinstance(burst_duration, torch.Tensor) else torch.tensor(burst_duration, dtype=torch.float32)
        burst_length = burst_length.detach().clone() if isinstance(burst_length, torch.Tensor) else torch.tensor(burst_length, dtype=torch.float32)
        burst_packet_num = burst_packet_num.detach().clone() if isinstance(burst_packet_num, torch.Tensor) else torch.tensor(burst_packet_num, dtype=torch.float32)
        burst_sequence_mask = burst_sequence_mask.detach().clone() if isinstance(burst_sequence_mask, torch.Tensor) else torch.tensor(burst_sequence_mask, dtype=torch.long)
        
        flow_bytes_id = flow_bytes_id.detach().clone() if isinstance(flow_bytes_id, torch.Tensor) else torch.tensor(flow_bytes_id, dtype=torch.long)
        byte_flow_mask = byte_flow_mask.detach().clone() if isinstance(byte_flow_mask, torch.Tensor) else torch.tensor(byte_flow_mask, dtype=torch.long)
        byte_flow_attention_mask = byte_flow_attention_mask.detach().clone() if isinstance(byte_flow_attention_mask, torch.Tensor) else torch.tensor(byte_flow_attention_mask, dtype=torch.long)
        byte_bert_attention_mask = byte_bert_attention_mask.detach().clone() if isinstance(byte_bert_attention_mask, torch.Tensor) else torch.tensor(byte_bert_attention_mask, dtype=torch.long)
        byte_flow_burst_indices = byte_flow_burst_indices.detach().clone() if isinstance(byte_flow_burst_indices, torch.Tensor) else torch.tensor(byte_flow_burst_indices, dtype=torch.long)
        
        label = torch.tensor(label, dtype=torch.long)

        return (
            packet_time_interval,
            packet_payload_length,
            packet_direction,
            packet_sequence_burst_indices,
            packet_sequence_mask,
            packet_sequence_attention_mask,
            
            burst_time_interval,
            burst_duration,
            burst_length,
            burst_packet_num,
            burst_sequence_mask,
            
            flow_bytes_id, 
            byte_flow_mask, 
            byte_flow_attention_mask, 
            byte_bert_attention_mask,
            byte_flow_burst_indices,
            ), label  
    
    def __len__(self):
        """Return dataset size."""
        return len(self.data)
