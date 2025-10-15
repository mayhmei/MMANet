import os
import pickle
from tqdm import tqdm
import torch

def safe_convert_to_numeric(string_list, convert_type):
    """
    Safely convert a list of strings to numeric values.

    Args:
        string_list: list of strings
        convert_type: target type (int or float)

    Returns:
        Converted numeric list. Uses 0 for values that fail to convert.
    """
    # If conversion fails, use default value 0
    result = []
    for item in string_list:
        try:
            if convert_type == int:
                result.append(int(item))
            else:
                result.append(float(item))
        except (ValueError, TypeError):
            result.append(0)
    return result

def calculate_burst_indices(burst_packet_num):
    """
    Compute burst index for each packet. If sequence is empty, return empty list.
    """
    if not burst_packet_num:  # If the burst packet count list is empty
        return []
    # burst_packet_num is already integers; use directly
    burst_indices = []
    current_burst_idx = 0
    for current_burst_packet_num in burst_packet_num:
        # burst_packet_num now is already integers; use directly
        burst_indices.extend([current_burst_idx] * current_burst_packet_num)
        current_burst_idx += 1
    return burst_indices

def prepare_data(config, data_path):  # Data preparation up to mapping bytes to token IDs
    cache_dir = config.path.cache_path  # Cache directory
    file_name = os.path.basename(data_path)  # Data file name
    file_name_without_ext = os.path.splitext(file_name)[0]  # File name without extension
    cached_dataset_file = cache_dir + '{}_{}_{}_{}_{}.txt'.format(  # Build cache file path
        file_name_without_ext, config.data.pad_num, config.data.pad_len, config.data.packet_sequence_num, config.data.burst_sequence_num
    )

    if os.path.exists(cached_dataset_file):  # If cache exists
        print("Loading features from cached file {}".format(cached_dataset_file))  # Cache load info
        with open(cached_dataset_file, "rb") as handle:  # Open cache file
            data = pickle.load(handle)  # Load cached data
    else:  # If cache does not exist
        print(f"Creating dataset from {data_path}....")  # Creating dataset info
        data = []  # Processed data container
        with open(data_path, 'r') as f:  # Open data file
            for line in tqdm(f):  # Iterate per line with progress bar
                if not line:  # Skip empty line
                    continue
                item = line.split('\t')  # Split by tab into fields
                if len(item) < 10:  # Fewer than required fields (must include sequences and label)
                    num_short += 1
                    print(f"[Insufficient fields] Line {num_total} skipped: less than 10 -> {item}")
                    continue
                # Extract sequences: packets, directions, payload lengths, packet intervals; burst (packet count, size, duration, intervals); label
                packets = item[:-8]  # Packet strings excluding trailing sequences and label
                burst_time_interval = safe_convert_to_numeric(item[-8].split(' '), float)   # Burst time intervals
                burst_duration = safe_convert_to_numeric(item[-7].split(' '), float)   # Burst durations
                burst_length = safe_convert_to_numeric(item[-6].split(' '), int)   # Burst sizes
                burst_packet_num = safe_convert_to_numeric(item[-5].split(' '), int)   # Packets per burst
                packet_time_interval = safe_convert_to_numeric(item[-4].split(' '), float)   # Packet time intervals
                packet_payload_length = safe_convert_to_numeric(item[-3].split(' '), int)   # Packet payload lengths
                packet_direction = safe_convert_to_numeric(item[-2].split(' '), int)  # Packet directions
                label = item[-1]  # Label

                # Ensure counts match across packet-level sequences
                if len(packets) != len(packet_direction) != len(packet_payload_length) != len(packet_time_interval):
                    print(f"⚠️ Data inconsistency: packet count ({len(packets)}) mismatched with sequence lengths; sample skipped.")
                    continue

                burst_indices = calculate_burst_indices(burst_packet_num)  # Compute burst index per packet (before padding)

                # 1) Sequence-level handling: truncate to packet_sequence_num; pad with zeros otherwise
                if len(packet_time_interval) > config.data.packet_sequence_num:
                    packet_time_interval = packet_time_interval[:config.data.packet_sequence_num]
                    packet_payload_length = packet_payload_length[:config.data.packet_sequence_num]
                    packet_sequence_burst_indices = burst_indices[:config.data.packet_sequence_num]
                    packet_direction = packet_direction[:config.data.packet_sequence_num]
                    sequence_valid_packet = len(packet_time_interval)
                else:
                    sequence_valid_packet = len(packet_time_interval)

                # packet_sequence_mask: for transformer linear-attention pooling; ignore padding packets during pooling
                packet_sequence_mask = torch.zeros(config.data.packet_sequence_num, dtype=torch.long)
                packet_sequence_mask[:sequence_valid_packet] = 1
                packet_sequence_attention_mask = torch.zeros((config.data.packet_sequence_num, config.data.packet_sequence_num), dtype=torch.long)
                packet_sequence_attention_mask[:sequence_valid_packet, :sequence_valid_packet] = 1

                #2) Burst-level handling: truncate to burst_sequence_num; pad with zeros otherwise
                if len(burst_time_interval) > config.data.burst_sequence_num:
                    burst_time_interval = burst_time_interval[:config.data.burst_sequence_num]
                    burst_duration = burst_duration[:config.data.burst_sequence_num]
                    burst_length = burst_length[:config.data.burst_sequence_num]
                    burst_packet_num = burst_packet_num[:config.data.burst_sequence_num]
                    sequence_valid_burst = len(burst_time_interval)
                else:   
                    sequence_valid_burst = len(burst_time_interval)
                    burst_time_interval = burst_time_interval + [0] * (config.data.burst_sequence_num - len(burst_time_interval))
                    burst_duration = burst_duration + [0] * (config.data.burst_sequence_num - len(burst_duration))
                    burst_length = burst_length + [0] * (config.data.burst_sequence_num - len(burst_length))
                    burst_packet_num = burst_packet_num + [0] * (config.data.burst_sequence_num - len(burst_packet_num))
                burst_sequence_mask = torch.zeros(config.data.burst_sequence_num, dtype=torch.long)
                burst_sequence_mask[:sequence_valid_burst] = 1
                # burst_sequence_attention_mask = torch.zeros((config.data.burst_sequence_num, config.data.burst_sequence_num), dtype=torch.long)
                # burst_sequence_attention_mask[:sequence_valid_burst, :sequence_valid_burst] = 1

                
                
                #3) Byte-level handling within each packet
                # Truncate flows exceeding pad_num
                if len(packets) > config.data.pad_num:
                    packets = packets[:config.data.pad_num]
                    byte_flow_burst_indices = burst_indices[:config.data.pad_num]
                else:
                    byte_flow_burst_indices = burst_indices
                flow_bytes_id = []
                flow_valid_byte_len = []

                # Process each packet
                for packet in packets:
                    packet_bytes = config.tokenizer.tokenize(packet)  # Tokenize packet bytes
                    CLS = getattr(config, 'CLS', '[CLS]')
                    SEP = getattr(config, 'SEP', '[SEP]')

                    # Truncate to pad_len - 2 (reserve for CLS and SEP)
                    if len(packet_bytes) > config.data.pad_len - 2:
                        packet_bytes = packet_bytes[:(config.data.pad_len - 2)]

                    # Add CLS and SEP markers
                    packet_bytes = [CLS] + packet_bytes + [SEP]
                    packet_valid_byte_len = len(packet_bytes)

                    # Convert to IDs
                    packet_bytes_ids = config.tokenizer.convert_tokens_to_ids(packet_bytes)
                    
                    # Pad to pad_len with zeros for fixed length
                    if len(packet_bytes_ids) < config.data.pad_len:
                        packet_bytes_ids += ([0] * (config.data.pad_len - len(packet_bytes_ids)))
                    
                    flow_bytes_id.append(packet_bytes_ids)
                    flow_valid_byte_len.append(packet_valid_byte_len)
                
                # Pad missing packets up to pad_num
                if len(flow_bytes_id) < config.data.pad_num:
                    len_tmp = len(flow_bytes_id)
                    packet_bytes_ids = [0] * config.data.pad_len
                    packet_valid_byte_len = 0
                    last_burst_idx = byte_flow_burst_indices[-1] if byte_flow_burst_indices else 0
                    
                    for i in range(config.data.pad_num - len_tmp):
                        flow_bytes_id.append(packet_bytes_ids)
                        flow_valid_byte_len.append(packet_valid_byte_len)
                        byte_flow_burst_indices.append(last_burst_idx + 1)
                        
                # Build pad_num x pad_num attention mask for inter-packet relations in transformer (explicit dtype)
                byte_flow_attention_mask = torch.zeros(config.data.pad_num, config.data.pad_num, dtype=torch.long)  # explicitly set dtype
                byte_flow_attention_mask[:flow_valid_packet_num, :flow_valid_packet_num] = 1

                # Build 1D mask of valid packets; positions of valid packets set to 1; used by final linear attention to pool packet representations
                byte_flow_mask = torch.zeros(config.data.pad_num, dtype=torch.long)
                byte_flow_mask[:flow_valid_packet_num] = 1

                # Build BERT attention mask per packet
                byte_bert_attention_mask = torch.zeros((config.data.pad_num, config.data.pad_len), dtype=torch.long)  # [pad_num, pad_len]
                for i, packet_valid_byte_len in enumerate(flow_valid_byte_len):
                    if packet_valid_byte_len > 0:  # If packet is valid
                        byte_bert_attention_mask[i, :packet_valid_byte_len] = 1  # Set positions of valid tokens to 1
                                    
                data.append((
                             packet_time_interval,
                             packet_payload_length,
                             packet_direction,
                             packet_sequence_burst_indices,  # For adding burst segment embedding to each sequence; length differs from Bert_burst_indices
                             packet_sequence_mask,
                             packet_sequence_attention_mask,
                             
                             burst_time_interval,
                             burst_duration,
                             burst_length,
                             burst_packet_num,
                             burst_sequence_mask,
                            #  burst_sequence_attention_mask,
                             
                             flow_bytes_id, 
                             byte_flow_mask, 
                             byte_flow_attention_mask, 
                             byte_bert_attention_mask,  # Used in pretraining: records valid byte length per packet in a flow, aligned with flow_bytes_id structure
                             byte_flow_burst_indices,
                             int(label)))
        
        print(f"Saving dataset cached file {cached_dataset_file}")
        os.makedirs(cache_dir, exist_ok=True)
        with open(cached_dataset_file, "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)         
    return data