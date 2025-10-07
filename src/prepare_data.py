import os
import pickle
from tqdm import tqdm
import torch

def safe_convert_to_numeric(string_list, convert_type=float):
    """
    安全地将字符串列表转换为数值列表
    Args:
        string_list: 字符串列表
        convert_type: 转换类型 (int 或 float)
    Returns:
        转换后的数值列表
    """
    result = []
    for item in string_list:
        try:
            if convert_type == int:
                result.append(int(item))
            else:
                result.append(float(item))
        except (ValueError, TypeError):
            # 如果转换失败，使用默认值0
            result.append(0)
    return result

def calculate_burst_indices(burst_packet_num):
    if not burst_packet_num:  # 如果方向序列为空
        return []
    burst_indices = []
    current_burst_idx = 0
    for current_burst_packet_num in burst_packet_num:
        # burst_packet_num现在已经是整数列表，直接使用
        burst_indices.extend([current_burst_idx] * current_burst_packet_num)
        current_burst_idx += 1
    return burst_indices

def prepare_data(config, data_path):  # 数据准备函数，准备到将字节映射成token ID这一步
    cache_dir = config.path.cache_path  # 缓存目录
    file_name = os.path.basename(data_path)  # 获取数据文件名
    file_name_without_ext = os.path.splitext(file_name)[0]  # 获取文件名（不包含扩展名）
    cached_dataset_file = cache_dir + '{}_{}_{}_{}_{}.txt'.format(  # 构建缓存文件路径
        file_name_without_ext, config.data.pad_num, config.data.pad_len, config.data.packet_sequence_num, config.data.burst_sequence_num
    )

    if os.path.exists(cached_dataset_file):  # 如果缓存文件存在
        print("Loading features from cached file {}".format(cached_dataset_file))  # 打印加载缓存信息
        with open(cached_dataset_file, "rb") as handle:  # 打开缓存文件
            data = pickle.load(handle)  # 加载缓存数据
    else:  # 如果缓存文件不存在
        print(f"Creating dataset from {data_path}....")  # 打印创建数据集信息
        data = []  # 存储处理后的数据
        num_short = 0
        num_total = 0
        with open(data_path, 'r') as f:  # 打开数据文件
            for line in tqdm(f):  # 遍历每一行，显示进度条
                if not line:  # 如果行为空
                    continue
                
                item = line.split('\t')  # 按制表符分割行，得到一个以\t分割好的列表
                if len(item) < 10:  # 如果项目数少于4（至少需要两个数据包、方向序列和标签）.................这个是后续一直需要不断修改的地方
                    num_short += 1
                    print(f"[字段不足] Line {num_total} skipped: less than 4 fields -> {item}")
                    continue
                
                # 获取数据包序列、方向序列、payload序列、时间间隔序列（与上一个数据包）、burst序列（4（burst包数量序列，burst大小序列，burst持续时间，burst时间间隔））、标签
                packets = item[:-8]  # 除去序列和标签的部分是数据包
                burst_time_interval = safe_convert_to_numeric(item[-8].split(' '), float)   # 倒数第二个是burst时间间隔序列，按空格分割
                burst_duration = safe_convert_to_numeric(item[-7].split(' '), float)   # 倒数第三个是burst持续时间序列，按空格分割
                burst_length = safe_convert_to_numeric(item[-6].split(' '), int)   # 倒数第四个是burst大小序列，按空格分割
                burst_packet_num = safe_convert_to_numeric(item[-5].split(' '), int)   # 倒数第五个是burst包数量序列，按空格分割
                packet_time_interval = safe_convert_to_numeric(item[-4].split(' '), float)   # 倒数第六个是packet时间间隔序列，按空格分割
                packet_payload_length = safe_convert_to_numeric(item[-3].split(' '), int)   # 倒数第七个是packet的payload长度序列，按空格分割
                packet_direction = safe_convert_to_numeric(item[-2].split(' '), int)  # 倒数第八个是方向序列，按空格分割成一个序列
                label = item[-1]  # 最后一个是标签
                
                # 确保数据包数量和方向序列长度匹配，对比两个序列的长度是否一致
                if len(packets) != len(packet_direction) != len(packet_payload_length) != len(packet_time_interval):
                    print(f"⚠️ 数据处理异常：数据包数量（{len(packets)}）与packet序列长度（{len(packet_direction)}）不一致，已跳过该样本。")
                    continue
                
                burst_indices = calculate_burst_indices(burst_packet_num)  # 计算每个数据包的burst索引,为什么在这里，因为要在burst_packet_num没有被处理之前进行填充
                
                # 1. 先处理流序列数据，超过packet_sequence_num的进行截断，不足的补零
                if len(packet_time_interval) > config.data.packet_sequence_num:
                    packet_time_interval = packet_time_interval[:config.data.packet_sequence_num]
                    packet_payload_length = packet_payload_length[:config.data.packet_sequence_num]
                    packet_sequence_burst_indices = burst_indices[:config.data.packet_sequence_num]
                    packet_direction = packet_direction[:config.data.packet_sequence_num]
                    sequence_valid_packet = len(packet_time_interval)
                else:
                    sequence_valid_packet = len(packet_time_interval)
                    packet_time_interval = packet_time_interval + [0] * (config.data.packet_sequence_num - len(packet_time_interval))
                    packet_payload_length = packet_payload_length + [0] * (config.data.packet_sequence_num - len(packet_payload_length))
                    packet_direction = packet_direction + [0] * (config.data.packet_sequence_num - len(packet_direction))
                    last_burst_idx = burst_indices[-1] if burst_indices else 0
                    packet_sequence_burst_indices = burst_indices + [last_burst_idx + 1] * (config.data.packet_sequence_num - len(burst_indices))
                    
                    
                    # packet_direction_sequence = packet_direction_sequence + [0] * (config.data.packet_sequence_num - len(packet_direction_sequence))
                # packet_sequence_mask: 用于处理transformer传出的计算结果，用于线性注意力层，线性注意力层计算sequence得综合向量，计算过程中忽略padding的数据包
                packet_sequence_mask = torch.zeros(config.data.packet_sequence_num, dtype=torch.long)
                packet_sequence_mask[:sequence_valid_packet] = 1
                packet_sequence_attention_mask = torch.zeros((config.data.packet_sequence_num, config.data.packet_sequence_num), dtype=torch.long)
                packet_sequence_attention_mask[:sequence_valid_packet, :sequence_valid_packet] = 1
                
                
                #2.  再处理burst序列数据，超过burst_sequence_num的进行截断，不足的补零
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
                
                
                
                #3.  最后处理包内字节数据，
                # 先截断超过规定包数量的流
                if len(packets) > config.data.pad_num:
                    packets = packets[:config.data.pad_num]
                    # packet_direction_sequence = packet_direction_sequence[:config.data.pad_num]
                    # packet_payload_length = packet_payload_length[:config.data.pad_num]
                    # packet_time_interval = packet_time_interval[:config.data.pad_num]
                    byte_flow_burst_indices = burst_indices[:config.data.pad_num]
                else:
                    byte_flow_burst_indices = burst_indices
                flow_bytes_id = []
                flow_valid_byte_len = []
                
                # 处理每个数据包
                for packet in packets:
                    packet_bytes = config.tokenizer.tokenize(packet)  # 分词
                    CLS = getattr(config, 'CLS', '[CLS]')
                    SEP = getattr(config, 'SEP', '[SEP]')
                    
                    # 先进行截断，确保长度不超过 pad_len - 2 (为CLS和SEP预留位置)
                    if len(packet_bytes) > config.data.pad_len - 2:
                        packet_bytes = packet_bytes[:(config.data.pad_len - 2)]
                    
                    # 添加CLS和SEP标记
                    packet_bytes = [CLS] + packet_bytes + [SEP]
                    packet_valid_byte_len = len(packet_bytes)
                    
                    # 转换为ID
                    packet_bytes_ids = config.tokenizer.convert_tokens_to_ids(packet_bytes)
                    
                    # 进行padding
                    if len(packet_bytes_ids) < config.data.pad_len:
                        packet_bytes_ids += ([0] * (config.data.pad_len - len(packet_bytes_ids)))
                    
                    flow_bytes_id.append(packet_bytes_ids)
                    flow_valid_byte_len.append(packet_valid_byte_len)
                
                # 填充不足的数据包
                if len(flow_bytes_id) < config.data.pad_num:
                    len_tmp = len(flow_bytes_id)
                    packet_bytes_ids = [0] * config.data.pad_len
                    packet_valid_byte_len = 0
                    last_burst_idx = byte_flow_burst_indices[-1] if byte_flow_burst_indices else 0
                    
                    for i in range(config.data.pad_num - len_tmp):
                        flow_bytes_id.append(packet_bytes_ids)
                        flow_valid_byte_len.append(packet_valid_byte_len)
                        byte_flow_burst_indices.append(last_burst_idx + 1)
                        
                # 生成pad_num_attention mask矩阵，用于transformer计算包间的关系
                byte_flow_attention_mask = torch.zeros(config.data.pad_num, config.data.pad_num, dtype=torch.long)  # 明确指定dtype
                flow_valid_packet_num = len(packets)
                byte_flow_attention_mask[:flow_valid_packet_num, :flow_valid_packet_num] = 1
                
                # 生成有效的packet_num的张量，一个一维的张量，有效packet的位置为1,用于最后的线性attention层计算包的代表向量
                byte_flow_mask = torch.zeros(config.data.pad_num, dtype=torch.long)
                byte_flow_mask[:flow_valid_packet_num] = 1
                
                # 生成bert_attention_mask矩阵
                byte_bert_attention_mask = torch.zeros((config.data.pad_num, config.data.pad_len), dtype=torch.long)  # [10, 400]
                for i, packet_valid_byte_len in enumerate(flow_valid_byte_len):
                    if packet_valid_byte_len > 0:  # 如果是有效的数据包
                        byte_bert_attention_mask[i, :packet_valid_byte_len] = 1  # 将有效token位置设为1
                                    
                data.append((
                             packet_time_interval,
                             packet_payload_length,
                             packet_direction,
                             packet_sequence_burst_indices, # 用于给每个序列添加burst segment embedding ，和Bert_burst_indices长度不一样
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
                             byte_bert_attention_mask,  # 预训练层要用到的,记录着一条流中每个数据包的有效字节，和flow_bytes_id结构一样
                             byte_flow_burst_indices, 
                             int(label)))
        
        print(f"Saving dataset cached file {cached_dataset_file}")
        os.makedirs(cache_dir, exist_ok=True)
        with open(cached_dataset_file, "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)         
    return data