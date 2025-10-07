import torch  # 导入PyTorch主库
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入神经网络函数库
import math  # 导入数学函数库


class MultiHeadAttention(nn.Module):  # 多头注意力机制类
    def __init__(self, d_model, num_heads, dropout=0.1):  # 初始化函数
        super(MultiHeadAttention, self).__init__()  # 调用父类初始化
        assert d_model % num_heads == 0  # 确保模型维度能被头数整除
        
        self.d_model = d_model  # 模型维度
        self.num_heads = num_heads  # 注意力头数
        self.d_k = d_model // num_heads  # 每个头的维度
        
        self.W_q = nn.Linear(d_model, d_model)  # Query的线性变换层
        self.W_k = nn.Linear(d_model, d_model)  # Key的线性变换层
        self.W_v = nn.Linear(d_model, d_model)  # Value的线性变换层
        self.W_o = nn.Linear(d_model, d_model)  # 输出的线性变换层
        
        self.dropout = nn.Dropout(dropout)  # dropout层，用于防止过拟合
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):  # 缩放点积注意力计算
        # print(f"Q shape: {Q.shape}")
        # print(f"K shape: {K.shape}")
        # print(f"V shape: {V.shape}")
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # 计算注意力分数并进行缩放
        # print(f"scores shape: {scores.shape}")
        
        if mask is not None:  # 如果有mask
            # print(f'mask shape: {mask.shape}')
            scores = scores.masked_fill(mask == 0, -1e9)  # 将mask位置的值设为极小值
            # print(f'scores after mask: {scores.shape}')
            
        attention_weights = F.softmax(scores, dim=-1)  # 计算注意力权重
        # print(f'attention_weights shape: {attention_weights.shape}')
        attention_weights = self.dropout(attention_weights)  # 对注意力权重应用dropout
        # print(f"attention_weights shape after dropout: {attention_weights.shape}")
        output = torch.matmul(attention_weights, V)  # 计算注意力输出
        # print(f"output shape: {output.shape}")
        
        return output
        
    def forward(self, Q, K, V, mask=None):  # 前向传播函数
        batch_size = Q.size(0)
        seq_len = Q.size(1)
        
        # 2. 线性变换后的维度
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        
        # 3. 重塑为多头形式后的维度
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 4. 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:  # 如果有mask
            # print(f'mask shape before unsqueeze: {mask.shape}')
            mask = mask.unsqueeze(1)  # 扩展mask维度
            # print(f'mask shape after unsqueeze: {mask.shape}')
            

        # 如果之后要恢复默认设置
        torch.set_printoptions(profile="default")
        
        output = self.scaled_dot_product_attention(Q, K, V, mask)  # 计算注意力
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)  # 重塑输出维度
        
        return self.W_o(output)  # 返回经过输出线性层的结果

class FeedForward(nn.Module):  # 前馈神经网络类
    def __init__(self, d_model, d_ff, dropout=0.1):  # 初始化函数
        super(FeedForward, self).__init__()  # 调用父类初始化
        self.linear1 = nn.Linear(d_model, d_ff)  # 第一个线性层
        self.linear2 = nn.Linear(d_ff, d_model)  # 第二个线性层
        self.dropout = nn.Dropout(dropout)  # dropout层
        
    def forward(self, x):  # 前向传播函数
        x = self.dropout(F.relu(self.linear1(x)))  # 第一层变换后接ReLU激活和dropout
        x = self.linear2(x)  # 第二层变换
        return x

class EncoderLayer(nn.Module):  # Transformer编码器层
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):  # 初始化函数
        super(EncoderLayer, self).__init__()  # 调用父类初始化
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)  # 多头自注意力层
        self.feed_forward = FeedForward(d_model, d_ff, dropout)  # 前馈神经网络层
        self.norm1 = nn.LayerNorm(d_model)  # 第一个层归一化
        self.norm2 = nn.LayerNorm(d_model)  # 第二个层归一化
        self.dropout = nn.Dropout(dropout)  # dropout层
        
    def forward(self, x, mask=None):  # 前向传播函数
        attn_output = self.self_attn(x, x, x, mask)  # 计算自注意力
        x = self.norm1(x + self.dropout(attn_output))  # 残差连接和层归一化
        ff_output = self.feed_forward(x)  # 前馈神经网络
        x = self.norm2(x + self.dropout(ff_output))  # 残差连接和层归一化
        return x

class Model(nn.Module):  # Transformer模型主类
    def __init__(self, config):  # 初始化函数
        super(Model, self).__init__()  # 调用父类初始化
        self.d_model = config.model.emb  # 模型维度
        self.num_heads = config.model.trf_heads  # 注意力头数
        self.num_layers = config.model.trf_layers  # 编码器层数
        self.d_ff = config.model.trf_feedforward  # 前馈网络维度
        self.dropout_rate = config.model.trf_dropout  # dropout比率
        self.max_burst_num = config.model.max_burst_num  # 设置最大burst数量
        self.use_segment_embedding = getattr(config.model, 'use_segment_embedding', False)  # 是否使用segment embedding
        self.use_position_embedding = getattr(config.model, 'use_position_embedding', True)  # 是否使用位置编码
        if self.use_position_embedding:
            self.pos_encoding = PositionalEncoding(self.d_model)  # 位置编码层，这个就是不用手动设置的啦
        if self.use_segment_embedding:
            self.segment_embedding = nn.Embedding(self.max_burst_num, self.d_model)  # 创建一个 segment embedding 表，包含 max_burst_num 个可学习的 embedding
        
        self.encoder_layers = nn.ModuleList([  # 创建多层编码器
            EncoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout_rate)
            for _ in range(self.num_layers)
        ])
        self.dropout = nn.Dropout(self.dropout_rate)  # dropout层
        
    def forward(self, x, burst_indices=None, mask=None):  # 前向传播函数
        """
        Args:
            x: 输入张量 [batch_size, pad_num, d_model]，每个数据包的表示
            burst_indices: burst索引张量 [batch_size, pad_num]，
                         表示每个数据包属于哪个burst
            mask: 注意力掩码
        """
        if self.use_position_embedding:
            x = self.pos_encoding(x)  # 添加位置编码
        
        if self.use_segment_embedding and burst_indices is not None:  # 如果使用segment embedding且提供了burst索引
            segment_embeddings = self.segment_embedding(burst_indices)  # 获取burst段嵌入
            x = x + segment_embeddings  # 将段嵌入添加到输入中
            
        x = self.dropout(x)  # 应用dropout
        
        for encoder_layer in self.encoder_layers:  # 依次通过每个编码器层
            x = encoder_layer(x, mask)  # 编码器层处理
            
        return x  # 返回最终输出

class PositionalEncoding(nn.Module):  # 位置编码类
    def __init__(self, d_model, max_len=5000):  # 初始化函数
        super(PositionalEncoding, self).__init__()  # 调用父类初始化
        pe = torch.zeros(max_len, d_model)  # 创建位置编码矩阵
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # 位置向量
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # 计算分母项
        pe[:, 0::2] = torch.sin(position * div_term)  # 计算偶数位置的正弦值
        pe[:, 1::2] = torch.cos(position * div_term)  # 计算奇数位置的余弦值
        pe = pe.unsqueeze(0)  # 增加批次维度
        self.register_buffer('pe', pe)  # 注册为缓冲区，不参与反向传播

    def forward(self, x):  # 前向传播函数
        return x + self.pe[:, :x.size(1)]  # 将位置编码添加到输入中,在5000个position embedding中只选取和输入长度相同的position embedding