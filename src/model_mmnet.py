import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from transformers import AutoModelForMaskedLM, AutoModel, AutoTokenizer, BertTokenizer
import Transformer  # Import custom Transformer module

# Add parent directory to sys.path to import other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from src_byte.MMNet_byte_model import MMNet_byte
# from src_burst_sequence_tfm_2.MMNet_burst_model import MMNet_burst
# from src_packet_sequence_trf_2.MMNet_packet_model import MMNet_packet

class ProjectionHead(nn.Module):
    """Projection head: map features from different models into a shared feature space"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return F.normalize(self.projection(x), p=2, dim=1)

class SequenceFeatureFusion(nn.Module):
    """Module to fuse burst and packet features"""
    def __init__(self, input_dim=128):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, burst_features, packet_features):
        # Concatenate the two features
        combined = torch.cat([burst_features, packet_features], dim=1)
        # Fuse features
        fused = self.fusion(combined)
        return fused

class AttentionPooling(nn.Module):
    """Linear attention pooling layer that pools flow outputs into a vector"""
    def __init__(self, emb_size):
        super().__init__()
        self.attention = nn.Linear(emb_size, 1)
        self.pooler = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask):
        # x: [B, T, H]
        # mask: [B, T] (bool), True=valid, False=padding
        scores = self.attention(x).squeeze(-1)  # [B, T]
        if mask is not None:
            mask = mask.bool()  # Convert mask to boolean
        scores = scores.masked_fill(~mask, -1e9)  # Mask out padding
        weights = torch.softmax(scores, dim=1)  # [B, T]
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # [B, H]
        pooled = torch.tanh(self.pooler(pooled))  # Mimic BERT pooler [B, H]
        return pooled

# Added: BERT-style lightweight fusion block (self-attention + cross-attention + FFN + residual + LN)
class FusionBertBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=dropout)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attention_mask, encoder_hidden_states, encoder_attention_mask):
        # attention_mask/encoder_attention_mask: [B, T], 1=valid, 0=pad
        key_pad_self = None if attention_mask is None else (~attention_mask.bool())
        key_pad_cross = None if encoder_attention_mask is None else (~encoder_attention_mask.bool())

        sa_out, _ = self.self_attn(x, x, x, key_padding_mask=key_pad_self, need_weights=False)
        x = self.ln1(x + self.dropout1(sa_out))
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).type_as(x)

        ca_out, _ = self.cross_attn(x, encoder_hidden_states, encoder_hidden_states,
                                    key_padding_mask=key_pad_cross, need_weights=False)
        x = self.ln2(x + self.dropout2(ca_out))
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).type_as(x)

        ffn_out = self.ffn(x)
        x = self.ln3(x + ffn_out)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).type_as(x)
        return x

class FusionBert(nn.Module):
    def __init__(self, hidden_size, num_heads, num_layers=1, intermediate_mul=4, dropout=0.1):
        super().__init__()
        intermediate_size = hidden_size * intermediate_mul  # Hidden size in FFN
        self.layers = nn.ModuleList([
            FusionBertBlock(hidden_size, num_heads, intermediate_size, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, encoder_embeds, attention_mask, encoder_hidden_states, encoder_attention_mask,
                return_dict=False, mode=None):
        x = encoder_embeds
        for layer in self.layers:
            x = layer(x, attention_mask, encoder_hidden_states, encoder_attention_mask)
        return x

class MMNet(nn.Module):
    """ALBEF-style contrastive MMNet model"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mode = getattr(config.model, 'mode', 'pretrain')  # 'pretrain' or 'finetune'
        
        # Learnable temperature parameter; initialized from config (default 0.07)
        self.temp = nn.Parameter(torch.tensor(float(getattr(config.model, 'temperature', 0.07))))

        # ====== Backbone network ======
        # Byte branch - use AutoModelForMaskedLM to support MLM
        self.inter_byte_encoder = AutoModelForMaskedLM.from_pretrained(config.path.pretrain_path)
        for param in self.inter_byte_encoder.parameters():
            param.requires_grad = config.model.update_pretrain_params
        self.byte_transformer = Transformer.Model(config=self.config)
        self.byte_attn_pool = AttentionPooling(self.config.model.emb)
        
        # Packet branch
        self.packet_transformer = Transformer.Model(config=self.config)
        self.packet_attn_pool = AttentionPooling(self.config.model.emb)
        self.packet_projection = nn.Linear(1, self.config.model.emb)
        
        # Burst branch
        self.burst_transformer = Transformer.Model(config=self.config)
        self.burst_attn_pool = AttentionPooling(self.config.model.emb)
        self.burst_projection = nn.Linear(1, self.config.model.emb)
        
        # Projection heads
        self.byte_projector = ProjectionHead(
            input_dim=self.config.model.emb, 
            hidden_dim=self.config.model.emb * 2, 
            output_dim=self.config.model.emb
        )
        self.sequence_projector = ProjectionHead(
            input_dim=self.config.model.emb * 2, 
            hidden_dim=self.config.model.emb * 4, 
            output_dim=self.config.model.emb
        )
        
        # Classification head (used in finetune)
        self.classifier = nn.Sequential(
            # nn.Linear(self.config.model.emb * 3, 512),
            nn.Linear(self.config.model.emb, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, config.model.num_classes)
        )
        
        # ===== ALBEF-style cross-modal fusion layer and BSM head =====
        heads = int(getattr(self.config.model, "fusion_heads", 4))
        # self.fusion_xattn = nn.MultiheadAttention(
        #     embed_dim=self.config.model.emb, num_heads=heads, batch_first=True
        # )
        fusion_layers = int(getattr(self.config.model, "fusion_layers", 1))
        fusion_inter_mul = int(getattr(self.config.model, "fusion_intermediate_mul", 4))
        fusion_dropout = float(getattr(self.config.model, "fusion_dropout", 0.1))
        self.fusion_fbert = FusionBert(
            hidden_size=self.config.model.emb,
            num_heads=heads,
            num_layers=fusion_layers,
            intermediate_mul=fusion_inter_mul,
            dropout=fusion_dropout
        )
        self.bsm_head = nn.Linear(self.config.model.emb, 2)
        # Loss weights for BSM, BSC, and MLM
        self.lambda_bsc = float(getattr(self.config.model, "lambda_bsc", 1.0))
        self.lambda_bsm = float(getattr(self.config.model, "lambda_itm", 1.0))  # compatibility with old configs
        self.lambda_mlm = float(getattr(self.config.model, "lambda_mlm", 1.0))
        self.mlm_probability = float(getattr(self.config.model, "mlm_probability", 0.15))
        
        # Tokenizer (used for MLM masking)
        # Originally loaded from pretrain path (will error if tokenizer files are missing)
        # self.tokenizer = AutoTokenizer.from_pretrained(config.path.pretrain_path)
        
        # Recommended: consistent with pretraining; initialize from local vocab.txt
        vocab_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Config", "vocab.txt"))
        self.tokenizer = BertTokenizer(vocab_file=vocab_path)
        
        # ====== ALBEF-style momentum branch (used only in pretrain mode) ======
        if self.mode == 'pretrain':
            self._setup_momentum_branch()
            self._setup_queues()
    
    def forward(self, inputs, labels=None):
        """Forward pass"""
        if self.mode == 'pretrain':
            # Pretrain mode: return contrastive features (ignore extra returned items)
            byte_feat, sequence_feat, _, _, _, _, _, _, _, _, _ = self._extract_features(inputs, use_momentum=False)
            return byte_feat, sequence_feat
        elif self.mode == 'finetune':
            # Finetune mode: return classification logits (ignore extra returned items)
            _, _, _, _, _, packet_flow_output, burst_flow_output, packet_flow_mask, burst_flow_mask, byte_flow_output, byte_flow_mask = self._extract_features(inputs, use_momentum=False)
            sequence_embeds = torch.cat([packet_flow_output, burst_flow_output], dim=1)
            sequence_mask = torch.cat([packet_flow_mask, burst_flow_mask], dim=1)
            fused_output = self.fusion_fbert(
                encoder_embeds = byte_flow_output,
                attention_mask = byte_flow_mask,
                encoder_hidden_states = sequence_embeds,
                encoder_attention_mask = sequence_mask,
            )  # [B, T_byte, H]
            # Apply attention pooling on fused token sequence to get [B, H] for classification
            fused_pooled = self.byte_attn_pool(fused_output, byte_flow_mask)  # [B, H]
            logits = self.classifier(fused_pooled)
            return logits
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _extract_features(self, inputs, use_momentum=False):
        """
        Extract features; consistent with original _extract_sequences
        Returns: byte_feat, sequence_feat, byte_pooled, packet_pooled, burst_pooled
        """
        (
            packet_time_interval, packet_payload_length, _,
            packet_sequence_burst_indices, packet_sequence_mask, _,
            burst_time_interval, burst_duration, burst_length, burst_packet_num,
            burst_sequence_mask,
            flow_bytes_id, byte_flow_mask, byte_flow_attention_mask,
            byte_bert_attention_mask, byte_flow_burst_indices
        ) = inputs

        device = next(self.parameters()).device
        batch_size = flow_bytes_id.size(0)

        # Move inputs to device
        packet_time_interval = packet_time_interval.to(device)
        packet_payload_length = packet_payload_length.to(device)
        packet_sequence_burst_indices = packet_sequence_burst_indices.to(device)
        packet_sequence_mask = packet_sequence_mask.to(device)
        burst_time_interval = burst_time_interval.to(device)
        burst_duration = burst_duration.to(device)
        burst_length = burst_length.to(device)
        burst_packet_num = burst_packet_num.to(device)
        burst_sequence_mask = burst_sequence_mask.to(device)
        flow_bytes_id = flow_bytes_id.to(device)
        byte_flow_mask = byte_flow_mask.to(device)
        byte_flow_attention_mask = byte_flow_attention_mask.to(device)
        byte_bert_attention_mask = byte_bert_attention_mask.to(device)
        byte_flow_burst_indices = byte_flow_burst_indices.to(device)
        
        # Build attention masks
        if self.config.data.Serial_features:
            packet_sequence_mask_expanded = packet_sequence_mask.repeat(1, self.config.data.pcaket_features_num)
            packet_sequence_attention_mask = packet_sequence_mask_expanded.unsqueeze(1) * packet_sequence_mask_expanded.unsqueeze(2)
            packet_sequence_burst_indices = packet_sequence_burst_indices.repeat(1, self.config.data.pcaket_features_num)
            burst_sequence_mask_expanded = burst_sequence_mask.repeat(1, self.config.data.burst_features_num)
            burst_sequence_attention_mask = burst_sequence_mask_expanded.unsqueeze(1) * burst_sequence_mask_expanded.unsqueeze(2)
        elif self.config.data.Parallel_features:
            packet_sequence_attention_mask = packet_sequence_mask.unsqueeze(1) * packet_sequence_mask.unsqueeze(2)
            burst_sequence_attention_mask = burst_sequence_mask.unsqueeze(1) * burst_sequence_mask.unsqueeze(2)
            packet_sequence_mask_expanded = packet_sequence_mask  # Rename for consistency: later logic uses packet_sequence_mask_expanded
            burst_sequence_mask_expanded = burst_sequence_mask
        
        packet_sequence_attention_mask = packet_sequence_attention_mask.to(device)
        burst_sequence_attention_mask = burst_sequence_attention_mask.to(device)
        
        # Choose backbone or momentum network
        if use_momentum:
            inter_byte_encoder = self.inter_byte_encoder_m
            byte_transformer = self.byte_transformer_m
            byte_attn_pool = self.byte_attn_pool_m
            packet_projection = self.packet_projection_m
            packet_transformer = self.packet_transformer_m
            packet_attn_pool = self.packet_attn_pool_m
            burst_projection = self.burst_projection_m
            burst_transformer = self.burst_transformer_m
            burst_attn_pool = self.burst_attn_pool_m
            byte_projector = self.byte_projector_m
            sequence_projector = self.sequence_projector_m
        else:
            inter_byte_encoder = self.inter_byte_encoder
            byte_transformer = self.byte_transformer
            byte_attn_pool = self.byte_attn_pool
            packet_projection = self.packet_projection
            packet_transformer = self.packet_transformer
            packet_attn_pool = self.packet_attn_pool
            burst_projection = self.burst_projection
            burst_transformer = self.burst_transformer
            burst_attn_pool = self.burst_attn_pool
            byte_projector = self.byte_projector
            sequence_projector = self.sequence_projector
        
        # 1. Byte feature extraction
        flow_bytes_id_flat = flow_bytes_id.view(batch_size * self.config.data.pad_num, -1)
        byte_bert_attention_mask_flat = byte_bert_attention_mask.view(batch_size * self.config.data.pad_num, -1)
        
        # For AutoModelForMaskedLM, obtain pooled output
        inter_byte_outputs = inter_byte_encoder(
            input_ids=flow_bytes_id_flat,
            attention_mask=byte_bert_attention_mask_flat,
            output_hidden_states=True
        )
        # Use [CLS] token hidden state as the pooled output
        inter_byte_cls_output = inter_byte_outputs.hidden_states[-1][:, 0, :]  # [batch*pad_num, emb]
        inter_byte_packet_output = inter_byte_cls_output.view(batch_size, self.config.data.pad_num, -1)  # Reshape into a packet sequence
        # Learn inter-packet byte relations
        byte_flow_output = byte_transformer(inter_byte_packet_output, byte_flow_burst_indices, byte_flow_attention_mask)
        byte_pooled_output = byte_attn_pool(byte_flow_output, byte_flow_mask)
        
        # 2. Packet feature extraction
        packet_features = torch.cat([packet_time_interval, packet_payload_length], dim=1).unsqueeze(-1)
        projected_packet = packet_projection(packet_features)
        packet_flow_output = packet_transformer(
            x=projected_packet, 
            mask=packet_sequence_attention_mask, 
            burst_indices=packet_sequence_burst_indices
        )
        packet_pooled_output = packet_attn_pool(packet_flow_output, packet_sequence_mask_expanded)
        
        # 3. Burst feature extraction
        # Construct segment embedding
        if self.config.model.use_feature_segment_embedding:
            seq_len = burst_length.shape[1]
            burst_features_num = self.config.data.burst_features_num
            segment_ids = []
            for i in range(burst_features_num):
                segment_ids.extend([i] * seq_len)
            segment_ids = torch.tensor(segment_ids, device=device).unsqueeze(0).repeat(batch_size, 1)
        else:
            segment_ids = None
            
        burst_features = torch.cat([
            burst_time_interval, burst_packet_num, burst_duration, burst_length
        ], dim=1).unsqueeze(-1)
        projected_burst = burst_projection(burst_features)
        burst_flow_output = burst_transformer(
            x=projected_burst, 
            burst_indices=segment_ids, 
            mask=burst_sequence_attention_mask
        )
        burst_pooled_output = burst_attn_pool(burst_flow_output, burst_sequence_mask_expanded)
        
        # Fuse sequence features and project
        combined_sequence_features = torch.cat([burst_pooled_output, packet_pooled_output], dim=1)
        byte_feat = F.normalize(byte_projector(byte_pooled_output), dim=-1)
        sequence_feat = F.normalize(sequence_projector(combined_sequence_features), dim=-1)
        
        # Additionally return token-level outputs and corresponding masks (1=valid, 0=padding)
        return (
            byte_feat, sequence_feat,
            byte_pooled_output, packet_pooled_output, burst_pooled_output,
            packet_flow_output, burst_flow_output,
            packet_sequence_mask_expanded, burst_sequence_mask_expanded,
            byte_flow_output, byte_flow_mask
        )


    def _setup_momentum_branch(self):
        """Set up momentum branch (EMA version of feature extractors)"""
        emb = self.config.model.emb
        
        # Momentum version of Byte branch - using AutoModelForMaskedLM
        self.inter_byte_encoder_m = AutoModelForMaskedLM.from_pretrained(self.config.path.pretrain_path)
        for p in self.inter_byte_encoder_m.parameters():
            p.requires_grad = False
        self.byte_transformer_m = Transformer.Model(config=self.config)
        for p in self.byte_transformer_m.parameters():
            p.requires_grad = False
        self.byte_attn_pool_m = AttentionPooling(emb)
        for p in self.byte_attn_pool_m.parameters():
            p.requires_grad = False

        # Momentum version of Packet branch
        self.packet_projection_m = nn.Linear(1, emb)
        for p in self.packet_projection_m.parameters():
            p.requires_grad = False
        self.packet_transformer_m = Transformer.Model(config=self.config)
        for p in self.packet_transformer_m.parameters():
            p.requires_grad = False
        self.packet_attn_pool_m = AttentionPooling(emb)
        for p in self.packet_attn_pool_m.parameters():
            p.requires_grad = False

        # Momentum version of Burst branch
        self.burst_projection_m = nn.Linear(1, emb)
        for p in self.burst_projection_m.parameters():
            p.requires_grad = False
        self.burst_transformer_m = Transformer.Model(config=self.config)
        for p in self.burst_transformer_m.parameters():
            p.requires_grad = False
        self.burst_attn_pool_m = AttentionPooling(emb)
        for p in self.burst_attn_pool_m.parameters():
            p.requires_grad = False

        # Momentum version of projection heads
        self.byte_projector_m = ProjectionHead(input_dim=emb, hidden_dim=emb * 2, output_dim=emb)
        for p in self.byte_projector_m.parameters():
            p.requires_grad = False
        self.sequence_projector_m = ProjectionHead(input_dim=emb * 2, hidden_dim=emb * 4, output_dim=emb)
        for p in self.sequence_projector_m.parameters():
            p.requires_grad = False

        # Maintain backbone/momentum pairs for easier EMA updates
        self.model_pairs = [
            [self.inter_byte_encoder, self.inter_byte_encoder_m],
            [self.byte_transformer, self.byte_transformer_m],
            [self.byte_attn_pool, self.byte_attn_pool_m],
            [self.packet_projection, self.packet_projection_m],
            [self.packet_transformer, self.packet_transformer_m],
            [self.packet_attn_pool, self.packet_attn_pool_m],
            [self.burst_projection, self.burst_projection_m],
            [self.burst_transformer, self.burst_transformer_m],
            [self.burst_attn_pool, self.burst_attn_pool_m],
            [self.byte_projector, self.byte_projector_m],
            [self.sequence_projector, self.sequence_projector_m],
        ]

    def _setup_queues(self):
        """Create and initialize feature queues (byte_queue and sequence_queue) and the queue pointer"""
        embed_dim = int(self.config.model.emb)
        queue_size = int(getattr(self.config.device, 'queue_size', 65536))
        self.queue_size = queue_size
        # Register as buffers for save/load and correct device placement
        self.register_buffer("byte_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("sequence_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        # Normalize by columns (each column represents one sample feature)
        self.byte_queue = F.normalize(self.byte_queue, dim=0)
        self.sequence_queue = F.normalize(self.sequence_queue, dim=0)
    
    def _update_momentum(self):
        """Update momentum branch parameters using EMA"""
        m = float(getattr(self.config.device, 'momentum', 0.995))
        for q, k in self.model_pairs:
            for param_q, param_k in zip(q.parameters(), k.parameters()):
                param_k.data = param_k.data * m + param_q.data * (1.0 - m)
    
    def _dequeue_and_enqueue(self, byte_feat, sequence_feat):
        """Enqueue current batch momentum features and update circular queues"""
        byte_feat = byte_feat.detach()
        sequence_feat = sequence_feat.detach()
        batch_size = byte_feat.shape[0]
        ptr = int(self.queue_ptr.item())
        assert self.queue_size % batch_size == 0, "queue_size must be divisible by batch_size"
        # Enqueue (store by columns)
        self.byte_queue[:, ptr:ptr + batch_size] = byte_feat.T
        self.sequence_queue[:, ptr:ptr + batch_size] = sequence_feat.T
        # Update pointer
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def contrastive_step(self, inputs, alpha=0.4, update=True):
        """
        Run key steps of contrastive learning: feature extraction, momentum update,
        and computing BSC, BSM, and MLM losses
        """
        # Ensure device is available
        device = next(self.parameters()).device

        # 1. Extract backbone features
        (
            byte_feat, sequence_feat,
            _, _, _,
            packet_tokens, burst_tokens,
            packet_mask, burst_mask,
            byte_tokens, byte_mask
        ) = self._extract_features(inputs, use_momentum=False)

        # 2. In pretrain mode, extract momentum features and prepare soft targets for contrastive learning
        if self.mode == 'pretrain':
            with torch.no_grad():
                self._update_momentum()
                (byte_feat_m, sequence_feat_m, _, _, _, _, _, _, _, _, _) = self._extract_features(inputs, use_momentum=True)

                # --- Soft targets for contrastive loss (BSC) ---
                sim_b2s_m = byte_feat_m @ self.sequence_queue.clone().detach()
                sim_s2b_m = sequence_feat_m @ self.byte_queue.clone().detach()
                sim_b2s_m_batch = byte_feat_m @ sequence_feat_m.T
                sim_s2b_m_batch = sequence_feat_m @ byte_feat_m.T
                sim_b2s_m_full = torch.cat([sim_b2s_m_batch, sim_b2s_m], dim=1)
                sim_s2b_m_full = torch.cat([sim_s2b_m_batch, sim_s2b_m], dim=1)
                sim_targets_b2s = F.softmax(sim_b2s_m_full / self.temp, dim=1)
                sim_targets_s2b = F.softmax(sim_s2b_m_full / self.temp, dim=1)

        # 3. Compute contrastive loss (BSC)
        sim_b2s = byte_feat @ self.sequence_queue.clone().detach()
        sim_s2b = sequence_feat @ self.byte_queue.clone().detach()
        sim_b2s_batch = byte_feat @ sequence_feat.T
        sim_s2b_batch = sequence_feat @ byte_feat.T
        sim_b2s_full = torch.cat([sim_b2s_batch, sim_b2s], dim=1)
        sim_s2b_full = torch.cat([sim_s2b_batch, sim_s2b], dim=1)
        loss_b2s = -torch.sum(F.log_softmax(sim_b2s_full / self.temp, dim=1) * sim_targets_b2s, dim=1).mean()
        loss_s2b = -torch.sum(F.log_softmax(sim_s2b_full / self.temp, dim=1) * sim_targets_s2b, dim=1).mean()
        loss_bsc = (loss_b2s + loss_s2b) / 2

        # 4. BSM (Byte-Sequence Matching) loss
        B = byte_feat.size(0)
        loss_bsm = torch.tensor(0.0, device=device)
    
        if B >= 2:
            # Positive samples
            byte_tokens_pos = byte_tokens
            byte_mask_pos = byte_mask
            sequence_embeds_pos = torch.cat([packet_tokens, burst_tokens], dim=1)
            sequence_mask_pos = torch.cat([packet_mask, burst_mask], dim=1)
    
            # Positive fusion (requires gradients)
            fused_pos = self.fusion_fbert(
                encoder_embeds=byte_tokens_pos, attention_mask=byte_mask_pos,
                encoder_hidden_states=sequence_embeds_pos, encoder_attention_mask=sequence_mask_pos
            )
            fused_pos_pooled = self.byte_attn_pool(fused_pos, byte_mask_pos)
    
            # Turn off gradients only for weights and index sampling
            with torch.no_grad():
                weights_b2s = F.softmax(sim_b2s_m[:, :B], dim=1)  # byte -> sequence
                weights_s2b = F.softmax(sim_s2b_m[:, :B], dim=1)  # sequence -> byte
                eye = torch.eye(B, device=weights_b2s.device, dtype=torch.bool)
                weights_b2s.masked_fill_(eye, 0)
                weights_s2b.masked_fill_(eye, 0)
                hard_neg_s_idx = torch.multinomial(weights_b2s, 1).squeeze(1)  # Sample one negative sequence per byte
                hard_neg_b_idx = torch.multinomial(weights_s2b, 1).squeeze(1)  # Sample one negative byte per sequence
    
            # Build negatives based on sampled indices
            byte_tokens_neg = byte_tokens_pos[hard_neg_b_idx]  # Negative byte for each sequence
            byte_masks_neg = byte_mask_pos[hard_neg_b_idx]
            sequence_embeds_neg = sequence_embeds_pos[hard_neg_s_idx]  # Negative sequence for each byte
            sequence_masks_neg = sequence_mask_pos[hard_neg_s_idx]
    
            # ALBEF logic: construct two types of negatives in one forward pass (interleave query-side and condition-side)
            byte_tokens_all = torch.cat([byte_tokens_pos, byte_tokens_neg], dim=0)
            byte_masks_all = torch.cat([byte_mask_pos, byte_masks_neg], dim=0)
            sequence_embeds_all = torch.cat([sequence_embeds_neg, sequence_embeds_pos], dim=0)
            sequence_masks_all = torch.cat([sequence_masks_neg, sequence_mask_pos], dim=0)
    
            fused_neg_all = self.fusion_fbert(
                encoder_embeds=byte_tokens_all, attention_mask=byte_masks_all,
                encoder_hidden_states=sequence_embeds_all, encoder_attention_mask=sequence_masks_all
            )
            fused_neg_all_pooled = self.byte_attn_pool(fused_neg_all, byte_masks_all)
    
            # Concatenate positives followed by two types of negatives (merged)
            vl_embeddings = torch.cat([fused_pos_pooled, fused_neg_all_pooled], dim=0)  # [3B, emb]
            bsm_logits = self.bsm_head(vl_embeddings)  # [3B, 2]
            bsm_labels = torch.cat([
                torch.ones(B, dtype=torch.long, device=vl_embeddings.device),   # positives B
                torch.zeros(2 * B, dtype=torch.long, device=vl_embeddings.device)  # negatives 2B (two mismatched types merged)
            ], dim=0)
            loss_bsm = F.cross_entropy(bsm_logits, bsm_labels)

            # ===== 4. MLM distillation (enabled only when lambda_mlm > 0) =====
            if self.lambda_mlm > 0.0:
                flow_bytes_id = inputs[11]
                byte_bert_attention_mask = inputs[14]
    
                # Note: currently only the first pad segment per sample is used (consistent with existing code).
                # If you want to expand all pad segments, we can update it.
                input_ids = flow_bytes_id[:, 0, :].clone().to(device)           # [B, L]
                attention_mask = byte_bert_attention_mask[:, 0, :].to(device)   # [B, L]
                labels = input_ids.clone()
    
                probability_matrix = torch.full(labels.shape, self.mlm_probability, device=device)
                input_ids, labels = self.mask(
                    input_ids, self.tokenizer.vocab_size, device,
                    targets=labels, probability_matrix=probability_matrix
                )
    
                # Teacher logits (momentum branch, no gradients)
                with torch.no_grad():
                    teacher_out = self.inter_byte_encoder_m(
                        input_ids=input_ids, attention_mask=attention_mask, return_dict=True
                    )
                    logits_m = teacher_out.logits  # [B, L, V]
    
                # Student forward: obtain CE and logits
                student_out = self.inter_byte_encoder(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True
                )
                logits_s = student_out.logits      # [B, L, V]
                loss_ce = student_out.loss         # Standard MLM hard-label CE
    
                # KD only at masked positions (ignore labels == -100)
                mask_mlm = (labels != -100)        # [B, L]
                if mask_mlm.any():
                    kd_loss = F.kl_div(
                        F.log_softmax(logits_s[mask_mlm], dim=-1),
                        F.softmax(logits_m[mask_mlm], dim=-1),
                        reduction='batchmean'
                    )
                else:
                    kd_loss = logits_s.new_tensor(0.0)
    
                # Blend hard-label CE and soft-label KD by alpha (ALBEF-style)
                loss_mlm = (1.0 - alpha) * loss_ce + alpha * kd_loss
            else:
                # MLM disabled: skip forward and distillation
                loss_mlm = vl_embeddings.new_tensor(0.0)
    
        # ===== 5. Combine total loss and update queues =====
        if update and self.mode == 'pretrain':
            with torch.no_grad():
                self._dequeue_and_enqueue(byte_feat_m, sequence_feat_m)

        total_loss = (self.lambda_bsc * loss_bsc +
                      self.lambda_bsm * loss_bsm +
                      self.lambda_mlm * loss_mlm)
        return total_loss

    def load_pretrained_weights(self, pretrained_path):
        """Load pretrained weights (for fine-tuning)"""
        print(f"Loading pretrained weights from {pretrained_path}")
        pretrained_dict = torch.load(pretrained_path, map_location='cpu')
        
        # Load only matching weights (exclude classifier and momentum branch)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and not k.startswith('classifier') and '_m.' not in k}
        
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=False)
        print(f"Loaded {len(pretrained_dict)} layers from pretrained model")
        
    def freeze_feature_extractors(self):
        """Freeze feature extractors and train classifier only"""
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        # Unfreeze classifier
        for param in self.classifier.parameters():
            param.requires_grad = True
        print("Frozen feature extractors, only classifier will be trained")