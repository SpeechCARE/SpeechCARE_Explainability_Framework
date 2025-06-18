import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as transforms
import torch.nn.functional as F

from transformers import (AutoModel,
                          AutoTokenizer,
                          AutoModelForSpeechSeq2Seq,
                          AutoProcessor, pipeline)
from transformers import WhisperProcessor, WhisperModel

from utils.dataset_utils import preprocess_audio


class MultiHeadAttentionAddNorm(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout):
        super(MultiHeadAttentionAddNorm, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-Head Attention
        attn_output, _ = self.mha(x, x, x)  # Self-attention: Q = K = V = x
        # Add & Norm
        x = self.norm(x + self.dropout(attn_output))
        return x


class GatingNetwork(nn.Module):
    def __init__(self, input_dim):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, 3)  # Output 3 weights for speech, text, and demography
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        gate_weights = self.fc(x)
        return self.softmax(gate_weights)  # Ensure weights sum to 1


class TBNet(nn.Module):
    def __init__(self, config,speech_only=False):
        super(TBNet, self).__init__()
        self.config = config
        self.predicted_label = None
        self.transcription = None

        # Initialize speech encoder
        if config.speech_transformer_chp == config.mHuBERT:
            self.speech_transformer = AutoModel.from_pretrained(config.speech_transformer_chp)
        elif config.speech_transformer_chp == config.WHISPER:
            self.speech_transformer = WhisperModel.from_pretrained(config.speech_transformer_chp)
        speech_embedding_dim = self.speech_transformer.config.hidden_size

        # CLS token and positional encoding
        self.cls_token = nn.Parameter(torch.randn(1, 1, speech_embedding_dim))
        max_seq_length = int(config.segment_size / 0.02) + 1 # +1 for CLS embedding
        self.positional_encoding = nn.Parameter(torch.randn(1, max_seq_length, speech_embedding_dim))

        # Transformer layers
        num_layers = 2
        self.layers = nn.ModuleList([
            MultiHeadAttentionAddNorm(speech_embedding_dim, 4, 0.1)
            for _ in range(num_layers)
        ])

        # Projection heads
        self.speech_head = nn.Sequential(
            nn.Linear(speech_embedding_dim, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, config.num_labels)
        )


        if not speech_only:
            max_seq_length = int(config.max_num_segments * ((config.segment_size / 0.02) - 1)) + 1
            self.positional_encoding = nn.Parameter(torch.randn(1, max_seq_length, speech_embedding_dim))

            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(config.txt_transformer_chp, trust_remote_code=True)

            # Initialize text encoder
            self.txt_transformer = AutoModel.from_pretrained(config.txt_transformer_chp, trust_remote_code=True)
            txt_embedding_dim = self.txt_transformer.config.hidden_size
        
            self.txt_head = nn.Sequential(
                nn.Linear(txt_embedding_dim, config.hidden_size),
                nn.Tanh(),
            )
            self.demography_head = nn.Sequential(
                nn.Linear(1, config.demography_hidden_size),
                nn.Tanh(),
            )
            self.speech_head = nn.Sequential(
                nn.Linear(speech_embedding_dim, config.hidden_size),
                nn.Tanh(),
            )
            self.speech_classifier = nn.Linear(config.hidden_size, config.num_labels)
            self.txt_classifier = nn.Linear(config.hidden_size, config.num_labels)
            self.demography_classifier = nn.Linear(config.demography_hidden_size, config.num_labels)
            self.weight_gate = GatingNetwork((config.hidden_size * 2) + config.demography_hidden_size)

        self._init_whisper_pipeline()

        self.labels = ['control', 'mci', 'adrd']
        self.label_map = {'control':0, 'mci':1, 'adrd':2}
        self.label_rev_map = {0:'control', 1:'mci', 2:'adrd'}

    def _init_whisper_pipeline(self):
        """Initialize Whisper pipeline for transcription"""
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.config.WHISPER,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        whisper_model.to(self.device)

        processor = AutoProcessor.from_pretrained(self.config.WHISPER)

        self.whisper_pipeline = pipeline(
            "automatic-speech-recognition",
            model=whisper_model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=self.device,
        )

    def forward(self, input_values, input_ids, demography, attention_mask):
        # Process speech modality
        batch_size, num_segments, seq_length = input_values.size()
        input_values = input_values.view(batch_size * num_segments, seq_length)

        if self.config.speech_transformer_chp == self.config.mHuBERT:
            speech_embeddings = self.speech_transformer(input_values)
        elif self.config.speech_transformer_chp == self.config.WHISPER:
            speech_embeddings = self.speech_transformer.encode(input_values)

        speech_embeddings = speech_embeddings.view(batch_size, num_segments, -1, speech_embeddings.size(-1))
        speech_embeddings = speech_embeddings.view(batch_size, num_segments * speech_embeddings.size(2), -1)

        # Add CLS token and positional encoding
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        speech_embeddings = torch.cat((cls_tokens, speech_embeddings), dim=1)
        speech_embeddings += self.positional_encoding[:, :speech_embeddings.size(1), :]

        # Transformer layers
        for layer in self.layers:
            speech_embeddings = layer(speech_embeddings)

        speech_cls = speech_embeddings[:, 0, :]

        # Process text modality
        txt_embeddings = self.txt_transformer(input_ids=input_ids, attention_mask=attention_mask)
        txt_cls = txt_embeddings.last_hidden_state[:, 0, :]

        # Process demography
        demography = demography.unsqueeze(1)
        demography_x = self.demography_head(demography.squeeze(-1))
        demography_x = demography_x.unsqueeze(1).squeeze(1)

        # Project modalities
        speech_x = self.speech_head(speech_cls)
        txt_x = self.txt_head(txt_cls)

        # Gated fusion
        gate_weights = self.weight_gate(torch.cat([speech_x, txt_x, demography_x], dim=1))
        self.last_gate_weights = gate_weights.detach().cpu().numpy()
        weight_speech, weight_txt, weight_demography = gate_weights[:, 0], gate_weights[:, 1], gate_weights[:, 2]

        # Classify each modality
        speech_out = self.speech_classifier(speech_x)
        txt_out = self.txt_classifier(txt_x)
        demography_out = self.demography_classifier(demography_x)

        # Fuse outputs
        fused_output = (
            weight_speech.unsqueeze(1) * speech_out +
            weight_txt.unsqueeze(1) * txt_out +
            weight_demography.unsqueeze(1) * demography_out
        )

        probabilities = F.softmax(fused_output, dim=1)

        return fused_output, probabilities

    def speech_only_forward(self, input_values, return_embeddings=False, return_features=False):
        device = next(self.parameters()).device
        needs_grad = input_values.requires_grad

        if input_values.dim() == 2:
            batch_size = 1
            num_segments, seq_length = input_values.size()
            input_values = input_values.reshape(batch_size, num_segments, seq_length)  # Preserves gradients
        else:
            batch_size, num_segments, seq_length = input_values.size()
            input_values = input_values.reshape(batch_size * num_segments, seq_length)  # Preserves gradients

        if needs_grad:
            input_values = input_values.requires_grad_(True).to(device)        

        # Get transformer output
        if self.config.speech_transformer_chp == self.config.mHuBERT:
            speech_embeddings = self.speech_transformer(input_values)
        elif self.config.speech_transformer_chp == self.config.WHISPER:
            speech_embeddings = self.speech_transformer.encode(input_values)

        output_embeddings = speech_embeddings.last_hidden_state  # (batch*segments, seq_len, hidden_dim)
        output_embeddings = output_embeddings.view(batch_size, -1, output_embeddings.size(-1))



        # Add CLS token and positional encoding
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, output_embeddings), dim=1)
        embeddings += self.positional_encoding[:, :embeddings.size(1), :]


        # Pass through layers (store final features)
        features = embeddings
        for layer in self.layers:
            features = layer(features)

        # Classification head
        cls = features[:, 0, :]  # CLS token
        x = self.speech_head(cls)
        # x = self.speech_classifier(x)


        if return_embeddings:
            return x, speech_embeddings
        if return_features:
            return x, features  # Return logits + feature maps
        return x

    def speech_only_inference(self, input_values, device='cuda'):
        self.eval()
        self.to(self.device)


        with torch.no_grad():
            predictions, embeddings = self.speech_only_forward(input_values, return_embeddings=True)

        return {
            "predictions": predictions.cpu().numpy(),
            "embeddings": embeddings.cpu().numpy(),
            "segments_tensor": input_values.squeeze(0).cpu().numpy()
        }
    def preprocess_data(self,audio_path,segment_length,demography_info, overlap=0.2, target_sr=16000):
        audio_path = str(audio_path)

        print("Transcribing audio...")
        transcription_result = self.whisper_pipeline(audio_path)
        self.transcription = transcription_result["text"]
        print(f"Transcription: {self.transcription}")
        print("Tokenizing transcription...")
        tokenized_text = self.tokenizer(self.transcription, return_tensors="pt", padding=True, truncation=True)
        input_ids = tokenized_text["input_ids"]
        attention_mask = tokenized_text["attention_mask"]

        print("Preprocessing audio...")
        if self.config.speech_transformer_chp == self.config.mHuBERT:
            processor= None
        elif self.config.speech_transformer_chp == self.config.WHISPER:
            processor= WhisperProcessor.from_pretrained(self.config.speech_transformer_chp)
        input_values = preprocess_audio(audio_path, processor,segment_length=segment_length, overlap=overlap,target_sr=target_sr)

        demography_tensor = torch.tensor([demography_info], dtype=torch.float32).unsqueeze(0)

        return input_values,input_ids,attention_mask,demography_tensor

    def inference(self, input_values,input_ids,attention_mask,demography_tensor, config):

        device = next(self.parameters()).device
        input_values = input_values.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        demography_tensor = demography_tensor.to(device)

        print("Running inference...")
        with torch.no_grad():
            logits, probabilities = self(input_values, input_ids, demography_tensor, attention_mask)

        predicted_label = torch.argmax(logits, dim=1).item()
        self.predicted_label = predicted_label

        return predicted_label, probabilities[0].tolist()

    def text_only_classification(self, input_ids, attention_mask):
        txt_embeddings = self.txt_transformer(input_ids=input_ids, attention_mask=attention_mask)
        txt_cls = txt_embeddings.last_hidden_state[:, 0, :]
        txt_x = self.txt_head(txt_cls)
        txt_out = self.txt_classifier(txt_x)
        return txt_out