from NLP.unidade2.attention_mecanism import MultiHeadAttention
import torch.nn as nn


class EncoderLayer(nn.Module):
    def __init__(self, dimensao_modelo, numero_cabecas, tamanho_feed_forward, taxa_dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.atencao_multi_cabeca = MultiHeadAttention(dimensao_modelo, numero_cabecas)

        self.camada_feed_forward = nn.Sequential(
            nn.Linear(dimensao_modelo, tamanho_feed_forward),
            nn.ReLU(),
            nn.Linear(tamanho_feed_forward, dimensao_modelo)
        )

        self.normalizacao_atencao = nn.LayerNorm(dimensao_modelo)
        self.normalizacao_feed_forward = nn.LayerNorm(dimensao_modelo)

        self.dropout = nn.Dropout(taxa_dropout)

    def forward(self, tensor_entrada, mascara=None):
        saida_atencao, pesos_atencao = self.atencao_multi_cabeca(tensor_entrada, mascara)
        tensor_entrada = self.normalizacao_atencao(tensor_entrada + self.dropout(saida_atencao))

        saida_feed_forward = self.camada_feed_forward(tensor_entrada)
        saida_final = self.normalizacao_feed_forward(tensor_entrada + self.dropout(saida_feed_forward))

        return saida_final, pesos_atencao