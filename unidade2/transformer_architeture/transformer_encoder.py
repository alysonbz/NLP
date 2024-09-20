from NLP.unidade2.4_encoder_layer import EncoderLayer
import torch.nn as nn


class TransformerEncoder(nn.Module):
    def __init__(self, dimensao_modelo, numero_cabecas, tamanho_feed_forward, numero_camadas, taxa_dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.camadas_encoder = nn.ModuleList(
            [EncoderLayer(dimensao_modelo, numero_cabecas, tamanho_feed_forward, taxa_dropout)
             for _ in range(numero_camadas)]
        )

        self.normalizacao_final = nn.LayerNorm(dimensao_modelo)

    def forward(self, tensor_entrada, mascara=None):
        for camada_encoder in self.camadas_encoder:
            tensor_entrada, pesos_atencao = camada_encoder(tensor_entrada, mascara)

        saida_final = self.normalizacao_final(tensor_entrada)
        return saida_final