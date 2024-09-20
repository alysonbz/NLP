import torch
import torch.nn as nn
import math


class PositionalEncoder(nn.Module):
    def __init__(self, dimensao_modelo, comprimento_maximo=5000):
        super(PositionalEncoder, self).__init__() # a classe herda de nn.Module

        indice_posicao = torch.arange(0, comprimento_maximo).unsqueeze(1)
        termo_divisor_frequencia = torch.exp(
            torch.arange(0, dimensao_modelo, 2) * -(math.log(10000.0) / dimensao_modelo))

        matriz_codificacao_posicional = torch.zeros(comprimento_maximo, dimensao_modelo)
        matriz_codificacao_posicional[:, 0::2] = torch.sin(indice_posicao * termo_divisor_frequencia)
        matriz_codificacao_posicional[:, 1::2] = torch.cos(indice_posicao * termo_divisor_frequencia)

        matriz_codificacao_posicional = matriz_codificacao_posicional.unsqueeze(0)
        self.register_buffer('matriz_codificacao_posicional', matriz_codificacao_posicional)

    def forward(self, tensor_entrada):
        tensor_entrada = tensor_entrada + self.matriz_codificacao_posicional[:, :tensor_entrada.size(1)].to(
            tensor_entrada.device)
        return tensor_entrada