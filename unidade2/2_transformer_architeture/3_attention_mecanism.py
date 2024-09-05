import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, dimensao_modelo, numero_cabecas):
        super(MultiHeadAttention, self).__init__()

        assert dimensao_modelo % numero_cabecas == 0, "dimensao_modelo deve ser divisível pelo número de cabeças"

        self.dimensao_cabeca = dimensao_modelo // numero_cabecas
        self.numero_cabecas = numero_cabecas

        # Camadas lineares para criar Q, K, V
        self.camada_linear_q = nn.Linear(dimensao_modelo, dimensao_modelo)
        self.camada_linear_k = nn.Linear(dimensao_modelo, dimensao_modelo)
        self.camada_linear_v = nn.Linear(dimensao_modelo, dimensao_modelo)

        self.camada_linear_saida = nn.Linear(dimensao_modelo, dimensao_modelo)

        self.dropout = nn.Dropout(0.1)

    def dividir_cabecas(self, tensor_entrada):
        tamanho_batch, comprimento_sequencia, dimensao_modelo = tensor_entrada.size()
        tensor_entrada = tensor_entrada.view(tamanho_batch, comprimento_sequencia, self.numero_cabecas,
                                             self.dimensao_cabeça)
        tensor_entrada = tensor_entrada.permute(0, 2, 1, 3)
        return tensor_entrada

    def calcular_atencao(self, q, k, v, mascara=None):
        pontuacoes = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dimensao_cabeca)

        if mascara is not None:
            pontuacoes = pontuacoes.masked_fill(mascara == 0, float('-inf'))

        pesos_atencao = torch.nn.functional.softmax(pontuacoes, dim=-1)
        pesos_atencao = self.dropout(pesos_atencao)

        saida = torch.matmul(pesos_atencao, v)
        return saida, pesos_atencao

    def forward(self, tensor_entrada, mascara=None):
        # Passa a entrada pelas camadas lineares para obter Q, K e V
        q = self.camada_linear_q(tensor_entrada)
        k = self.camada_linear_k(tensor_entrada)
        v = self.camada_linear_v(tensor_entrada)

        # Divide Q, K e V em cabeças de atenção
        q = self.dividir_cabecas(q)
        k = self.dividir_cabecas(k)
        v = self.dividir_cabecas(v)

        saida_atencao, pesos_atencao = self.calcular_atencao(q, k, v, mascara)

        tamanho_batch, numero_cabecas, comprimento_sequencia, dimensao_cabeca = saida_atencao.size()
        saida_atencao = saida_atencao.permute(0, 2, 1, 3).contiguous()
        saida_atencao = saida_atencao.view(tamanho_batch, comprimento_sequencia,
                                           numero_cabecas * dimensao_cabeca)

        saida = self.camada_linear_saida(saida_atencao)
        return saida, pesos_atencao
