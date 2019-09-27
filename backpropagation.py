import numpy as np
import textwrap as tw

LIMIAR = 0
TAXA_DE_APRENDIZADO = 0.2
NEURONIOS_NA_CAMADA_DE_ENTRADA = 63
NEURONIOS_NA_CAMADA_DE_SAIDA = 7
NEURONIOS_NA_CAMADA_INTERMEDIARIA = 9
EPOCAS = 500
MODULO = int(EPOCAS / 10)


def representacao_bipolar(dados):
    """Recebe uma lista de string e retorna uma matriz com representação bipolar"""
    return np.array(
        [list(map(lambda y: {".": -1, "#": 1}.get(y, 0), x)) for x in dados]
    )


def iniciar_pesos(c0, c1):
    """Inicia os pesos"""
    return 2 * np.random.random((c0, c1)) - 1


def f(x):
    "Foward"
    return (2 / (1 + np.exp(-x))) - 1


def f_lin(x):
    "Backward"
    return (1 / 2) * (1 + x) * (1 - x)


with open("entrada.txt") as arquivo_entrada, open("saida.txt") as arquivo_saida:
    X = representacao_bipolar(arquivo_entrada.read().split())
    y = representacao_bipolar(arquivo_saida.read().split())

pesos0 = iniciar_pesos(NEURONIOS_NA_CAMADA_DE_ENTRADA, NEURONIOS_NA_CAMADA_INTERMEDIARIA)
pesos1 = iniciar_pesos(NEURONIOS_NA_CAMADA_INTERMEDIARIA, NEURONIOS_NA_CAMADA_DE_SAIDA)

for epoca in range(EPOCAS):
    # Propagação
    camada0 = X
    camada1 = f(camada0 @ pesos0)
    camada2 = f(camada1 @ pesos1)

    # Cálculo e retropropagação do erro
    camada2_erro = y - camada2
    camada2_delta = camada2_erro * f_lin(camada2)
    camada1_erro = camada2_delta @ pesos1.T
    camada1_delta = camada1_erro * f_lin(camada1)

    if (epoca % MODULO) == 0:
        print("Erro:", str(np.mean(np.abs(camada2_erro))))

    # Ajuste dos pesos
    pesos1 += camada1.T @ camada2_delta * TAXA_DE_APRENDIZADO
    pesos0 += camada0.T @ camada1_delta * TAXA_DE_APRENDIZADO

with open("teste.txt") as arquivo_teste:
    raw = arquivo_teste.read().split()
    t = representacao_bipolar(raw)

h = f(t @ pesos0)
s = f(h @ pesos1)
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})
for x, saida in enumerate(s):
    print(f"Entrada: {x}")
    print("\n".join(tw.wrap(raw[x], 7)))
    print("Possiveis letras: ", end="")
    for i, letra in enumerate("ABCDEJK"):
        if saida[i] >= LIMIAR:
            print(letra, end=", ")
    print("\n\n")
