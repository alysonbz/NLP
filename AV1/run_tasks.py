from classifier import experimento_item_A, experimento_item_B, experimento_item_C
from preprocessing import ler_datasets

train, test, validation = ler_datasets()

print("Experimento Item A:")
resultados_A = experimento_item_A(train, test, "cohesion")
print(resultados_A)

print("\nExperimento Item B:")
resultados_B = experimento_item_B(train, test, "cohesion")

print(resultados_B)

print("\nExperimento Item C:")
resultados_C = experimento_item_C(train, test, "cohesion")

print(resultados_C)