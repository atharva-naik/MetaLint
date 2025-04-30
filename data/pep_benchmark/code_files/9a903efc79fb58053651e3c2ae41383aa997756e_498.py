'''
Exercício 01

- Solicite ao usuário 10 números inteiros e armazene-os em um arquivo de texto.
'''
print('Digite 10 números inteiros: ')
lista = []
for num in range(10):
    num = int(input())
    lista.append(num)

lista.sort()
arquivo_ex_01 = open('C:\\Users\\USUARIO\\Desktop\\JuNiNhOoOo\\Exercícios Python\\Python IMPACTA\\Atividades-Linguagem_II\\Persistências e arquivos\\arquivo_ex_01.txt', 'w')

for num in lista:
    arquivo_ex_01.write(str(num) + '\n')
