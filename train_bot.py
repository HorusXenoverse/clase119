# Biblioteca de preprocesamiento de datos de texto
import nltk
nltk.download('punkt')

# importamos la Función para obtener palabras raíz (derivación)
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

import json
# convierte matrices o listas en diccionarios (pickle)
import pickle

#Es el equivalente a la biblioteca math
import numpy as np

words=[]
classes = []
word_tags_list = []
ignore_words = ['?', '!',',','.', "'s", "'m"]
train_data_file = open('intents.json').read()
intents = json.loads(train_data_file)

# Función para añadir palabras raíz (stem words)
def get_stem_words(words, ignore_words):
    stem_words = []
    for word in words:
        if word not in ignore_words:
            w = stemmer.stem(word.lower())
            stem_words.append(w)  
    return stem_words

for intent in intents['intents']:
    
        # Agregar todas las palabras de los patrones a una lista
        for pattern in intent['patterns']:            
            pattern_word = nltk.word_tokenize(pattern)            
            words.extend(pattern_word)                      
            word_tags_list.append((pattern_word, intent['tag']))
        # Agregar todas las etiquetas a la lista de clases
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            stem_words = get_stem_words(words, ignore_words)
print("Palaras Raíz")
print(stem_words)
print("Lista de parejas")
print(word_tags_list[0]) 
print("Clases o etiquetas")
print(classes)   

# Crear un corpus de palabras para el chatbot
def create_bot_corpus(stem_words, classes):

    #La función "sorted" ordena las listas de palabras y de clases alfabéticamente
    stem_words = sorted(list(set(stem_words)))
    classes = sorted(list(set(classes)))

    #Crea 2 documentos para guardar las listas de palabras y clases ordenadas en forma binaria
    pickle.dump(stem_words, open('words.pkl','wb'))
    pickle.dump(classes, open('classes.pkl','wb'))

    return stem_words, classes

stem_words, classes = create_bot_corpus(stem_words,classes)  

print("Palabras raíz alfabéticamente")
print(stem_words)
print("Clases o etiquetas alfabéticamente")
print(classes)

# Crear una bolsa de palabras
words_bag = []
class_aviable = len(classes)
labels = [0] * class_aviable

#ciclo for para recorrer la lista de parejas, oración-etiqueta(word_tags_list)
for bag_list in word_tags_list:
    #Matriz para crear guardar la bolsa de palabras de cada frase
    bag_words = []
    #pattern solo guarda la frase de cada pareja
    pattern = bag_list[0]
    #Ciclo for para recorrer las frases y encontrar palabras raíz
    for bag_list2 in pattern:
        #Es para obtener el índice de cada elemento
        index = pattern.index(bag_list2)
        #Es para obtener la palabra raíz y el lower para convertirla en minúscula
        bag_list2 = stemmer.stem(bag_list2.lower())
        #En cada palabra le asigna la palabra raíz q le corresponde
        pattern[index] = bag_list2
    #Recorre para ver si están las palabras precentes en las palabras raíz si está agrega un 1 de lo contrario agrega un 0
    for stemWord in stem_words:
        if stemWord in pattern:
            bag_words.append(1)
        else:
            bag_words.append(0)
    print("Imprime cada bolsita de palabras")
    print(bag_words)

    #Variable para guardar la lista q inicialmente declaramos con 0
    labels_encoding = list(labels)
    #Variable para guardar solo las etiquetas
    tag = bag_list[1]
    #Variable para obtener el índice de cada etiqueta
    tag_index = classes.index(tag)
    #Agrega un 1 a la lista de 0 según la etiqueta q es
    labels_encoding[tag_index] = 1
    #Añade la  bolsa de las palaras y la lista de etiquetas
    words_bag.append([bag_words, labels_encoding])
print("Imprime el primer elemento de la bolsa suprema")
print(words_bag[0])

# Crear datos de entrenamiento
def preprocess(words_bag):
    matriz = np.array(words_bag, dtype = object)
    matriz_words = list(matriz[:,0])
    matriz_tag = list(matriz[:,1])
    print(matriz_words[0])
    print(matriz_tag[0])
    return matriz_words, matriz_tag

matriz_words, matriz_tag = preprocess(words_bag)
