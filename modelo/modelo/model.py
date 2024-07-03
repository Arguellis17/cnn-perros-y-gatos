#Importación de librerias
import tensorflow as tf #pip install tensorflow
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt #pip install matplotlib
from PIL import Image
import os #Esta librería nos permite acceder a las rutas de nuestro sistema operativo

#Para poder trabajar con CNN debemos de definir los layers (capas)
from tensorflow.keras import layers

#Creación de arreglos  
categorias = []
labels = []
imagenes = []

# Especifica la ruta hacia tu dataset
#ruta = "C:\\Users\\argue\\OneDrive\\Escritorio\\datasets\\train"
categorias = os.listdir(ruta)

#Verificar que hayan cargado las categorias
print(categorias)

# Recorrer el directorio de las imágenes
for x, directorio in enumerate(categorias):
    for imagen_nombre in os.listdir(os.path.join(ruta, directorio)):
        # Abrir la imagen y redimensionarla
        imagen = Image.open(os.path.join(ruta, directorio, imagen_nombre)).resize((200, 200)) #Se crea un objeto PIL
        
        # Convertir la imagen a un arreglo numpy y modificar el canal RGB
        imagen = np.array(imagen)
        imagen = imagen[:, :, 0]  # Tomar solo el primer canal (rojo)
        imagen_x = imagen
        # Añadir la imagen al arreglo de imágenes y la etiqueta al arreglo de labels
        imagenes.append(imagen)
        labels.append(x)

# Convertir las listas de imágenes y etiquetas en arreglos de numpy
imagenes = np.array(imagenes)
labels = np.array(labels)

# Verificar las dimensiones de las imágenes y las etiquetas
print("Dimensiones de las imágenes:\n", imagenes.shape)
print("Etiquetas asociadas a las imagenes:\n", labels)

# Visualizar imagen
plt.figure()
plt.imshow(imagenes[5], cmap='gray')
plt.colorbar()
plt.grid(False)
plt.show()

#Creación del modelo sin CNN
#model = tf.keras.models.Sequential([
    #tf.keras.layers.Input(shape=(200, 200)),  # Capa de entrada explícita
    #tf.keras.layers.Flatten(),
    #tf.keras.layers.Dense(128, activation='relu'),
    #tf.keras.layers.Dropout(0.2),
    #tf.keras.layers.Dense(10, activation='softmax')
#])

#Creacion del modelo con CNN
model = tf.keras.models.Sequential([

    # Capa de convolución 2D con 100 filtros, cada uno de tamaño 3x3
    layers.Conv2D(100, (3, 3), activation='relu', input_shape=(200, 200, 1)),
    # Capa de agrupación (pooling) 2D
    layers.MaxPooling2D((2, 2)),
    #Capa que se encarga de apagar neuronas (overfitting)
    layers.Dropout(0,2),
    
    # Capa de convolución 2D con 80 filtros, cada uno de tamaño 3x3
    layers.Conv2D(80, (3, 3), activation='relu'),
    # Capa de agrupación (pooling) 2D
    layers.MaxPooling2D((2, 2)),

    # Capa de convolución 2D con 40 filtros, cada uno de tamaño 3x3
    layers.Conv2D(40, (3, 3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    # Capa de aplanado (flatten)
    layers.Flatten(), #Transforma una matriz tridimensional y la convierte a unidimensional
    #Capa de apagado 
    layers.Dropout(0,5),

    # Capa completamente conectada con 100 neuronas 
    layers.Dense(100, activation='relu'),
    # Capa de salida con 50 unidades y función de activación softmax
    layers.Dense(20, activation='softmax')
])

#Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Crear el numpyarray de los labels
#labels = np.asarray(labels)
#print(labels) 

#Entrenar el modelo
#Al momento de entrenar nuestro modelo, podemos especificar el tamaño del lote (subconjunto de datos) con el
#que trabajara el modelo en cuestión, a traves del atributo 'batch_size = tamaño_del_lote'. Si no se especifica
#el tamaño, el modelo lo tomara con un valor de 'none', en referencia a que no esta limitado.
history = model.fit(imagenes, labels, epochs = 25)

#Especifica la ruta hacia la imagen de prueba
# Evaluar el modelo
#im = Image.open('C:\\Users\\argue\\OneDrive\\Escritorio\\datasets\\test\\1.jpg').resize((200, 200))
im = np.asarray(im)
im = im[:, :, 0]  # Tomar solo el primer canal (rojo)
im = np.expand_dims(im, axis=0)  # Agregar una dimensión que indica que el lote es de una sola imagen 

#En la variable de predicciones, se encuentran las probabilidades de que la imagen pertenezca a una categoria
predicciones = model.predict(im)

#Se imprimen las probabilidades 
print('EXAMPLE',predicciones)

#Dentro del arreglo de las probabilidades, se toma el valor máximo asociado a su categoria, gracias a que esta concatenada con las etiquetas :D
indice_prediccion = np.argmax(predicciones[0])

print('Indice prediccion: ', indice_prediccion)

# Obtener el nombre de la categoría predicha
categoria_predicha = categorias[indice_prediccion]

# Mostrar la imagen
plt.imshow(im[0])
plt.axis('off')  # Ocultar los ejes
plt.title(categoria_predicha)  # Establecer el título como el nombre de la categoría predicha
plt.show()

# Obtener la pérdida y la precisión de entrenamiento de la historia
loss = history.history['loss']
accuracy = history.history['accuracy']

# Crear un rango de épocas para el eje x
epochs = range(1, len(loss) + 1)

# Graficar la pérdida
plt.plot(epochs, loss, 'b', label='Pérdida de entrenamiento')
plt.title('Pérdida de entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

# Graficar la precisión
plt.plot(epochs, accuracy, 'r', label='Precisión de entrenamiento')
plt.title('Precisión de entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.show()

#Exportar modelo: COMANDOS EN CMD
#pip install tensorflowjs
#tensorflowjs_converter --input_format=tf_saved_model "C:\Users\argue\OneDrive\Escritorio\clasificacion\modelo\saved_model" "C:\Users\argue\OneDrive\Escritorio\clasificacion\modelo_js"

#Obtener el resumen del modelo
print(model.summary()),

# ME VA A IR BIEN EN LA EXPO, ME VA A IR BIEN EN LA EXPO, SI TODO SALE BIEN SERÉ EXIMIDO 
