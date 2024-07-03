# 🐱🐶 Proyecto: Clasificación de Imágenes de Gatos y Perros con Redes Convolucionales 🐱🐶
Este proyecto consiste en la creación de un dataset propio de imágenes de gatos y perros, así como en la implementación de un modelo de red convolucional en Python para clasificar dichas imágenes. El proyecto se ejecuta de manera local utilizando Visual Studio Code.

## 👀 Vistas del modelo 👀
<div align="center">  
  
[![1.png](https://i.postimg.cc/Jhyb5YBv/1.png)](https://postimg.cc/gwWXYDsH)  
[![2.png](https://i.postimg.cc/MT6z8Nbr/2.png)](https://postimg.cc/yJbMmfGF)
[![3.png](https://i.postimg.cc/qv2r6415/3.png)](https://postimg.cc/18R2LhhG)
</div>

## 📋 Requisitos 📋
- Python 3.7 o superior
- Visual Studio Code
- Bibliotecas de Python:
  - TensorFlow
  - Keras
  - NumPy
  - Matplotlib
  - Pillow
  - Jupyter
## 💻 Instalacion 💻
1. Clonar el repositorio:  
`git clone https://github.com/usuario/proyecto_gatos_perros.git`
2. Instalar las dependencias:
`pip install -r requirements.txt`

## 📸 Preparación del Dataset 📸
Inicialmente, se dispone de 100 imágenes de gatos y 104 imágenes de perros, sumando un total de 204 imágenes. Si se considera necesario, es posible ampliar los conjuntos de datos añadiendo más imágenes. Es importante recordar que en una red convolucional, los aspectos cruciales son la configuración de las capas, el modelo y el proceso de entrenamiento.

Para agregar más imágenes, simplemente coloque las imágenes de perros en la carpeta "dataset/dog" y las imágenes de gatos en la carpeta "dataset/cat".

## 🎯 Uso del Modelo 🎯
Es fundamental ajustar las rutas del dataset y de las imágenes de prueba según la ubicación donde hayas descargado este proyecto. El modelo está entrenado y configurado para redimensionar automáticamente las imágenes añadidas, por lo que no es necesario limitarse a buscar imágenes con dimensiones específicas.

El redimensionamiento utilizado fue de 200 px x 200 px, lo cual se considera ideal. Además, los canales se han trabajado en blanco y negro para reducir la carga de datos en los arrays. Adicionalmente, se generan gráficos que muestran la precisión del entrenamiento y la pérdida durante el entrenamiento. Se ha utilizado la técnica DropOut para evitar el sobreajuste (overfitting) en los resultados del modelo y la función de activación ReLU.

## 📄 Licencia 📄
Este proyecto está bajo la Licencia MIT. Fue presentado en la Universidad Francisco de Paula Santander, en la asignatura electiva profesional "Inteligencia Artificial". 
