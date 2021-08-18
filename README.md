# Redes neuronales recurrentes de alto orden (RHONN) modulo para Python.

_El modulo rhonn.py permite crear arquitecturas de redes neuronales basadas en la arquitectura RHONN, tiene la finalidad de predecir series en el tiempo con ayuda de las mediciones del sistema. Y hace facil su implementacion en un sistema cualquiera programado en Python._

## Comenzando Instalación 🚀

_Estas instrucciones te permitirán obtener una copia del proyecto en funcionamiento en tu máquina local para propósitos de desarrollo y pruebas._

1. Clonar este repositorio.
```
git clone https://github.com/alfredoaguiararce/rhonn-module-python
```
2. Crear un entorno virtual y activarlo.
```
python -m venv venv
source venv/Scripts/activate
```
3. Instalar las dependencias necesarias para el proyecto utilizando pip.
```
pip install -r requirements.txt
```

### Pre-requisitos 📋

_Tener instalada una version de Python puedes descargarlo desde el sitio web oficial en este [link](https://www.python.org/downloads/)._

## Construido con 🛠️

_Herramientas utilizadas para el desarrollo de este modulo._

* [Numpy](https://numpy.org/) - Libreria usada para las operaciones matriciales.
* [pip](https://pip.pypa.io/en/stable/) - Manejador de paquetes en python.

## Ejemplos y modo de empleo.

_Puedes consultar la carpeta 'examples' que contiene ejemplos de como funciona el modulo._

_La manera de utilizar el modulo es la siguiente:_
1. Importamos la libreria.
```
from rhonn import rhonn # Esto importa la arquitectura RHONN.
from rhonn import activation # Esto importa funciones de activacion pre-establecidad.
```
2. Definimos nuestras variables iniciales (esta parde debe ir fuera del proceso iterativo de la simulacion)
```
# Inicializamos las entradas iniciales para W y Z
Z1 = [0 ,0]
W1 = [0 ,0]

# Parametros iniciales para el Filtro Extendido de Kalman.
P1 = 1 * (10**8)
P2 = 1 * (10**2)
P3 = 1 * (10**8)

# Inicializamos nuestro objeto neurona.
neurona_X1 =  rhonn(W1, Z1)
# Inicializamos el filtro de Kalman extendido Interno de la neurona.
neurona_X1.set_ekf(P1, Q1, R1, 0.5) 
```

3.- Durante el proceso iterativo se deben actualizar los valores leidos por la neurona y a su vez enviar el valor medido sobre el cual buscamos aproximar la prediccion.
```
while True:
    # Donde x11[k] representa en este ejemplo el valor actual que deseamos aproximar y x12[k] representa otro parametro del sistema, ambos mediciones del sistema.
    
    # S(x11) , S(x12)
    entradas = [activation.soft_sigmoid(x11[k]), activation.soft_sigmoid(x12[k])]
    # Actualizamos los estados internos de la neurona.
    neurona_X1.update(entradas, x11[k])
    # Obtenemos el valor que predice el modelo.
    prediccion = neurona_X1.predict()
```

_Para mas informacion referente a los parametros, metodo de empleo y funcionamiento del modulo consultar la pagina de documentacion en el siguiente [Link]()._
## Autores ✒️

* **Alfredo Aguiar Arce.** - *Simulacion y programacion.*
* **Dr. Antonio Navarrete Guzman** - *Documentación y asesoria.*

## Licencia 📄

Este proyecto está bajo la Licencia (MIT License) - mira el archivo [LICENSE.md](LICENSE.md) para detalles.

---
Por [Alfredo Aguiar Arce](www.alfredoagrar.com), 2021.