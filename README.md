<h1 align="center">Sistemas de Inteligencia Artificial</h1>
<h3 align="center">TP3: Perceptrón Simple y Multicapa</h3>
<h4 align="center">Primer cuatrimestre 2025</h4>

# Requisitos

* Python ([versión 3.12.9](https://www.python.org/downloads/release/python-3129/))
* [UV](https://docs.astral.sh/uv/getting-started/installation/)

# Instalando las dependencias

```bash
# Si python 3.12.9 no esta instalado se puede instalar haciendo
uv python install 3.12.9

# Para crear y activar el entorno virtual
uv venv
source .venv/bin/activate  # En Unix
.venv\Scripts\activate     # En Windows

# Para instalar las dependencias
uv sync
```

# Corriendo el proyecto

El proyecto consta de varios archivos _frontend_ con el cuál se puede demostrar
el funcionamiento de las diferentes implementaciones de perceptrones. Sin
embargo, en cada consigna, los archivos importantes son los que tienen el
formato `ejXY.py` donde, `X` es el número de ejercicio e `Y` el inciso del
mismo.

* Para el **ejercicio 1** se demuestra el funcionamiento de perceptrones simples
  con función de activación de tipo _escalón_, para intentar clasificar
funciones lógicas de `and` y `xor`.

* Para el **ejercicio 2**, se compara el funcionamiento de perceptrones simples
  lineales y no lineales para intentar entrenarlo con un set de datos dado.

* Para el **ejercicio 3**, se demuestra el funcionamiento de perceptrones
multicapa que permite el entrenamiento para tareas más complejas, como la
clasificación de la tabla de verdad de `xor`, la discriminación por paridad de
números, o el reconocimiento de números por imágenes.

Además de estos, se encuentran múltiples archivos de _análisis_ que permiten
demostrar la variación de funcionamiento de cada elemento con método como _cross
validation_, variación de _learning rates_, implementación de optimizadores,
entre otros.

Para ejecutar cada archivo, simplemente correr:

```bash
uv run ejXY.py
```


# Utilizando el motor de perceptrón

Si se desea crear un propio _frontend_ se puede utilizar los diferentes
perceptrones diseñados. Para estos se declararon múltiples clases `Perceptron`
que reciben diferentes parámetros.

```python
from src.perceptron import PerceptronSimple, PerceptronLineal, PerceptronNoLineal, PerceptronMulticapa

# Para el simple
perceptron = PerceptronSimple(input_size, tita, learning_rate)

# Para el lineal
perceptron = PerceptronLineal(input_size, learning_rate)

# Para el no lineal
perceptron = PerceptronNoLineal(input_size, tita, tita_prime, learning_rate)

# Para el multicapa
perceptron = PerceptronMulticapa(capas, tita, tita_prime, optimizer)
```

Donde:

* `tita`: es la función de activación
* `tita_prime`: es la derivada de la función _tita_
* `learning_rate`: es el ratio de aprendizaje que utilizará el perceptrón
* `capas`: es un array de la forma `[i, h1, h2, ..., hn, o]`, que especifica la
  cantidad de entradas de cada capa en orden (entrada, oculta 1, oculta 2, ...,
oculta n, salida)

> **NOTA:** Para el perceptrón multicapa, el valor que se le pasa a
> `tita_prime`, ya es el valor de `tita(h)`, por lo tanto, al definir la
> derivada, asumir que el valor de entrada ya es el mencionado.

Luego, para entrenar al perceptrón se le pasan los datos:

```python
result = perceptron.train(data, labels, epochs, tolerance)
```

Donde:

* `data`: son los valores de entrada
* `labels`: los valores reales o esperados para cada entrada
* `epochs`: es la cantidad máxima de epocas que entrena el perceptrón si no
alcanza la convergencia.
* `tolerance`: es el valor mínimo de error que tiene que alcanzar el
entrenamiento para corte el entrenamiento.

El entrenamiento devuelve `result` que es un diccionario con la información de
los pesos y el error calculado en cada epoca, permitiendo realizar gráficos de
evolución temporal.

En el caso del perceptrón multicapa, por una cuestión del tamaño de la red, solo
guardamos el error calculado por época.

Una vez entrenado el perceptrón, podemos predecir el resultado para diferentes
valores de testeo realizando:

```python
prediction = perceptron.predict(x)
```

Donde:

* `x` es el valor de entrada de testeo

# Extra: generador de imágenes

Para generar las imágenes utilizadas para el entrenamiento del perceptrón
multicapa, realizamos un script, que, utilizando `pillow` genera imágenes de
números, variando su posición, tamaño y rotación en el canvas y hasta
agregandole ruido.

Para correr dicho script se especifica la cantidad de imágenes de training por
número y la cantidad de imágenes de testing por número. Los números van del 0 al
9, por lo que si se corre con 10 imágenes de training se generarán 100 imágenes
en total.

```bash
uv run generate_number_images.py train_size test_size
```
