#Segundo parcial High Performance Computing
##CPU v/s GPU
###Multiplicación de matrices utilizando CUDA-Aware MPI y MPI

													Integrantes: Eduardo López

---

#Introducción

Hoy en día existen una infinidad de aplicaciones o implementaciones que requieren de un buen desempeño, muchas de estas a nivel matemático (como por ejemplo algebra lineal). En este caso se estudiará el desempeño de la multiplicación de matrices aplicando técnicas de procesamiento tanto en GPU como también en CPU, utilizando hasta 4 computadores de alto desempeño mediante la conexión a un servidor remoto.

 Para cálculos en CPU se utilizará la especificación para paso de mensajes MPI. Y desarrollo en GPU se utilizará la especificación para paso de mensajes MPI junto a CUDA.
En el presente caso de estudio se presentarán comparaciones tales como:

-	MPI vs MPI + CUDA
-	Rendimiento
-	Aceleración

Donde finalmente para concluir se presentará que técnica es mejor, cuándo y por qué. 

#Desarrollo

Para el presente caso de estudio se han tomado en consideración matrices de tamaños:

-	1000 x 1000
-	2000 x 2000
-	2500 x 2500
-	5000 x 5000
-	10000 x 10000 (este último para términos de cálculo en CPU ha sido omitido)

Antes de comenzar a ver netamente número y obtener conclusiones de estos, se debe entender que se empleara para el desarrollo y obtención de estos.

##MPI
###¿Qué es MPI?

MPI (iniciales de Message Passing Interface), es un estándar para comunicación vía paso de mensajes entre procesos distribuidos, esto es usualmente usado en HPC (High Performance Computing) para construir aplicaciones que escalan en clústeres multi nodos.
MPI es la primera librería de paso de mensajes estándar y portable, especificada por consenso por el MPI Forum, con unas 40 organizaciones participantes, como modelo que permita desarrollar programas que puedan ser migrados a diferentes computadores paralelos. [1]

#####Características básicas de la programación con MPI 

Siguiendo el modelo SPMD, el usuario escribirá su aplicación como un proceso secuencial del que se lanzarán varias instancias que cooperan entre sí. 
Los procesos invocan diferentes funciones MPI que permiten:
 
- iniciar, gestionar y finalizar procesos MPI 
- comunicar datos entre dos procesos 
- realizar operaciones de comunicación entre grupos de procesos
- crear tipos arbitrarios de datos

##CUDA

###¿Qué es CUDA?
CUDA es una arquitectura de cálculo paralelo de NVIDIA que aprovecha la gran potencia de la GPU (unidad de procesamiento gráfico) para proporcionar un incremento extraordinario del rendimiento del sistema.

![](http://puu.sh/sqWeF/e93f71bee3.png)

[Imagen1. Ejemplo de aplicación CUDA.](http://puu.sh/sqWeF/e93f71bee3.png)

Ahora bien, lo realmente importante es cómo podemos obtener un mejor desempeño abstrayendo lo entregado por CUDA y MPI. De esto Nvidia obtuvo lo que se conoce hoy en día como CUDA-Aware MPI.

##CUDA-Aware MPI

###¿En qué consiste CUDA-Aware MPI?

Es una técnica que permite combinar lo implementado con CUDA, es decir GPU y MPI para traspaso de mensajes. Esto comúnmente sirve para:

-	Set de datos muy grandes para caber en memoria de una sola GPU, que requieren tiempo exagerado para computarlos en un único nodo.
-	Acelerar la existencia de aplicaciones MPI con GPU, para permitir que una aplicación multi-GPU de único nodo pueda escalar a múltiples nodos.

Lo esencial de esta combinación es optimizar el paso de mensajes de Host a Device y viceversa.
Antes de combinarse estas 2 técnicas de optimización el paso de mensajes y traspaso de información de CPU a GPU se hacía de la siguiente manera:

######MPI rank 0

cudaMemcpy(s _ buf _ h,s _ buf _ d,size,cudaMemcpyDeviceToHost);

MPI_Send(s _ buf _ h,size,MPI _ CHAR,1,100,MPI _ COMM _ WORLD);

######MPI rank 1

MPI_Recv(r _ buf _ h,size,MPI _ CHAR,0,100,MPI _ COMM _ WORLD, &status);

cudaMemcpy(r _ buf _ d,r _ buf _ h,size,cudaMemcpyHostToDevice);

Donde cudaMemcpy representa el traspaso de información ya sea de CPU a GPU o de GPU a CPU.
Las siguientes versiones permitieron evitar este traspaso y optimizar de este convirtiéndolo solo en una línea de comando familiarizada a MPI.

######MPI rank 0

MPI_Send(s _ buf _ d,size,MPI _ CHAR,1,100,MPI _ COMM _ WORLD);

######MPI rank n-1

MPI_Recv(r _ buf _ d,size,MPI _ CHAR,0,100,MPI _ COMM _ WORLD, &status);

Esto, en pocas palabras, es de que trata CUDA-Aware MPI.

Esta implementación no solo vuelve sencilla trabajar con aplicaciones CUDA+MPI, esto también permite correr a las aplicaciones más eficientemente por dos razones:

-	Todas las operaciones que son requeridas para llevar a cabo la transferencia de mensajes puede ser canalizada (pipeline).
-	Las tecnologías de aceleración como GPUDirect pueden ser utilizadas por las librerías de MPI transparentemente al usuario.


##Caso de estudio

Bueno basta de explicaciones y entremos al caso de estudio como tal. Para esta representación se busca estudiar el desempeño de implementación MPI y CUDA, como también ya se mencionó antes el desarrollo de ambas juntas (CUDA+MPI), de tal manera que quede plasmado su forma de correr aplicaciones de gran esfuerzo computacional como lo son las multiplicaciones de matrices.

Para el presente caso de estudio se evaluará el desempeño de la multiplicación de matrices utilizando hasta 4 nodos remotos.

#####Desarrollo utilizando CUDA + MPI

Para todo el caso de estudio se han utilizado matrices de tamaño cuadrático por lo que el tamaño en las tablas representa fila x columna.

![](http://puu.sh/sqWFT/8bd30cca2e.png)

[Tabla1. Valores obtenidos de procesamiento según tiempo y cantidad de nodos utilizados.](http://puu.sh/sqWFT/8bd30cca2e.png)

#####Grafico procesamiento CUDA+MPI
![](http://puu.sh/sqWSX/c8a868c94e.png)

[Grafico1. Representación gráfica de Tabla1.](http://puu.sh/sqWSX/c8a868c94e.png)

De los valores obtenidos presentados en la Tabla1 se puede destacar que para matrices grandes como por ejemplo la de tamaño 10000x10000 para 2 nodos, ha presentado un mayor tiempo de computo. A pesar de que para matrices más pequeñas el tiempo de computo sea mayor para más nodos, esto tiene una explicación la cual consiste en lo siguiente. Los valores en las matrices son doubles y no enteros, esto quiere decir al momento del traspaso de información mediante MPI o CUDA como también al momento de efectuar la multiplicación, la cantidad de datos que se posee es muy alta ya que un valor entero posee 16 bits en cambio un double posee 64 bits.

También se esperaba que el desempeño fuese mejor a medida que aumentaban los nodos, pero esto depende de muchos factores, siendo el más relevante la implementación. Ya que como se mencionó anteriormente el paso de información durante el proceso de MPI y distribuir el trabajo entre nodos puede ser costoso. Sin embargo, esto ha sido probado hasta un tamaño de 10.000 filas x 10.000 columnas lo que da un total de 100.000.000 elementos y no basta con solo esta información, al igual que sucede con multiplicación de matrices de manera secuencial y en paralelo, la optimización de esta se puede empezar a reflejar con cálculos superiores a 10000x10000 elementos. 

Para detectar de manera correcta el desempeño de esta implementación cabe señalar que es relevante el tiempo de procesamiento durante el envió y recibimiento de información para Cuda-Aware MPI, donde:

#####Tiempo de operación Send

![](http://puu.sh/sqX8W/dd411967c8.png)

[Tabla2.Tiempo de operación Send](http://puu.sh/sqX8W/dd411967c8.png)

#####Tiempo de operación Receive

![](http://puu.sh/sqX9w/ca8a307a1b.png)

[Tabla3.Tiempo de operación Receive](http://puu.sh/sqX9w/ca8a307a1b.png)

En las 2 tablas presentadas anteriormente (véase Tabla2 y Tabla3) se puede notar que tanto como el envió y recibo de información en tiempos es bastante similar, sin embargo, ambas son efectuadas durante todo el proceso, es decir, para la matriz de 10000x10000 utilizando 4 nodos cuyo tiempo total fue de 36,2736 segundos para todo el proceso. Durante todo este proceso, las operaciones de envió y recibo de información han requerido aproximadamente 18 segundos, la mitad del tiempo requerido para el proceso total, en este caso multiplicar las matrices.

###Comparación GPU vs CPU

En este caso se va a tomar en consideración que la implementación en CPU será netamente MPI y GPU lo ya antes presentado.

#####Tiempo de procesamiento para desarrollo utilizando solo MPI

![](http://puu.sh/sqXoi/4bb738847a.png)

[Tabla4.Tiempos de procesamiento utilizando MPI.](http://puu.sh/sqXoi/4bb738847a.png)

Claramente los tiempos de procesamiento serán mayores ya que esta implementación solo posee MPI y no existe trabajo en GPU como lo hace Cuda+MPI. Por lo que el tiempo de procesamiento crece drásticamente y cuyo desarrollo en matrices de tamaño 10000x10000 elementos requiere bastante tiempo.

![](http://puu.sh/sqXoJ/3a4e8ae03f.png)

[Grafico2.. Tiempo procesamiento utilizando MPI.](http://puu.sh/sqXoJ/3a4e8ae03f.png)


####Comparación mediante grafico GPU vs CPU

![](http://puu.sh/sqXpw/180922acef.png)

[Grafico3.Comparación entre procesamiento GPU vs CPU.](http://puu.sh/sqXpw/180922acef.png)

En el Grafico3 se puede destacar el mejor desempeño que se posee al multiplicar matrices hasta un tamaño de 5000x5000 elementos utilizando el procesamiento en GPU.

###Aceleración del procesamiento por hardware

En informática, la aceleración por hardware es el uso del hardware para realizar alguna función más rápido de lo que es posible usando software ejecutándose en una CPU de propósito general.

El hardware que realiza la aceleración, cuando se encuentra en una unidad separada de la CPU, es denominado acelerador por hardware, o a menudo más específicamente como un acelerador gráfico o unidad de coma flotante, etc. Estos términos, sin embargo, son antiguos y se han sustituido por términos menos descriptivos como "placa de vídeo" o "placa gráfica".
Ahora bien, para obtener la aceleración respectiva en este caso de estudio solo basta dividir la velocidad del tiempo de procesamiento en CPU por tiempo de procesamiento en GPU.

![](http://puu.sh/sqXEq/e651d5dd8a.png)

[Tabla5. Aceleración por hardware](http://puu.sh/sqXEq/e651d5dd8a.png)

Cabe destacar que para este procesamiento ha sido considerado TODO el tiempo de procesamiento incluyendo los tiempos obtenidos de “send” y “receive”, tanto para GPU como para CPU.

Como puede observarse en la tabla la aceleración va aumentando a medida que el tamaño de la matriz aumenta, para una mejor percepción véase el siguiente gráfico.

![](http://puu.sh/sqXEQ/6e5fdd7501.png)

[Grafico4.Aceleración por hardware](http://puu.sh/sqXEQ/6e5fdd7501.png)

Como puede observarse en el grafico anterior la aceleración indica que el procesamiento en GPU es bastante satisfactorio para procesamiento de cálculos colosales sin embargo esto depende de que es lo que se desea calcular. 

La GPU hoy en día cumple la mayor parte de sus funcionalidades a nivel de calidad gráfica, todo lo que se puede observar, como gran ejemplo los videojuegos, para cálculos donde el desempeño debe ser realizado a nivel de pixel.


##Conclusiones

MPI es una excelente herramienta y de fácil uso para traspaso de mensajes donde, un proceso puede ser dividido en tareas las cuales podrán ser desarrolladas, en este caso, por múltiples nodos remotos. Dividir estas tareas utilizando MPI simplifica en gran parte el tiempo de procesamiento por ejemplo en la Tabla4; el tiempo mínimo utilizado para la multiplicación de matrices de 5000x5000 elementos ha sido de 39,2259 segundos. Este mismo cálculo de manera secuencial (Véase Imagen6 en Primer Parcial HPC), el tiempo aproximadamente redondea los 700 segundos de procesamiento. Ahora bien, MPI es una herramienta poderosa, pero lo es más aun cuando se trabaja utilizando GPU, aquí es donde entra CUDA cuya finalidad es utilizar los cientos de núcleos que se poseen en GPU para acelerar el poder de cálculo colosal. Donde para el mismo proceso de multiplicar matrices de 5000x5000 elementos utilizando CUDA+MPI, el mejor tiempo obtenido ha sido de 7,32606 segundos, claramente es un mejor tiempo de procesamiento que MPI por si solo y de manera secuencial.

Aun así, no ha sido posible demostrar con claridad el desempeño de utilizar nodos remotos sin embargo está de más decir que los data set utilizados para demostrar su desempeño han sido pequeños para términos de GPU ya que probablemente la optimización de estos recursos se ve reflejada sobre los 100.000.000 de elementos a calcular.

De todas maneras, queda más que claro que CUDA+MPI utilizando este principio de traspaso de mensajes y GPU es mucha más eficiente que solo el traspaso de mensajes MPI.



###Referencias

https://devblogs.nvidia.com/parallelforall/introduction-cuda-aware-mpi/ [1]
http://www.cenits.es/faq/preguntas-generales/que-es-mpi
http://informatica.uv.es/iiguia/ALP/materiales2005/2_2_introMPI.htm
http://www.nvidia.es/object/cuda-parallel-computing-es.html



