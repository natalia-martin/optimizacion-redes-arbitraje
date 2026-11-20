"""
Laboratorio: Caminos mínimos — Dijkstra y Bellman-Ford
UCM — Flujos en Redes (3.º Matemáticas)

Completad las dos funciones. No modificar helpers_caminos.py.
Ejecutad test_env_caminos.py para comprobar los resultados.
"""

from typing import Dict, List, Tuple, Optional
from helpers_caminos import reconstruct_path, get_all_nodes


# ─────────────────────────────────────────────────────────────────
# FUNCIÓN 1: DIJKSTRA
# ─────────────────────────────────────────────────────────────────

def dijkstra(
    graph: Dict[str, Dict[str, float]],
    source: str,
    target: str,
) -> dict:
    """
    Algoritmo de Dijkstra (conjuntos P y T).

    Parámetros
    ----------
    graph  : diccionario de adyacencia {nodo: {vecino: peso, ...}, ...}
    source : nodo origen (str)
    target : nodo destino (str)

    Devuelve
    --------
    dict con claves:
        'distances'    : dict {str: float} — distancia mínima desde source
        'predecessors' : dict {str: str o None} — predecesor de cada nodo
        'path'         : list[str] — camino [source, ..., target], o None
        'steps'        : list[dict] — un dict por cada vértice que pasa a P:
                           'current'      : str  — vértice recién confirmado
                           'distances'    : dict — copia de dist en este instante
                           'predecessors' : dict — copia de prev
                           'visited'      : set  — copia de P (permanentes)

    Notas
    -----
    - Inicialización: P = {source}, dist[source] = 0.
      Para cada j ≠ source: dist[j] = peso del arco (source, j) si existe, +∞ si no.
    - En cada iteración: k = min dist[j] para j en T, pasar k a P,
      actualizar vecinos de k que estén en T.
    - get_all_nodes(graph) devuelve la lista de todos los vértices del grafo.
    - IMPORTANTE: hacer copia (dict(...), set(...)) al registrar cada step.
    """
    V=set(get_all_nodes(graph)) #Hallamos todos los vértices del grafo.
    #Inicializamos: distancia infinita y sin predecesores.
    dist={j: float('inf') for j in V} 
    prev={j: None for j in V}
    dist[source]=0 #El origen tiene distancia cero.

    #Si el origen tiene algun arco que sale de él, sus vecinos directos
    # toman el peso del arco
    if source in graph:
        for j,w in graph[source].items():
            dist[j]=w
            prev[j]=source
    # Creamos los dos conjuntos de vértices:
    # P(permanentes): tienen distancia mínima ya confirmada. Incluirá al
    # origen en primer lugar.
    # T(transitorios): con distancia pendiente de confirmar, serán el resto.
    P={source}
    T=V-P
    steps=[] #Lista para guardar los pasos del algoritmo.
    while T:
        k=min(T, key=lambda j:dist[j]) #Elegimos el nodo con menor distancia.
        P.add(k) #Añadimos k a los permanentes.
        T.discard(k) #Quitamos k de los transitorios.
        # Registramos el paso para la animación:
        paso={} #Creamos diccionario vacío.
        paso['current']=k #Guardamos el vértice recién confirmado. 
        paso['distances']=dict(dist) #Guardamos una copia del diccionario de distancias.
        paso['predecessors']=dict(prev) #Guardamos una copia del diccionario de predecesores.
        paso['visited']=set(P) #Guardamos una copia del conjunto P
        steps.append(paso) #Añadimos el diccionario paso a la lista steps.
        # Si el vértice que acabamos de encontrar es el destino, ya hemos encontrado la
        # distancia mínima y paramos.
        if k==target:
            break
        # Si la distancia mínima es infinito, no se puede llegar a los demás vértices desde 
        # el origen y paramos
        if dist[k]==float('inf'):
            break
        # Recorremos los arcos que salen de k para intentar mejorar la distancia de sus 
        # vecinos que siguen en transitorios. 
        if k in graph:
            for j, w in graph[k].items():
                # Si pasar por k da un camino más corto hasta j, actualizamos la 
                # distancia y su predecesor
                if j in T and dist[k]+w<dist[j]:
                    dist[j]=dist[k]+w
                    prev[j]=k
    # Usamos la función de helpers reconstruct_path que reconstruye el camino de
    # source a target a partir del diccionario de predecesores.               
    path=reconstruct_path(prev, source, target)
    # La función dijkstra devolverá las distancias, los predecesores, el camino 
    # óptimo y los pasos para la animación.
    return{
            'distances':dist,
            'predecessors':prev,
            'path':path,
            'steps':steps,
        }

# ─────────────────────────────────────────────────────────────────
# FUNCIÓN 2: BELLMAN-FORD
# ─────────────────────────────────────────────────────────────────

def bellman_ford(
    graph: Dict[str, Dict[str, float]],
    source: str,
    target: str,
) -> dict:
    """
    Algoritmo de Bellman-Ford con detección de circuitos negativos.

    Parámetros
    ----------
    graph  : diccionario de adyacencia {nodo: {vecino: peso, ...}, ...}
    source : nodo origen (str)
    target : nodo destino (str), puede ser None para single-source

    Devuelve
    --------
    dict con claves:
        'distances'          : dict {str: float} — distancia mínima desde source
        'predecessors'       : dict {str: str o None} — predecesor de cada nodo
        'path'               : list[str] — camino [source, ..., target], o None
        'steps'              : list[dict] — un dict por cada pasada completa:
                                 'iteration'     : int  — número de pasada (1, 2, ...)
                                 'distances'     : dict — copia de dist tras la pasada
                                 'predecessors'  : dict — copia de prev
                                 'updated_edges' : list[tuple[str,str]] — arcos mejorados
        'has_negative_cycle' : bool — True si se detecta un circuito negativo

    Notas
    -----
    - Inicialización: dist[source] = 0.
      Para cada j ≠ source: dist[j] = peso del arco (source, j) si existe, +∞ si no.
      Esto cubre los caminos de 1 arco.
    - Después: |V|-2 pasadas recorriendo TODOS los arcos del grafo.
    - Convergencia anticipada: si ninguna distancia mejora en una pasada, parar.
    - Detección de circuito negativo: una pasada extra tras las regulares.
    - get_all_nodes(graph) devuelve la lista de todos los vértices del grafo.
    - IMPORTANTE: hacer copia (dict(...)) al registrar cada step.
    """
    #La inicialización es similar a la de Dijkstra
    V=set(get_all_nodes(graph))
    dist={j: float('inf') for j in V}
    prev={j: None for j in V}
    dist[source]=0
    if source in graph:
        for j,w in graph[source].items():
            dist[j]=w
            prev[j]=source
    steps=[]
    #Hacemos len(V)-2 iteraciones
    for t in range(1, len(V)-1):
        #Hacemos copias para actualizar (serían las u' del pseudocódigo)
        dist_nueva=dict(dist)
        prev_nueva=dict(prev)
        updated_edges=[] #lista con los arcos que mejoran en la iteración.
        cambio=False #Añadimos la variable cambio con valor falso
        #Recorremos todos los arcos del grafo.
        for k in graph:
            #Si no se puede llegar a k, pasamos y continuamos
            if dist[k]==float('inf'):
                continue
            #Comprobamos si pasar por k mejora la distancia a j
            for j, w in graph[k].items():
                if dist[k]+w<dist_nueva[j]:
                    dist_nueva[j]=dist[k]+w
                    prev_nueva[j]=k
                    updated_edges.append((k,j))
                    cambio=True
        #Actualizamos las distancias y predecesores
        dist=dist_nueva
        prev=prev_nueva
        #Registramos el paso para la animación:
        paso={} #Creamos un diccionario vacío
        paso['iteration']=t #Guardamos el número de pasada
        paso['distances']=dict(dist) #Guardamos una copia del diccionario de distancias
        paso['predecessors']=dict(prev) #Guardamos una copia del diccionario de predecesores
        paso['updated_edges']=list(updated_edges) #Guardamos la lista de los arcos que se han
        # mejorado en esta pasada
        steps.append(paso) #Se añade el diccionario paso a la lista steps
        #Si no ha habido cambio, ninguna mejora, paramos, ya hemos llegado a la solución
        if not cambio: 
            break
    #Detectamos si hay algún ciclo negativo, para ver si pueden mejorarse las distancias todavía.
    has_negative_cycle=False
    for k in graph:
        #Si no podemos llegar a k desde el origen, ese vértice no influye para detectar ciclos negativos
        if dist[k]==float('inf'):
            continue
        #Si aún podemos mejorar la distancia de j pasando por k, existirá un ciclo negativo
        for j, w in graph[k].items():
            if dist[k]+w<dist[j]:
                has_negative_cycle=True
                break
        #Si ya hemos detectado un ciclo negativo, no seguimos revisando.
        if has_negative_cycle:
            break
    # Si hay algún ciclo negativo, devuelve None porque no existe camino mínimo. 
    # En caso contrario, reconstruimos el camino.   
    if has_negative_cycle:
        path=None
    else:
        path=reconstruct_path(prev,source,target)
    #La función bellman_ford devuelve toda la información: distancias,
    # predecesores, el camino óptimo, los pasos y si hay ciclo negativo.
    return {
        'distances':dist,
        'predecessors':prev,
        'path':path,
        'steps':steps,
        'has_negative_cycle': has_negative_cycle,
    }

# ─────────────────────────────────────────────────────────────────
# REFLEXIÓN (responder aquí como comentario)
# ─────────────────────────────────────────────────────────────────
#
# P1. Ejecutad Dijkstra sobre el Problema 2 (arcos negativos).
#     ¿Qué camino devuelve y con qué coste? ¿Por qué falla?
#     ¿Qué nodo se marca como permanente prematuramente?
#
# P1: Dijkstra devuelve el camino: Madrid → Guadalajara → Cuenca → Valencia.
# Este tiene coste 11, mientras que el óptimo sería 9.
# Falla porque una vez que un nodo se incorpora al conjunto de permanentes P,
# su distancia se considera definitiva y no se vuelve a actualizar. Dijkstra 
# funciona bien cuando las distancias son mayores o iguales a cero y en este 
# caso hay alguna distancia negativa, por lo que el algoritmo falla.
# El algoritmo marca Guadalajara como permanente con distancia 2 desde Madrid.
# Pero existe un camino mejor si pasase por Toledo (Madrid → Toledo → Guadalajara)
# con coste 3+(-3)=0.
#
# P2. Comparad el número de comprobaciones de arcos de Dijkstra
#     y Bellman-Ford en el Problema 1. ¿Por qué Dijkstra necesita
#     menos? Relacionad con O(n^2) frente a O(n^3).
#
# P2: Dijkstra necesita menos comprobaciones debido a que es un algoritmo que, en cada 
# iteración, selecciona un único vértice, el de distancia mínima. Lo "fija" y pasa a permanentes P,
# y después solo comprueba los arcos que salen de él. Como Dijkstra trabaja con pesos no 
# negativos, asegura que no existe otro camino que mejore la distancia.
# Por otro lado, Bellman-Ford, en cada pasada, recorre todos los arcos del grafo, no 
# solo los vecinos, hasta garantizar que ha obtenido el óptimo.
# Dijkstra tiene orden O(n^2) porque el algoritmo realiza n iteraciones y, en cada una, encuentra
# el vértice con distancia mínima. Solo actualiza las distancias de los vecinos del 
# seleccionado. Mientras que Bellman-Ford tiene orden O(n^3), ya que realiza hasta n-1 pasadas
# y en cada una examina todos los arcos, que pueden ser del orden de n^2. 
# En conclusión, Dijkstra reduce el número de comprobaciones, trabajando de forma más local, 
# mientras que Bellman-Ford realiza comprobaciones globales repetidas, lo que aumenta su coste
# computacional.
#
# P3. En el Problema 3, la transformación w = -ln(r) exige r > 0.
#     ¿Es razonable? Si Bellman-Ford no detecta ciclo negativo,
#     ¿significa que no hay ninguna oportunidad de beneficio, o solo
#     que no hay arbitraje sin riesgo? Razonad la diferencia.
#
# P3: La condición r>0 es razonable, ya que una tasa de cambio representa lo que te dan de una
# moneda a cambio de otra. No puede ser negativa porque eso no tendría sentido económico.
# Además, tampoco puede ser 0, porque, si no, significaría que la moneda no vale nada. Luego
# tiene sentido que sea mayor que cero. Igualmente, en el problema usamos logaritmos, si alguna
# tasa fuera 0 o negativa el logaritmo no estaría definido.
# Si Bellman-Ford no detecta un ciclo negativo significa que no existe una oportunidad de 
# arbitraje sin riesgo. Es decir, no hay ningún ciclo de cambios de divisa cuyo producto 
# de tasas sea mayor que 1. Luego no es posible empezar con una cantidad de dinero y, tras 
# recorrer un ciclo, obtener de manera segura más dinero.
# Aún así, esto no implica que no se pueda sacar beneficio, sino que no hay beneficio 
# garantizado e inmediato. 
# En cambio, si Bellman-Ford detectara algún ciclo negativo entonces habría encontrado un 
# ciclo donde, al multiplicar las tasas, el resultado es mayor que 1 y por tanto, habría 
# una ganancia asegurada y sin riesgo(-(lnr12+lnr23+lnr31)<0 sí y solo sí r12*r23*r31>1).
#
# P4. ¿Qué ocurre si aplicáis Bellman-Ford al Problema 1?
#     ¿Resultado correcto? ¿Cuántas pasadas antes de la convergencia
#     anticipada vs. iteraciones de Dijkstra?
#
# P4: Bellman-Ford obtiene el resultado correcto del problema 1. Aunque se use normalmente cuando
# hay pesos negativos, funciona también para grafos con pesos positivos, como en este
# caso, donde se obtiene la solución correcta. 
# Bellman-Ford necesita 4 pasadas, ya que el camino más corto entre Moncloa y Retiro tiene 4 arcos
# y Bellman-Ford necesita una pasada por cada arco. 
# Por otro lado, vemos que Dijkstra necesita 7 iteraciones, una por cada vértice que pasa al conjunto 
# de permanentes. En cada iteración selecciona el vértice de distancia mínima y fija su valor, 
# construyendo así el camino mínimo.
# Pero, que Bellman-Ford necesite menos no significa que sea más eficiente, ya que debe recorrer 
# todos los arcos, lo cual será más costoso computacionalmente, como hemos visto en la segunda 
# pregunta.  
#
# OPCIONAL. Suponed que cada conversión de divisas tiene una comisión
# fija de c euros. ¿Cómo modificaríais los pesos? ¿Sigue siendo
# válida la transformación logarítmica? Si no, proponed alternativa.
#
# OPCIONAL: Al introducir una comisión fija c en cada operación de cambio, la cantidad
# de dinero al pasar de la divisa i a la j deja de ser multiplicativa. Viene dado por: 
# xj=xi*rij-c, donde xi es el capital en la divisa de origen, rij la tasa de cambio y c
# la comisión fija. El "coste" del arco ya no sería constante, sino que dependerá del 
# capital que se tiene en ese punto del recorrido.
# La transformación logarítmica wij=-ln(rij) deja de ser válida. El logaritmo permite
# transformar productos en sumas: ln(a*b)=ln(a)+ln(b), lo que se utiliza para aplicar 
# Bellman-Ford cuando no hay comisiones. Pero no existe una identidad equivalente cuando 
# se tiene una resta. No se puede convertir el problema en uno donde cada arco tenga un
# peso fijo y el coste total sea la suma, ya que la comisión depende del dinero que se 
# lleva en ese momento. 
# Una alternativa sería modelar ahora el problema en función de la tasa de cambio y del 
# capital disponible. Sería maximizando el capital.
# Si denotamos f(j,k) al máximo capital que se puede conseguir en la divisa j después de 
# k cambios, tendríamos f(j,k)=máx_i(rij*f(i,k-1)-c). Habrá oportunidad de arbitraje si 
# para algún k(número de cambios) partiendo de una divisa inicial d, con capital x0,
# se tiene f(d,k)>x0, es decir, se podría volver a la divisa inicial con más dinero que 
# el inicial.
# Otra forma equivalente sería modelarlo considerando cada arco como una función afín: 
# fij(x)=x*rij-c.
# Al recorrer un ciclo, lo que hacemos es componer las funciones, así obtendríamos una 
# transformación del estilo: x_final=a*x0-b, siendo a el producto de 
# las tasas de cambio y b recoge el valor acumulado de todas las comisiones. La condición 
# para decidir si hay arbitraje sería (a-1)*x0>b.