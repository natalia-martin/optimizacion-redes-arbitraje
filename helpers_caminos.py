"""
Laboratorio: Caminos mínimos — Dijkstra y Bellman-Ford
Universidad Complutense de Madrid — Flujos en Redes (3º Matemáticas)

FUNCIONES BASE — NO MODIFICAR
Estas funciones están implementadas. Las usaréis en vuestra implementación.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import Dict, List, Tuple, Optional
import math

# ═══════════════════════════════════════════════════════════════════
# TIPOS
# ═══════════════════════════════════════════════════════════════════

Graph = Dict[str, Dict[str, float]]
Pos = Dict[str, Tuple[float, float]]


# ═══════════════════════════════════════════════════════════════════
# 1. RECONSTRUCCIÓN DE CAMINO
# ═══════════════════════════════════════════════════════════════════

def reconstruct_path(
    predecessors: Dict[str, Optional[str]],
    source: str,
    target: str
) -> Optional[List[str]]:
    """
    Reconstruye el camino de source a target a partir del diccionario
    de predecesores.

    Parámetros
    ----------
    predecessors : dict  {nodo: nodo_predecesor o None}
    source       : nodo origen
    target       : nodo destino

    Devuelve
    --------
    Lista de nodos [source, ..., target] o None si target no es alcanzable.
    """
    if target not in predecessors or (predecessors[target] is None and target != source):
        # target inalcanzable — comprobar si es el propio source
        if target == source:
            return [source]
        return None

    path = []
    node = target
    visited = set()
    while node is not None:
        if node in visited:
            return None  # ciclo en predecesores
        visited.add(node)
        path.append(node)
        node = predecessors.get(node)

    path.reverse()
    if path[0] != source:
        return None
    return path


# ═══════════════════════════════════════════════════════════════════
# 2. UTILIDADES DE GRAFO
# ═══════════════════════════════════════════════════════════════════

def get_all_nodes(graph: Graph) -> List[str]:
    """Devuelve todos los nodos del grafo (incluyendo destinos sin salida)."""
    nodes = set(graph.keys())
    for u in graph:
        for v in graph[u]:
            nodes.add(v)
    return sorted(nodes)


def get_all_edges(graph: Graph) -> List[Tuple[str, str, float]]:
    """Devuelve lista de (origen, destino, peso) para todas las aristas."""
    edges = []
    for u in graph:
        for v, w in graph[u].items():
            edges.append((u, v, w))
    return edges


# ═══════════════════════════════════════════════════════════════════
# 3. DEFINICIÓN DE PROBLEMAS
# ═══════════════════════════════════════════════════════════════════

def create_problem_dijkstra() -> dict:
    """
    Problema 1: Red de mensajería urbana en Madrid.
    Todos los pesos positivos (tiempos en minutos).

    Devuelve dict con: graph, pos, source, target, name, description,
                       optimal_dist, optimal_path
    """
    graph = {
        'Moncloa':    {'Arguelles': 2, 'Tetuan': 4},
        'Arguelles':  {'Moncloa': 2, 'Chamberi': 3, 'Sol': 10},
        'Tetuan':     {'Moncloa': 4, 'Chamartin': 5, 'Chamberi': 3},
        'Sol':        {'Arguelles': 10, 'Chamberi': 8, 'Lavapies': 4, 'Atocha': 6},
        'Chamberi':   {'Arguelles': 3, 'Tetuan': 3, 'Sol': 8, 'Chamartin': 7, 'Salamanca': 5},
        'Chamartin':  {'Tetuan': 5, 'Chamberi': 7, 'Salamanca': 3},
        'Lavapies':   {'Sol': 4, 'Atocha': 3},
        'Atocha':     {'Sol': 6, 'Lavapies': 3, 'Retiro': 5},
        'Salamanca':  {'Chamartin': 3, 'Chamberi': 5, 'Retiro': 4},
        'Retiro':     {'Atocha': 5, 'Salamanca': 4},
    }

    # Posiciones aproximadas (coordenadas estilizadas)
    pos = {
        'Moncloa':    (-3.0,  1.5),
        'Arguelles':  (-2.0,  0.5),
        'Tetuan':     (-1.0,  2.5),
        'Sol':        ( 0.0,  0.0),
        'Chamberi':   ( 0.0,  1.5),
        'Chamartin':  ( 1.0,  2.5),
        'Lavapies':   (-0.5, -1.2),
        'Atocha':     ( 1.0, -1.0),
        'Salamanca':  ( 2.0,  1.0),
        'Retiro':     ( 2.5, -0.5),
    }

    return {
        'graph': graph,
        'pos': pos,
        'source': 'Moncloa',
        'target': 'Retiro',
        'name': 'P1: Red de mensajería urbana (Dijkstra)',
        'description': (
            'Empresa de mensajería en Madrid. Encontrar la ruta más rápida\n'
            'de Moncloa a Retiro. Pesos = minutos de tránsito.'
        ),
        'optimal_dist': 14,
        'optimal_path': ['Moncloa', 'Arguelles', 'Chamberi', 'Salamanca', 'Retiro'],
    }


def create_problem_bellman_ford() -> dict:
    """
    Problema 2: Red de distribución con carga de retorno.

    Cada arco tiene coste neto = transporte − ingresos por mercancía.
    En el arco Toledo→Guadalajara existe un transportista que necesita
    llenar camiones vacíos de vuelta y paga por ello; el ingreso supera
    el coste de transporte, dando un coste neto de −3. Esta oportunidad
    es propia del arco (camión vacío haciendo esa ruta), no del vértice:
    llegar a Toledo desde cualquier origen permite aprovecharla.

    Dijkstra falla porque finaliza Guadalajara a distancia 2 (ruta directa
    Madrid→Guadalajara) antes de descubrir que pasando por Toledo y
    aprovechando el retorno se llega con coste 0.

    Devuelve dict con: graph, pos, source, target, name, description,
                       optimal_dist, optimal_path
    """
    # Pesos = coste neto del trayecto (transporte − ingreso por carga)
    # Toledo→Guadalajara: transporte 2, pago por retorno 5 → neto = −3
    graph = {
        'Madrid':       {'Guadalajara': 2, 'Toledo': 3, 'Segovia': 9},
        'Guadalajara':  {'Cuenca': 6},
        'Toledo':       {'Guadalajara': -3, 'Ciudad_Real': 5},
        'Segovia':      {'Cuenca': 7},
        'Cuenca':       {'Albacete': 2, 'Valencia': 3},
        'Ciudad_Real':  {'Albacete': 3},
        'Albacete':     {'Valencia': 2},
        'Valencia':     {},
    }

    pos = {
        'Madrid':       (-2.5,  0.8),
        'Toledo':       (-1.5, -0.8),
        'Segovia':      (-3.0,  2.2),
        'Guadalajara':  ( 0.0,  1.8),
        'Cuenca':       ( 1.8,  0.8),
        'Ciudad_Real':  ( 0.5, -1.2),
        'Albacete':     ( 2.8, -0.5),
        'Valencia':     ( 3.8,  0.8),
    }

    return {
        'graph': graph,
        'pos': pos,
        'source': 'Madrid',
        'target': 'Valencia',
        'name': 'P2: Red de distribución con carga de retorno (Bellman-Ford)',
        'description': (
            'Red de distribución entre ciudades. Los pesos representan el coste\n'
            'neto de cada trayecto (transporte − ingresos por carga). En el arco\n'
            'Toledo→Guadalajara hay un transportista que paga por rellenar camiones\n'
            'vacíos de vuelta (coste neto −3). Dijkstra finaliza Guadalajara a\n'
            'distancia 2 por la ruta directa y no descubre el atajo por Toledo.'
        ),
        'optimal_dist': 9,
        'optimal_path': ['Madrid', 'Toledo', 'Guadalajara', 'Cuenca', 'Valencia'],
    }


def create_problem_negative_cycle() -> dict:
    """
    Problema 3: Detección de arbitraje cambiario.
    Las tasas se transforman a pesos w_ij = -ln(r_ij).
    Existe un ciclo negativo EUR → USD → GBP → EUR.

    Devuelve dict con: graph, pos, source, name, description,
                       has_negative_cycle, rates (tasas originales)
    """
    # Tasas de cambio (filas = origen, columnas = destino)
    rates = {
        ('EUR', 'USD'): 1.12,
        ('EUR', 'GBP'): 0.84,
        ('EUR', 'JPY'): 130.0,
        ('EUR', 'CHF'): 1.05,
        ('USD', 'EUR'): 0.88,
        ('USD', 'GBP'): 0.77,
        ('USD', 'JPY'): 115.0,
        ('GBP', 'EUR'): 1.22,
        ('GBP', 'USD'): 1.31,
        ('GBP', 'CHF'): 1.27,
        ('JPY', 'EUR'): 0.0078,
        ('JPY', 'USD'): 0.0088,
        ('CHF', 'EUR'): 0.96,
        ('CHF', 'GBP'): 0.80,
    }

    # Transformar a pesos: w = -ln(r)
    graph = {}
    for (u, v), r in rates.items():
        if u not in graph:
            graph[u] = {}
        graph[u][v] = round(-math.log(r), 6)

    # Asegurar que todos los nodos tienen entrada en graph
    for node in ['EUR', 'USD', 'GBP', 'JPY', 'CHF']:
        if node not in graph:
            graph[node] = {}

    pos = {
        'EUR': ( 0.0,  2.0),
        'USD': ( 2.0,  1.0),
        'GBP': (-2.0,  1.0),
        'JPY': ( 1.5, -1.0),
        'CHF': (-1.5, -1.0),
    }

    # Verificar arbitraje: EUR → USD → GBP → EUR
    # 1.12 * 0.77 * 1.22 = 1.05197... > 1 → hay arbitraje
    cycle_product = rates[('EUR','USD')] * rates[('USD','GBP')] * rates[('GBP','EUR')]

    return {
        'graph': graph,
        'pos': pos,
        'source': 'EUR',
        'target': 'EUR',
        'name': 'P3: Detección de arbitraje cambiario',
        'description': (
            f'Red de divisas con 5 monedas. Tasas transformadas a w = -ln(r).\n'
            f'Ciclo EUR→USD→GBP→EUR: producto = {cycle_product:.5f} > 1 → arbitraje.'
        ),
        'has_negative_cycle': True,
        'rates': rates,
    }


# ═══════════════════════════════════════════════════════════════════
# 4. VISUALIZACIÓN ESTÁTICA
# ═══════════════════════════════════════════════════════════════════

# Paleta
_COL = {
    'node_default':  '#b0bec5',
    'node_visited':  '#66bb6a',
    'node_current':  '#fdd835',
    'node_source':   '#42a5f5',
    'node_target':   '#ef5350',
    'edge_default':  '#90a4ae',
    'edge_path':     '#e53935',
    'edge_negative': '#f57c00',
    'text':          '#212121',
    'weight':        '#455a64',
    'dist_label':    '#1565c0',
}


def _draw_arrow(ax, p1, p2, color='#90a4ae', lw=1.5, alpha=0.8,
                shrink_source=0.22, shrink_target=0.22, style='->', zorder=1):
    """Dibuja una flecha curvada entre dos puntos."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return

    # Acortar
    ux, uy = dx / length, dy / length
    x1 = p1[0] + shrink_source * ux
    y1 = p1[1] + shrink_source * uy
    x2 = p2[0] - shrink_target * ux
    y2 = p2[1] - shrink_target * uy

    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle='->', color=color, lw=lw,
                    connectionstyle='arc3,rad=0.1',
                    alpha=alpha),
                zorder=zorder)


def draw_graph(
    graph: Graph,
    pos: Pos,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    highlight_edges: Optional[List[Tuple[str, str]]] = None,
    highlight_nodes: Optional[Dict[str, str]] = None,
    dist_labels: Optional[Dict[str, float]] = None,
    node_size: float = 0.2,
    show_weights: bool = True,
):
    """
    Dibuja el grafo dirigido con pesos.

    Parámetros
    ----------
    graph           : diccionario de adyacencia
    pos             : posiciones {nodo: (x, y)}
    ax              : eje de matplotlib (se crea uno nuevo si None)
    title           : título
    highlight_edges : lista de (u, v) a resaltar en rojo
    highlight_nodes : dict {nodo: color_matplotlib}
    dist_labels     : dict {nodo: distancia} para mostrar bajo el nodo
    node_size       : radio visual del nodo
    show_weights    : si mostrar los pesos de las aristas
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    highlight_edges = set(highlight_edges or [])
    highlight_nodes = highlight_nodes or {}

    # Aristas
    edges = get_all_edges(graph)
    for u, v, w in edges:
        if u not in pos or v not in pos:
            continue
        is_hl = (u, v) in highlight_edges
        color = _COL['edge_path'] if is_hl else (
            _COL['edge_negative'] if w < 0 else _COL['edge_default'])
        lw = 2.5 if is_hl else 1.3
        zo = 3 if is_hl else 1
        _draw_arrow(ax, pos[u], pos[v], color=color, lw=lw,
                    shrink_source=node_size, shrink_target=node_size, zorder=zo)

        # Peso
        if show_weights:
            mx = 0.35 * pos[u][0] + 0.65 * pos[v][0]
            my = 0.35 * pos[u][1] + 0.65 * pos[v][1]
            # Desplazar ligeramente perpendicular
            dx = pos[v][0] - pos[u][0]
            dy = pos[v][1] - pos[u][1]
            length = math.hypot(dx, dy)
            if length > 1e-9:
                nx, ny = -dy / length, dx / length
                mx += 0.12 * nx
                my += 0.12 * ny
            weight_color = _COL['edge_negative'] if w < 0 else _COL['weight']
            fmt = f'{w:.3f}' if isinstance(w, float) and (w != int(w)) else f'{w:g}'
            ax.text(mx, my, fmt, fontsize=7, ha='center', va='center',
                    color=weight_color, fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=2.5, foreground='white')],
                    zorder=4)

    # Nodos
    all_nodes = get_all_nodes(graph)
    for node in all_nodes:
        if node not in pos:
            continue
        x, y = pos[node]
        color = highlight_nodes.get(node, _COL['node_default'])
        circle = plt.Circle((x, y), node_size, facecolor=color,
                             edgecolor='#37474f', linewidth=1.2, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, node, fontsize=7.5, ha='center', va='center',
                fontweight='bold', color=_COL['text'], zorder=6)

        # Etiqueta de distancia
        if dist_labels and node in dist_labels:
            d = dist_labels[node]
            dstr = '∞' if d == float('inf') else f'{d:g}'
            ax.text(x, y - node_size - 0.15, dstr, fontsize=7,
                    ha='center', va='top', color=_COL['dist_label'],
                    fontweight='bold', zorder=6)

    ax.set_aspect('equal')
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    margin = 1.0
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=10, fontweight='bold', pad=12)


def draw_solution(
    graph: Graph,
    pos: Pos,
    path: Optional[List[str]],
    distances: Dict[str, float],
    source: str,
    target: str,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
):
    """Dibuja el grafo con el camino mínimo resaltado."""
    highlight_edges = []
    highlight_nodes = {}

    if path:
        for i in range(len(path) - 1):
            highlight_edges.append((path[i], path[i + 1]))
        for node in path:
            highlight_nodes[node] = '#ffcdd2'  # rojo claro
    highlight_nodes[source] = _COL['node_source']
    highlight_nodes[target] = _COL['node_target']

    draw_graph(graph, pos, ax=ax, title=title,
               highlight_edges=highlight_edges,
               highlight_nodes=highlight_nodes,
               dist_labels=distances)


# ═══════════════════════════════════════════════════════════════════
# 5. ANIMACIÓN GIF
# ═══════════════════════════════════════════════════════════════════

def animate_algorithm(
    graph: Graph,
    pos: Pos,
    steps: List[dict],
    final_path: Optional[List[str]],
    source: str,
    target: str,
    filename: str = 'animation.gif',
    algorithm_name: str = 'Algoritmo',
    fps: int = 1,
    hold_last: int = 3,
):
    """
    Genera un GIF animado del progreso del algoritmo.

    Cada fotograma muestra:
      - Nodo actual en amarillo
      - Nodos visitados/finalizados en verde
      - Nodos no alcanzados en gris
      - Distancias actuales como etiquetas
      - Aristas relajadas en la iteración (Bellman-Ford)
    El último fotograma resalta el camino final.

    Parámetros
    ----------
    graph          : grafo
    pos            : posiciones
    steps          : lista de diccionarios (formato Dijkstra o Bellman-Ford)
    final_path     : camino mínimo final
    source, target : origen y destino
    filename       : nombre del fichero GIF
    algorithm_name : para el título
    fps            : fotogramas por segundo
    hold_last      : segundos extra en el último fotograma
    """
    if not steps:
        print(f"  [!] Sin pasos para animar ({algorithm_name})")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    # Determinar tipo de algoritmo
    is_dijkstra = 'visited' in steps[0]

    # Total frames: steps + frame final
    n_frames = len(steps) + 1

    def _update(frame_idx):
        ax.clear()

        if frame_idx < len(steps):
            step = steps[frame_idx]
            dists = step.get('distances', {})

            node_colors = {}
            highlight_edges_frame = []

            if is_dijkstra:
                visited = step.get('visited', set())
                current = step.get('current', None)
                for n in get_all_nodes(graph):
                    if n == current:
                        node_colors[n] = _COL['node_current']
                    elif n in visited:
                        node_colors[n] = _COL['node_visited']
                    else:
                        node_colors[n] = _COL['node_default']
                node_colors[source] = _COL['node_source']
                if current == target:
                    node_colors[target] = _COL['node_target']
                frame_title = f'{algorithm_name} — procesando: {current}'
            else:
                # Bellman-Ford
                iteration = step.get('iteration', '?')
                updated = step.get('updated_edges', [])
                highlight_edges_frame = updated
                for n in get_all_nodes(graph):
                    d = dists.get(n, float('inf'))
                    if d < float('inf'):
                        node_colors[n] = _COL['node_visited']
                    else:
                        node_colors[n] = _COL['node_default']
                node_colors[source] = _COL['node_source']
                frame_title = (f'{algorithm_name} — pasada {iteration}'
                               f'  ({len(updated)} aristas relajadas)')

            draw_graph(graph, pos, ax=ax, title=frame_title,
                       highlight_edges=highlight_edges_frame,
                       highlight_nodes=node_colors,
                       dist_labels=dists)
        else:
            # Frame final: camino resaltado
            final_dists = steps[-1].get('distances', {})
            draw_solution(graph, pos, final_path, final_dists,
                          source, target, ax=ax,
                          title=f'{algorithm_name} — camino mínimo')

    # Crear frames con pausa extra al final
    frame_list = list(range(n_frames)) + [n_frames - 1] * (hold_last * fps)

    anim = FuncAnimation(fig, _update, frames=frame_list,
                         interval=1000 // fps, repeat=False)

    try:
        anim.save(filename, writer=PillowWriter(fps=fps))
        print(f"  [✓] Animación guardada: {filename}")
    except Exception as e:
        print(f"  [!] Error al guardar animación: {e}")
        print("      Instala Pillow: pip install Pillow")
    finally:
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# 6. VERIFICACIÓN
# ═══════════════════════════════════════════════════════════════════

def verify_result(result: dict, expected_dist: float, expected_path: List[str],
                  problem_name: str, algo_name: str) -> bool:
    """
    Verifica que el resultado del alumno coincide con la solución esperada.
    Imprime diagnóstico por consola.
    """
    ok = True
    dist = result.get('distances', {})
    path = result.get('path')
    target = expected_path[-1] if expected_path else None

    if target and target in dist:
        if abs(dist[target] - expected_dist) > 1e-6:
            print(f"  [✗] {problem_name}/{algo_name}: distancia incorrecta. "
                  f"Esperada {expected_dist}, obtenida {dist[target]}")
            ok = False
        else:
            print(f"  [✓] {problem_name}/{algo_name}: distancia correcta = {expected_dist}")
    else:
        print(f"  [✗] {problem_name}/{algo_name}: destino no alcanzado")
        ok = False

    if path != expected_path:
        print(f"      Camino esperado: {expected_path}")
        print(f"      Camino obtenido: {path}")
        # El camino puede ser distinto pero igualmente óptimo
        if target and target in dist and abs(dist[target] - expected_dist) < 1e-6:
            print(f"      (Camino distinto pero distancia correcta — aceptable)")
    else:
        print(f"  [✓] {problem_name}/{algo_name}: camino correcto")

    steps = result.get('steps', [])
    if not steps:
        print(f"  [✗] {problem_name}/{algo_name}: lista de steps vacía")
        ok = False
    else:
        print(f"  [✓] {problem_name}/{algo_name}: {len(steps)} pasos registrados")

    return ok
