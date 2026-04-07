import numpy as np
from scipy.sparse import lil_matrix


def _build_hines_order(parent_idx: np.ndarray, n_comp: int) -> np.ndarray:
    """Build Hines elimination order (leaves-to-root) for tree solver.
    
    Returns array of compartment indices in order from distal (leaves)
    to proximal (root), which is the correct order for forward elimination
    in the Hines algorithm.
    """
    # Build children list for each node
    children = [[] for _ in range(n_comp)]
    for i in range(n_comp):
        p = parent_idx[i]
        if p >= 0:
            children[p].append(i)
    
    # BFS/DFS from leaves to root
    order = []
    visited = [False] * n_comp
    
    def visit_leaves_first(node):
        if visited[node]:
            return
        visited[node] = True
        # Visit all children first (they are more distal)
        for child in children[node]:
            visit_leaves_first(child)
        # Then add this node (post-order)
        order.append(node)
    
    # Find root (parent = -1) and start from it
    root = 0
    for i in range(n_comp):
        if parent_idx[i] < 0:
            root = i
            break
    
    visit_leaves_first(root)
    # order is now root-to-leaves (post-order), reverse to get leaves-to-root
    return np.array(order[::-1], dtype=np.int64)


def gax(d: float, Ra: float, dx: float) -> float:
    """Аксиальная проводимость однородного цилиндра (мСм). | Axial conductance of uniform cylinder (mS)."""
    # Формула: g = (pi * r^2) / (Ra * dx), умножаем на 1000 для перевода в мСм | Formula: g = (pi * r^2) / (Ra * dx), multiply by 1000 for mS conversion
    return np.pi * (d**2) / (4.0 * Ra * dx) * 1000.0

def gax_pair(d1: float, d2: float, Ra: float, dx: float) -> float:
    """Проводимость на стыке двух компартментов разного диаметра (сома->аксон). | Conductance at junction of two compartments of different diameters (soma->axon)."""
    R1 = 4.0 * Ra * (dx / 2.0) / (np.pi * d1**2)
    R2 = 4.0 * Ra * (dx / 2.0) / (np.pi * d2**2)
    return 1000.0 / (R1 + R2)

def add_link(L: lil_matrix, i: int, j: int, g: float):
    """Добавляет связь между компартментами i и j с проводимостью g. | Adds connection between compartments i and j with conductance g."""
    L[i, i] -= g
    L[j, j] -= g
    L[i, j] += g
    L[j, i] += g

class MorphologyBuilder:
    """Класс для сборки геометрии нейрона, площадей мембраны и Лапласиана. | Class for assembling neuron geometry, membrane areas, and Laplacian."""
    
    @staticmethod
    def build(config) -> dict:
        mc = config.morphology
        cc = config.channels
        
        # Индексация компартментов (0-based) | Compartment indexing (0-based)
        if mc.single_comp:
            N_comp = 1
        else:
            N_comp = 1 + mc.N_ais + mc.N_trunk + mc.N_b1 + mc.N_b2
            
        areas = np.zeros(N_comp)
        
        # 1. Площади мембран (см²) | 1. Membrane areas (cm²)
        areas[0] = np.pi * (mc.d_soma**2)  # Сома (сфера) | Soma (sphere)
        
        if not mc.single_comp:
            # AIS (Аксонная начальная сегментация)
            for k in range(1, 1 + mc.N_ais):
                areas[k] = np.pi * mc.d_ais * mc.dx
            # Trunk (Ствол)
            idx_trunk_start = 1 + mc.N_ais
            idx_fork = idx_trunk_start + mc.N_trunk - 1
            for k in range(idx_trunk_start, idx_fork + 1):
                areas[k] = np.pi * mc.d_trunk * mc.dx
            # Branch 1 (Ветвь 1)
            idx_b1_start = idx_fork + 1
            for k in range(idx_b1_start, idx_b1_start + mc.N_b1):
                areas[k] = np.pi * mc.d_b1 * mc.dx
            # Branch 2 (Ветвь 2)
            idx_b2_start = idx_b1_start + mc.N_b1
            for k in range(idx_b2_start, idx_b2_start + mc.N_b2):
                areas[k] = np.pi * mc.d_b2 * mc.dx

        # 1b. Per-compartment diameters (cm) — for volume-dependent B_Ca (Stage 3.4)
        diameters = np.full(N_comp, mc.d_soma, dtype=np.float64)
        if not mc.single_comp:
            for k in range(1, 1 + mc.N_ais):
                diameters[k] = mc.d_ais
            idx_trunk_start = 1 + mc.N_ais
            idx_fork = idx_trunk_start + mc.N_trunk - 1
            for k in range(idx_trunk_start, idx_fork + 1):
                diameters[k] = mc.d_trunk
            idx_b1_start = idx_fork + 1
            for k in range(idx_b1_start, idx_b1_start + mc.N_b1):
                diameters[k] = mc.d_b1
            idx_b2_start = idx_b1_start + mc.N_b1
            for k in range(idx_b2_start, idx_b2_start + mc.N_b2):
                diameters[k] = mc.d_b2

        # 2. Векторы проводимостей (с AIS-множителями) в плотности [мСм/см²] | 2. Conductance vectors (with AIS multipliers) in density [mS/cm²]
        gNa_v = np.full(N_comp, cc.gNa_max, dtype=np.float64)
        gK_v  = np.full(N_comp, cc.gK_max, dtype=np.float64)
        gL_v  = np.full(N_comp, cc.gL, dtype=np.float64)
        
        gIh_v  = np.zeros(N_comp, dtype=np.float64)
        gCa_v  = np.zeros(N_comp, dtype=np.float64)
        gA_v   = np.zeros(N_comp, dtype=np.float64)
        gSK_v  = np.zeros(N_comp, dtype=np.float64)
        gTCa_v = np.zeros(N_comp, dtype=np.float64)
        gIM_v  = np.zeros(N_comp, dtype=np.float64)
        gNaP_v = np.zeros(N_comp, dtype=np.float64)
        gNaR_v = np.zeros(N_comp, dtype=np.float64)

        if cc.enable_Ih:
            gIh_v.fill(cc.gIh_max)
        if cc.enable_ICa:
            gCa_v.fill(cc.gCa_max)
        if cc.enable_IA:
            gA_v.fill(cc.gA_max)
        if cc.enable_SK:
            gSK_v.fill(cc.gSK_max)
        if cc.enable_ITCa:
            gTCa_v.fill(cc.gTCa_max)
        if cc.enable_IM:
            gIM_v.fill(cc.gIM_max)
        if cc.enable_NaP:
            gNaP_v.fill(cc.gNaP_max)
        if cc.enable_NaR:
            gNaR_v.fill(cc.gNaR_max)
        
        # Membrane capacitance [µF/cm²]
        Cm_v = np.full(N_comp, cc.Cm, dtype=np.float64)
        
        # Apply AIS multipliers for all channels (matches Scilab v9.0 behaviour)
        if not mc.single_comp and mc.N_ais > 0:
            ais_slice = slice(1, 1 + mc.N_ais)
            gNa_v[ais_slice] *= mc.gNa_ais_mult
            gK_v[ais_slice]  *= mc.gK_ais_mult
            gIh_v[ais_slice] *= mc.gIh_ais_mult
            gCa_v[ais_slice]  *= mc.gCa_ais_mult
            gA_v[ais_slice]   *= mc.gA_ais_mult
            gTCa_v[ais_slice] *= mc.gCa_ais_mult  # T-type uses same AIS mult as L-type
            # M-type: primarily somatic/dendritic, no AIS boost (Shah 2008, J Neurosci)
            # gIM_v[ais_slice] *= 1.0  — intentionally no multiplier
            # Persistent & Resurgent Na: same AIS enrichment as transient Na
            gNaP_v[ais_slice] *= mc.gNa_ais_mult
            gNaR_v[ais_slice] *= mc.gNa_ais_mult

        # 3. Матрица Лапласа (LIL формат для быстрого заполнения) | 3. Laplacian matrix (LIL format for fast filling)
        L_matrix = lil_matrix((N_comp, N_comp), dtype=np.float64)
        
        if not mc.single_comp:
            if mc.N_ais > 0:
                # Сома -> AIS | Soma -> AIS
                add_link(L_matrix, 0, 1, gax_pair(mc.d_soma, mc.d_ais, mc.Ra, mc.dx))
                # Внутри AIS | Within AIS
                g_aa = gax(mc.d_ais, mc.Ra, mc.dx)
                for k in range(1, mc.N_ais):
                    add_link(L_matrix, k, k + 1, g_aa)
                # AIS -> Trunk
                if mc.N_trunk > 0:
                    add_link(L_matrix, mc.N_ais, idx_trunk_start, gax_pair(mc.d_ais, mc.d_trunk, mc.Ra, mc.dx))
            elif mc.N_trunk > 0:
                # Если нет AIS, Сома напрямую к Trunk | If no AIS, Soma directly to Trunk
                add_link(L_matrix, 0, 1, gax_pair(mc.d_soma, mc.d_trunk, mc.Ra, mc.dx))
                
            # Внутри Trunk | Within Trunk
            g_tt = gax(mc.d_trunk, mc.Ra, mc.dx)
            for k in range(idx_trunk_start, idx_fork):
                add_link(L_matrix, k, k + 1, g_tt)
                
            # Развилка (Fork) -> Ветви | Fork -> Branches
            if mc.N_b1 > 0:
                add_link(L_matrix, idx_fork, idx_b1_start, gax_pair(mc.d_trunk, mc.d_b1, mc.Ra, mc.dx))
                g_b1 = gax(mc.d_b1, mc.Ra, mc.dx)
                for k in range(idx_b1_start, idx_b1_start + mc.N_b1 - 1):
                    add_link(L_matrix, k, k + 1, g_b1)
                    
            if mc.N_b2 > 0:
                add_link(L_matrix, idx_fork, idx_b2_start, gax_pair(mc.d_trunk, mc.d_b2, mc.Ra, mc.dx))
                g_b2 = gax(mc.d_b2, mc.Ra, mc.dx)
                for k in range(idx_b2_start, idx_b2_start + mc.N_b2 - 1):
                    add_link(L_matrix, k, k + 1, g_b2)

        # Нормировка по площадям (L·V даёт плотность тока мкА/см²) | Normalization by areas (L·V gives current density µA/cm²)
        for k in range(N_comp):
            L_matrix[k, :] = L_matrix[k, :] / areas[k]

        # Преобразуем в CSR формат | Convert to CSR format
        L_csr = L_matrix.tocsr()

        # Build parent_idx and hines_order for native Hines solver
        parent_idx = np.full(N_comp, -1, dtype=np.int64)
        hines_order = np.arange(N_comp, dtype=np.int64)  # Default: root-to-leaves order
        
        if not mc.single_comp:
            # Build parent index array based on morphology topology
            # Soma is root (parent = -1)
            if mc.N_ais > 0:
                # AIS compartments connect to soma (0) or previous AIS
                parent_idx[1] = 0  # First AIS connects to soma
                for k in range(2, 1 + mc.N_ais):
                    parent_idx[k] = k - 1
                
                # Trunk connects to last AIS
                if mc.N_trunk > 0:
                    parent_idx[idx_trunk_start] = mc.N_ais  # First trunk connects to last AIS
                    for k in range(idx_trunk_start + 1, idx_fork + 1):
                        parent_idx[k] = k - 1
            elif mc.N_trunk > 0:
                # No AIS, trunk connects directly to soma
                parent_idx[1] = 0
                for k in range(2, 1 + mc.N_trunk):
                    parent_idx[k] = k - 1
            
            # Branches connect to fork (last trunk compartment)
            if mc.N_b1 > 0:
                parent_idx[idx_b1_start] = idx_fork
                for k in range(idx_b1_start + 1, idx_b1_start + mc.N_b1):
                    parent_idx[k] = k - 1
            
            if mc.N_b2 > 0:
                parent_idx[idx_b2_start] = idx_fork
                for k in range(idx_b2_start + 1, idx_b2_start + mc.N_b2):
                    parent_idx[k] = k - 1
            
            # Build hines_order: leaves-to-root (reverse topological order)
            # For Hines algorithm, we need to eliminate from leaves to root
            # Simple approach: use reverse BFS order
            hines_order = _build_hines_order(parent_idx, N_comp)

        # Build axial conductance arrays for Hines solver
        g_axial_to_parent = np.zeros(N_comp, dtype=np.float64)
        g_axial_parent_to_child = np.zeros(N_comp, dtype=np.float64)
        
        if not mc.single_comp:
            for i in range(N_comp):
                p = parent_idx[i]
                if p >= 0:
                    # Get conductance from L matrix (already normalized by area)
                    g_val = L_matrix[i, p] if L_matrix[i, p] != 0 else L_matrix[p, i]
                    if g_val == 0:
                        # Fallback: compute from diameters
                        d_child = diameters[i]
                        d_parent = diameters[p]
                        g_val = gax_pair(d_child, d_parent, mc.Ra, mc.dx)
                    g_axial_to_parent[i] = abs(g_val)
            
            # Sum conductances from parent to all children
            for i in range(N_comp):
                for j in range(N_comp):
                    if parent_idx[j] == i:
                        g_axial_parent_to_child[i] += g_axial_to_parent[j]

        return {
            'N_comp': N_comp,
            'areas': areas,
            'diameters': diameters,
            'gNa_v': gNa_v, 'gK_v': gK_v, 'gL_v': gL_v,
            'gIh_v': gIh_v, 'gCa_v': gCa_v, 'gA_v': gA_v, 'gSK_v': gSK_v, 'gTCa_v': gTCa_v,
            'gIM_v': gIM_v, 'gNaP_v': gNaP_v, 'gNaR_v': gNaR_v,
            'Cm_v': Cm_v,
            'L_data': L_csr.data,
            'L_indices': L_csr.indices,
            'L_indptr': L_csr.indptr,
            'parent_idx': parent_idx,
            'hines_order': hines_order,
            'g_axial_to_parent': g_axial_to_parent,
            'g_axial_parent_to_child': g_axial_parent_to_child,
        }
