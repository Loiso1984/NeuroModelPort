import numpy as np
from scipy.sparse import lil_matrix

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
        
        # Apply trunk gNa reduction (demyelination model: low internodal Na density)
        if not mc.single_comp and mc.N_trunk > 0 and mc.gNa_trunk_mult < 1.0:
            trunk_slice = slice(1 + mc.N_ais, 1 + mc.N_ais + mc.N_trunk)
            gNa_v[trunk_slice] *= mc.gNa_trunk_mult

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

        # ── TASK 1: Hines topology arrays ─────────────────────────────────────
        # parent_idx[i]  : index of parent compartment (-1 for soma root)
        # hines_order    : post-order (leaves → root) traversal for Hines elimination
        # g_axial_to_parent[i]        : L_csr[i, parent[i]] — area-normalized coupling
        #                               used as a[i] in child's row (positive off-diagonal)
        # g_axial_parent_to_child[i]  : L_csr[parent[i], i] — coupling used as b[i]
        #                               in parent's row for eliminating child i
        parent_idx = np.full(N_comp, -1, dtype=np.int32)
        g_axial_to_parent = np.zeros(N_comp, dtype=np.float64)
        g_axial_parent_to_child = np.zeros(N_comp, dtype=np.float64)
        _hines_list: list = []

        if mc.single_comp:
            _hines_list = [0]
        else:
            _ht_start = 1 + mc.N_ais           # index of first trunk compartment
            _hfork    = _ht_start + mc.N_trunk - 1  # last trunk / fork point
            _hb1      = _hfork + 1              # first branch-1 compartment
            _hb2      = _hb1 + mc.N_b1          # first branch-2 compartment

            # AIS chain: soma → AIS[1] → ... → AIS[N_ais]
            if mc.N_ais > 0:
                parent_idx[1] = 0
                for _k in range(2, 1 + mc.N_ais):
                    parent_idx[_k] = _k - 1

            # Trunk chain: last-AIS (or soma) → trunk[0] → ... → fork
            if mc.N_trunk > 0:
                parent_idx[_ht_start] = mc.N_ais if mc.N_ais > 0 else 0
                for _k in range(_ht_start + 1, _hfork + 1):
                    parent_idx[_k] = _k - 1

            # Branch-1 chain: fork → b1[0] → ... → b1[N_b1-1]
            if mc.N_b1 > 0:
                parent_idx[_hb1] = _hfork
                for _k in range(_hb1 + 1, _hb1 + mc.N_b1):
                    parent_idx[_k] = _k - 1

            # Branch-2 chain: fork → b2[0] → ... → b2[N_b2-1]
            if mc.N_b2 > 0:
                parent_idx[_hb2] = _hfork
                for _k in range(_hb2 + 1, _hb2 + mc.N_b2):
                    parent_idx[_k] = _k - 1

            # Hines order: leaves → root (post-order)
            if mc.N_b1 > 0:
                _hines_list.extend(range(_hb1 + mc.N_b1 - 1, _hb1 - 1, -1))
            if mc.N_b2 > 0:
                _hines_list.extend(range(_hb2 + mc.N_b2 - 1, _hb2 - 1, -1))
            if mc.N_trunk > 0:
                _hines_list.extend(range(_hfork, _ht_start - 1, -1))
            if mc.N_ais > 0:
                _hines_list.extend(range(mc.N_ais, 0, -1))
            _hines_list.append(0)

        hines_order = np.array(_hines_list, dtype=np.int32)

        # Extract area-normalized coupling from L_csr (off-diagonals are positive)
        for _i in range(1, N_comp):
            _p = int(parent_idx[_i])
            if _p >= 0:
                g_axial_to_parent[_i]       = L_csr[_i, _p]   # a[i]
                g_axial_parent_to_child[_i] = L_csr[_p, _i]   # b[i]

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
