import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
# Импортируем наши кинетические функции
from .kinetics import *

@dataclass
class GateInfo:
    name: str
    power: int
    alpha_fn: callable
    beta_fn: callable

@dataclass
class Channel:
    name: str
    is_leak: bool = False
    is_Ca: bool = False
    color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    gates: List[GateInfo] = field(default_factory=list)

class ChannelRegistry:
    """ООП-реестр каналов для взаимодействия с GUI и сборки вектора состояний."""
    
    def __init__(self):
        self.channels = []
        self._build_registry()

    def _build_registry(self):
        # 1. Na (m^3 * h)
        self.channels.append(Channel(
            name="Na", color=(1.0, 0.0, 0.0),
            gates=[
                GateInfo('m', 3, am, bm),
                GateInfo('h', 1, ah, bh)
            ]
        ))
        # 2. K (n^4)
        self.channels.append(Channel(
            name="K", color=(0.0, 0.0, 1.0),
            gates=[GateInfo('n', 4, an, bn)]
        ))
        # 3. Leak (Утечка)
        self.channels.append(Channel(name="Leak", is_leak=True, color=(0.1, 0.6, 0.1)))
        
        # 4. Ih (HCN)
        self.channels.append(Channel(
            name="Ih", color=(0.6, 0.0, 0.6),
            gates=[GateInfo('r', 1, ar_Ih, br_Ih)]
        ))
        
        # 5. ICa (L-type Ca2+)
        self.channels.append(Channel(
            name="ICa", is_Ca=True, color=(0.8, 0.5, 0.0),
            gates=[
                GateInfo('s', 2, as_Ca, bs_Ca),
                GateInfo('u', 1, au_Ca, bu_Ca)
            ]
        ))
        
        # 6. IA (Transient K+)
        self.channels.append(Channel(
            name="IA", color=(0.0, 0.7, 0.7),
            gates=[
                GateInfo('a', 1, aa_IA, ba_IA),
                GateInfo('b', 1, ab_IA, bb_IA)
            ]
        ))
        
        # 7. SK (Ca2+-activated K+)
        # Gate z_sk: ODE-based with tau_SK (Hirschberg 1998), z_inf depends on [Ca²⁺]
        self.channels.append(Channel(
            name="SK", color=(0.9, 0.1, 0.9),
            gates=[GateInfo('z', 1, None, None)]
        ))

        # 8. I_T (T-type Ca2+, low-threshold, CaV3.x — Destexhe 1998)
        self.channels.append(Channel(
            name="ITCa", is_Ca=True, color=(1.0, 0.8, 0.0),
            gates=[
                GateInfo('p', 2, am_TCa, bm_TCa),   # activation (m² in paper, 'p' to avoid confusion with Na m)
                GateInfo('q', 1, ah_TCa, bh_TCa)     # inactivation
            ]
        ))

        # 9. I_M (M-type K+, muscarinic-sensitive)
        self.channels.append(Channel(
            name="IM", color=(0.5, 0.8, 0.2),
            gates=[GateInfo('w', 1, aw_IM, bw_IM)]
        ))

        # 10. I_NaP (Persistent Na+ — Magistretti & Alonso 1999)
        self.channels.append(Channel(
            name="NaP", color=(1.0, 0.4, 0.4),
            gates=[GateInfo('x', 1, ax_NaP, bx_NaP)]  # single activation, no inactivation
        ))

        # 11. I_NaR (Resurgent Na+ — Raman & Bean 2001, phenomenological)
        self.channels.append(Channel(
            name="NaR", color=(0.8, 0.2, 0.6),
            gates=[
                GateInfo('y', 1, ay_NaR, by_NaR),   # activation
                GateInfo('j', 1, aj_NaR, bj_NaR)    # inactivation/block
            ]
        ))

    def compute_initial_states(self, V0: float, config) -> np.ndarray:
        """
        Вычисляет стационарные значения всех гейтов при потенциале V0.
        Создает начальный вектор y0 = [V, m, h, n, r, s, u, a, b, Ca]
        для всех компартментов.
        """
        mc = config.morphology
        if mc.single_comp:
            N = 1
        else:
            N = 1 + mc.N_ais + mc.N_trunk + mc.N_b1 + mc.N_b2
        y0_list = [np.full(N, V0)]

        def _append_gate(alpha_fn, beta_fn):
            a_val, b_val = alpha_fn(V0), beta_fn(V0)
            y0_list.append(np.full(N, a_val / (a_val + b_val)))

        # Gate order must match native_loop/rhs cursor offsets exactly.
        gate_specs = [
            (True, [(am, bm), (ah, bh), (an, bn)]),  # Na m/h, then K n
            (config.channels.enable_Ih, [(ar_Ih, br_Ih)]),
            (config.channels.enable_ICa, [(as_Ca, bs_Ca), (au_Ca, bu_Ca)]),
            (config.channels.enable_IA, [(aa_IA, ba_IA), (ab_IA, bb_IA)]),
            (config.channels.enable_ITCa, [(am_TCa, bm_TCa), (ah_TCa, bh_TCa)]),
            (config.channels.enable_IM, [(aw_IM, bw_IM)]),
            (config.channels.enable_NaP, [(ax_NaP, bx_NaP)]),
            (config.channels.enable_NaR, [(ay_NaR, by_NaR), (aj_NaR, bj_NaR)]),
        ]
        for enabled, gate_pairs in gate_specs:
            if not enabled:
                continue
            for alpha_fn, beta_fn in gate_pairs:
                _append_gate(alpha_fn, beta_fn)

        # SK gate z_sk: initialise at steady-state z_inf(Ca_rest)
        if config.channels.enable_SK:
            z0 = z_inf_SK(config.calcium.Ca_rest)
            y0_list.append(np.full(N, z0))

        # Динамика кальция
        if config.calcium.dynamic_Ca:
            y0_list.append(np.full(N, config.calcium.Ca_rest))
            
        # Динамика АТФ (metabolism) - LAST variable in state vector
        if config.metabolism.enable_dynamic_atp:
            y0_list.append(np.full(N, config.metabolism.atp_max_mM))
            y0_list.append(np.full(N, config.metabolism.na_i_rest_mM))
            y0_list.append(np.full(N, config.metabolism.k_o_rest_mM))

        return np.concatenate(y0_list)
