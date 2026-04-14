"""
core/expert_system.py - Biophysical Expert System for Automated Interpretation

Provides rule-based expert insights for neuron simulation results.
Analyzes firing patterns, metabolic state, and dynamical properties
to generate human-readable scientific interpretations.

BILINGUAL SUPPORT:
All expert messages support Russian (RU) and English (EN) languages.
Use set_language() to switch output language globally.
"""

from typing import Callable, Dict, List, Any, Optional

# ── PHYSICAL CONSTANTS (must be defined BEFORE use in lambda rules) ──
ATP_ISCHEMIC_THRESHOLD = 0.5  # mM - K_ATP opens below this threshold

# ── GLOBAL LANGUAGE SETTINGS ──
_CURRENT_LANGUAGE: str = "EN"  # Default language: "EN" or "RU"


def set_language(lang: str) -> None:
    """Set output language for expert insights.
    
    Parameters
    ----------
    lang : str
        Language code: "EN" for English, "RU" for Russian
    """
    global _CURRENT_LANGUAGE
    _CURRENT_LANGUAGE = lang.upper() if lang.upper() in ("EN", "RU") else "EN"


def get_language() -> str:
    """Get current language setting.
    
    Returns
    -------
    str
        Current language code: "EN" or "RU"
    """
    return _CURRENT_LANGUAGE


# Expert rule type definition with bilingual support
ExpertRule = Dict[str, Any]

# Default expert rules for neurophysiological interpretation
DEFAULT_EXPERT_RULES: List[ExpertRule] = [
    {
        "id": "metabolic_warning",
        "condition": lambda s: (
            s.get('firing_rate_hz', 0) > 100 and 
            s.get('atp_min_mM', 10) < 0.5
        ),
        "message_en": (
            "⚠️ **Metabolic Warning**: High firing rate ({:.1f} Hz) with critically low ATP "
            "({:.2f} mM). Depolarization block imminent. Reduce stimulation or increase ATP synthesis."
        ),
        "message_ru": (
            "⚠️ **Метаболическое предупреждение**: Высокая частота ({:.1f} Гц) при критически низком АТФ "
            "({:.2f} мМ). Неминуемый деполяризационный блок. Уменьшите стимул или увеличьте синтез АТФ."
        ),
        "severity": "critical",
        "format_args": lambda s: [s.get('firing_rate_hz', 0), s.get('atp_min_mM', 0)]
    },
    {
        "id": "energy_crisis",
        "condition": lambda s: s.get('atp_min_mM', 10) < ATP_ISCHEMIC_THRESHOLD,
        "message_en": (
            "🔋 **Energy Crisis**: ATP below ischemic threshold ({:.2f} mM < {:.1f} mM). "
            "K_ATP channels activate. Neuron enters protective down-regulation."
        ),
        "message_ru": (
            "🔋 **Энергетический кризис**: АТФ ниже ишемического порога ({:.2f} мМ < {:.1f} мМ). "
            "Активируются K_ATP каналы. Нейрон входит в защитное торможение."
        ),
        "severity": "warning",
        "format_args": lambda s: [s.get('atp_min_mM', 0), ATP_ISCHEMIC_THRESHOLD]
    },
    {
        "id": "chaos_detected",
        "condition": lambda s: s.get('lle_per_ms', -1) > 0,
        "message_en": (
            "🔀 **Chaos Detected**: Positive LLE ({:.4f} 1/ms = {:.2f} 1/s). "
            "Firing is chaotic - highly sensitive to initial conditions. "
            "Predictability: ~{:.1f} ms."
        ),
        "message_ru": (
            "🔀 **Хаос обнаружен**: Положительный ЛЯ ({:.4f} 1/мс = {:.2f} 1/с). "
            "Разрядка хаотична - высокая чувствительность к начальным условиям. "
            "Предсказуемость: ~{:.1f} мс."
        ),
        "severity": "info",
        "format_args": lambda s: [
            s.get('lle_per_ms', 0),
            s.get('lle_per_ms', 0) * 1000,
            1.0 / max(s.get('lle_per_ms', 1e-6), 1e-6)
        ]
    },
    {
        "id": "adaptation_phenotype",
        "condition": lambda s: s.get('adaptation_index', 0) > 0.3,
        "message_en": (
            "📊 **Phenotype**: Strong spike-frequency adaptation (index={:.2f}). "
            "IM/SK currents highly active. Type-A behavior typical of regular spiking (RS) neurons."
        ),
        "message_ru": (
            "📊 **Фенотип**: Сильная адаптация частоты спайков (индекс={:.2f}). "
            "IM/SK токи активны. Поведение типа-A, характерное для RS нейронов коры."
        ),
        "severity": "info",
        "format_args": lambda s: [s.get('adaptation_index', 0)]
    },
    {
        "id": "bursting_detected",
        "condition": lambda s: (
            s.get('cv_isi', 0) > 0.5 and 
            s.get('burst_spike_ratio', 0) > 0.3
        ),
        "message_en": (
            "💥 **Phenotype**: Bursting detected. CV(ISI)={:.2f}, burst ratio={:.1%}. "
            "ITCa/ICa drive high-threshold bursting typical of thalamocortical relay neurons."
        ),
        "message_ru": (
            "💥 **Фенотип**: Обнаружено пакетное разряжение. CV(ISI)={:.2f}, доля пакетов={:.1%}. "
            "ITCa/ICa обеспечивают высокопороговые пакеты, характерные для таламокортикальных нейронов."
        ),
        "severity": "info",
        "format_args": lambda s: [s.get('cv_isi', 0), s.get('burst_spike_ratio', 0)]
    },
    {
        "id": "hyperexcitable",
        "condition": lambda s: (
            s.get('firing_rate_hz', 0) > 50 and 
            s.get('threshold_rheobase_pA', 100) < 50
        ),
        "message_en": (
            "⚡ **Pathology Alert**: Hyperexcitable membrane. Very low rheobase ({:.1f} pA). "
            "Check gNaP/gNaR - possible epileptiform conditions or channelopathy."
        ),
        "message_ru": (
            "⚡ **Патологическая гипервозбудимость**: Очень низкий реобаз ({:.1f} пА). "
            "Проверьте gNaP/gNaR - возможны эпилептиформные условия или каналопатия."
        ),
        "severity": "warning",
        "format_args": lambda s: [s.get('threshold_rheobase_pA', 0)]
    },
    {
        "id": "calcium_overload",
        "condition": lambda s: s.get('ca_i_max_nM', 0) > 1000,
        "message_en": (
            "🧪 **Calcium Overload**: Peak [Ca²⁺]ᵢ = {:.1f} nM > 1000 nM threshold. "
            "Risk of excitotoxic cascade. Reduce ICa/ITCa or add calcium buffers."
        ),
        "message_ru": (
            "🧪 **Кальциевая перегрузка**: Пик [Ca²⁺]ᵢ = {:.1f} нМ > 1000 нМ порог. "
            "Риск эксцитотоксичности. Уменьшите ICa/ITCa или добавьте буферы кальция."
        ),
        "severity": "warning",
        "format_args": lambda s: [s.get('ca_i_max_nM', 0)]
    },
    {
        "id": "stable_periodic",
        "condition": lambda s: (
            s.get('cv_isi', 1) < 0.1 and 
            s.get('firing_rate_hz', 0) > 0
        ),
        "message_en": (
            "🎯 **Dynamical State**: Highly periodic firing (CV={:.3f}). "
            "Stable limit cycle attractor. Reliable encoding channel."
        ),
        "message_ru": (
            "🎯 **Динамическое состояние**: Высокопериодичная разрядка (CV={:.3f}). "
            "Стабильный предельный цикл. Надёжный канал кодирования."
        ),
        "severity": "info",
        "format_args": lambda s: [s.get('cv_isi', 0)]
    },
    {
        "id": "silent_neuron",
        "condition": lambda s: (
            s.get('n_spikes', 0) == 0 and 
            s.get('stim_amplitude_pA', 0) > 100
        ),
        "message_en": (
            "🔇 **Silent Neuron**: No spikes at {:.0f} pA. Check rheobase - "
            "possible high gK or low gNa. Increase stimulus or reduce K+ conductances."
        ),
        "message_ru": (
            "🔇 **Молчаливый нейрон**: Нет спайков при {:.0f} пА. Проверьте реобаз - "
            "возможно высокий gK или низкий gNa. Увеличьте стимул или уменьшите K+ токи."
        ),
        "severity": "info",
        "format_args": lambda s: [s.get('stim_amplitude_pA', 0)]
    },
    {
        "id": "depolarization_block",
        "condition": lambda s: (
            s.get('n_spikes', 0) == 0 and 
            s.get('mean_v_mV', -70) > -40 and
            s.get('stim_amplitude_pA', 0) > 200
        ),
        "message_en": (
            "🚫 **Depolarization Block**: High current blocked firing (V_mean={:.1f} mV). "
            "Neuron stuck depolarized. Reduce stimulus or check Na+ inactivation."
        ),
        "message_ru": (
            "🚫 **Деполяризационный блок**: Сильный ток заблокировал разрядку (V={:.1f} мВ). "
            "Нейрон застрял в деполяризованном состоянии. Уменьшите стимул или проверьте инактивацию Na+."
        ),
        "severity": "warning",
        "format_args": lambda s: [s.get('mean_v_mV', -70)]
    },
    {
        "id": "extreme_frequency",
        "condition": lambda s: s.get('firing_rate_hz', 0) > 200,
        "message_en": (
            "⚠️ **Extreme Frequency**: {:.1f} Hz > 200 Hz threshold. "
            "Risk of Na+ inactivation and metabolic exhaustion. Verify Q10 and gNa."
        ),
        "message_ru": (
            "⚠️ **Экстремальная частота**: {:.1f} Гц > 200 Гц порог. "
            "Риск инактивации Na+ и метаболистического истощения. Проверьте Q10 и gNa."
        ),
        "severity": "warning",
        "format_args": lambda s: [s.get('firing_rate_hz', 0)]
    },
    {
        "id": "metabolic_exhaustion",
        "condition": lambda s: (
            s.get('atp_decline_rate_mM_per_s', 0) > 0.1 and
            s.get('atp_min_mM', 2) < 1.0
        ),
        "message_en": (
            "🔥 **Metabolic Exhaustion**: ATP declining at {:.3f} mM/s. "
            "Rapid depletion limits firing. Reduce duration or increase ATP synthesis."
        ),
        "message_ru": (
            "🔥 **Метаболистическое истощение**: АТФ падает на {:.3f} мМ/с. "
            "Быстрое истощение ограничивает разрядку. Уменьшите длительность или увеличьте синтез АТФ."
        ),
        "severity": "warning",
        "format_args": lambda s: [s.get('atp_decline_rate_mM_per_s', 0)]
    },
    {
        "id": "temperature_compensation",
        "condition": lambda s: s.get('temperature_celsius', 37) > 40,
        "message_en": (
            "🌡️ **Temperature Warning**: {:.1f}°C > physiological range (36-38°C). "
            "High T accelerates kinetics. Check Q10 and gating constants."
        ),
        "message_ru": (
            "🌡️ **Предупреждение температуры**: {:.1f}°C > физиологического диапазона (36-38°C). "
            "Высокая T ускоряет кинетику. Проверьте Q10 и константы гейтинга."
        ),
        "severity": "info",
        "format_args": lambda s: [s.get('temperature_celsius', 37)]
    },
    {
        "id": "low_temperature",
        "condition": lambda s: s.get('temperature_celsius', 37) < 20,
        "message_en": (
            "❄️ **Low Temperature**: {:.1f}°C - room T simulation. "
            "Kinetics slowed ~3× (Q10). Expect lower rates and longer latencies."
        ),
        "message_ru": (
            "❄️ **Низкая температура**: {:.1f}°C - комнатная T. "
            "Кинетика замедлена ~3× (Q10). Ожидайте меньшие частоты и большие задержки."
        ),
        "severity": "info",
        "format_args": lambda s: [s.get('temperature_celsius', 20)]
    },
    {
        "id": "stochastic_dominance",
        "condition": lambda s: (
            s.get('cv_isi', 0) > 1.0 and 
            s.get('isi_std_ms', 0) > s.get('isi_mean_ms', 100) and
            s.get('isi_mean_ms', 0) > 0  # Ensure valid mean (not NaN)
        ),
        "message_en": (
            "🎲 **Stochastic Firing**: CV(ISI)={:.2f} > 1. Poisson-like spiking. "
            "Noise dominates dynamics. Check OU noise or synaptic background."
        ),
        "message_ru": (
            "🎲 **Стохастическая разрядка**: CV(ISI)={:.2f} > 1. Пуассоноподобная. "
            "Шум доминирует над динамикой. Проверьте шум OU или синаптический фон."
        ),
        "severity": "info",
        "format_args": lambda s: [s.get('cv_isi', 0)]
    },
    # ── v11.7 NEW RULES ──
    {
        "id": "refractory_abnormal",
        "condition": lambda s: (
            s.get('refractory_period_ms', 5) < 1.0 and 
            s.get('n_spikes', 0) > 5
        ),
        "message_en": (
            "⚡ **Abnormal Refractory Period**: {:.2f} ms < 1 ms. "
            "Extremely short recovery. Check gK dynamics and AHP currents (IM/SK)."
        ),
        "message_ru": (
            "⚡ **Аномальный рефрактерный период**: {:.2f} мс < 1 мс. "
            "Экстремально короткое восстановление. Проверьте динамику gK и AHP токи (IM/SK)."
        ),
        "severity": "warning",
        "format_args": lambda s: [s.get('refractory_period_ms', 1.0)]
    },
    {
        "id": "hyperpolarized_silent",
        "condition": lambda s: (
            s.get('n_spikes', 0) == 0 and 
            s.get('mean_v_mV', -70) < -75 and
            s.get('stim_amplitude_pA', 0) > 50
        ),
        "message_en": (
            "🔋 **Hyperpolarized Silence**: V_mean={:.1f} mV, no spikes at {:.0f} pA. "
            "Strong K+ leak or Ih current. Reduce gL or increase Iext."
        ),
        "message_ru": (
            "🔋 **Гиперполяризованное молчание**: V={:.1f} мВ, нет спайков при {:.0f} пА. "
            "Сильный K+ утечка или ток Ih. Уменьшите gL или увеличьте Iext."
        ),
        "severity": "info",
        "format_args": lambda s: [s.get('mean_v_mV', -70), s.get('stim_amplitude_pA', 0)]
    },
    {
        "id": "synaptic_dominance",
        "condition": lambda s: (
            s.get('synaptic_charge_ratio', 0) > 0.5 and 
            s.get('n_spikes', 0) > 0
        ),
        "message_en": (
            "🔗 **Synaptic Drive Dominant**: {:.1%} of charge from synapses. "
            "Network-coupled behavior. Check synaptic reversal and conductance scaling."
        ),
        "message_ru": (
            "🔗 **Синаптическое доминирование**: {:.1%} заряда от синапсов. "
            "Сетевое поведение. Проверьте реверсал синапсов и масштабирование проводимости."
        ),
        "severity": "info",
        "format_args": lambda s: [s.get('synaptic_charge_ratio', 0)]
    },
    {
        "id": "prominent_ahp",
        "condition": lambda s: (
            s.get('V_ahp_mV', -65) < -75 and 
            s.get('n_spikes', 0) > 3
        ),
        "message_en": (
            "📉 **Prominent Afterhyperpolarization**: AHP={:.1f} mV. "
            "Strong K+ activation limits burst capability. Check SK/IKCa currents."
        ),
        "message_ru": (
            "📉 **Выраженная послейперполяризация**: AHP={:.1f} мВ. "
            "Сильная активация K+ ограничивает пакетную способность. Проверьте токи SK/IKCa."
        ),
        "severity": "info",
        "format_args": lambda s: [s.get('V_ahp_mV', -65)]
    },
    {
        "id": "irregular_bursting",
        "condition": lambda s: (
            s.get('burst_spike_ratio', 0) > 0.2 and 
            s.get('cv_isi', 0) > 0.3 and
            s.get('n_spikes', 0) > 10
        ),
        "message_en": (
            "🎆 **Irregular Bursting**: {:.1%} burst ratio with CV={:.2f}. "
            "Mixed single-spike and burst modes. Typical of hippocampal CA3/pyramidal neurons."
        ),
        "message_ru": (
            "🎆 **Нерегулярное пакетное разряжение**: {:.1%} пакетов при CV={:.2f}. "
            "Смешанные одиночные и пакетные режимы. Типично для CA3/пирамидальных нейронов."
        ),
        "severity": "info",
        "format_args": lambda s: [s.get('burst_spike_ratio', 0), s.get('cv_isi', 0)]
    },
]


def generate_expert_insights(stats: Dict[str, Any], 
                             rules: List[ExpertRule] = None,
                             language: Optional[str] = None) -> List[str]:
    """Generate expert insights from simulation statistics.
    
    Applies biophysical rules to identify interesting phenomena
    and returns formatted insight messages in the specified language.
    
    Parameters
    ----------
    stats : dict
        Dictionary of simulation statistics including:
        - firing_rate_hz: Mean firing frequency
        - atp_min_mM: Minimum ATP concentration reached
        - lle_per_ms: Lyapunov exponent (1/ms)
        - adaptation_index: Spike-frequency adaptation metric
        - cv_isi: Coefficient of variation of ISIs
        - ca_i_max_nM: Peak calcium concentration
        - threshold_rheobase_pA: Rheobase current
        - burst_spike_ratio: Fraction of spikes in bursts
        
    rules : list, optional
        Custom rule set. Defaults to DEFAULT_EXPERT_RULES.
        
    language : str, optional
        Language code: "EN" or "RU". Defaults to global _CURRENT_LANGUAGE.
        
    Returns
    -------
    list[str]
        Formatted insight messages, sorted by severity
        (critical > warning > info)
    """
    if rules is None:
        rules = DEFAULT_EXPERT_RULES
    
    # Use provided language or fall back to global setting
    lang = (language or _CURRENT_LANGUAGE).upper()
    
    insights = []
    
    for rule in rules:
        try:
            # Evaluate condition
            if rule["condition"](stats):
                # Select message based on language
                msg_key = f"message_{lang.lower()}"
                if msg_key in rule:
                    message_template = rule[msg_key]
                elif "message_en" in rule:
                    message_template = rule["message_en"]  # Fallback to EN
                elif "message" in rule:
                    message_template = rule["message"]  # Legacy fallback
                else:
                    continue  # Skip rule if no message found
                
                # Format message with dynamic args if provided
                if "format_args" in rule:
                    args = rule["format_args"](stats)
                    message = message_template.format(*args)
                else:
                    message = message_template
                
                insights.append({
                    "severity": rule["severity"],
                    "message": message,
                    "id": rule["id"]
                })
        except Exception:
            # Silently skip rules that fail to evaluate
            # (e.g., missing keys in stats)
            continue
    
    # Sort by severity priority
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    insights.sort(key=lambda x: severity_order.get(x["severity"], 3))
    
    # Return just the formatted messages
    return [insight["message"] for insight in insights]


def get_severity_emoji(severity: str) -> str:
    """Get emoji prefix for severity level."""
    return {
        "critical": "🔴",
        "warning": "🟡",
        "info": "🟢"
    }.get(severity, "⚪")


def format_insights_html(insights: List[str], language: Optional[str] = None) -> str:
    """Format insights as HTML for rich text display.
    
    Parameters
    ----------
    insights : list[str]
        Insight messages from generate_expert_insights()
    language : str, optional
        Language for empty message. Defaults to global setting.
        
    Returns
    -------
    str
        HTML formatted string
    """
    lang = (language or _CURRENT_LANGUAGE).upper()
    
    if not insights:
        empty_msg = (
            "<p><i>Нет специфических инсайтов для этой симуляции.</i></p>" 
            if lang == "RU" 
            else "<p><i>No specific insights for this simulation.</i></p>"
        )
        return empty_msg
    
    html_parts = ["<ul>"]
    for insight in insights:
        html_parts.append(f"<li style='margin-bottom: 8px;'>{insight}</li>")
    html_parts.append("</ul>")
    
    return "\n".join(html_parts)


def format_insights_markdown(insights: List[str], language: Optional[str] = None) -> str:
    """Format insights as Markdown for text display.
    
    Parameters
    ----------
    insights : list[str]
        Insight messages from generate_expert_insights()
    language : str, optional
        Language for empty message. Defaults to global setting.
        
    Returns
    -------
    str
        Markdown formatted string
    """
    lang = (language or _CURRENT_LANGUAGE).upper()
    
    if not insights:
        return "_Нет специфических инсайтов_" if lang == "RU" else "_No specific insights for this simulation._"
    
    return "\n\n".join(f"• {insight}" for insight in insights)


def get_quick_recommendations(stats: Dict[str, Any], language: Optional[str] = None) -> List[str]:
    """Generate actionable recommendations based on simulation results.
    
    Provides specific parameter adjustments to improve simulation outcomes.
    Useful for quick troubleshooting and parameter tuning.
    
    Parameters
    ----------
    stats : dict
        Simulation statistics dictionary
    language : str, optional
        Language code: "EN" or "RU". Defaults to global _CURRENT_LANGUAGE.
        
    Returns
    -------
    list[str]
        List of actionable recommendations
    """
    lang = (language or _CURRENT_LANGUAGE).upper()
    recs = []
    
    # Bilingual recommendation dictionaries
    _REC_TEXTS = {
        "silent_low_stim": {
            "EN": "→ Increase stimulus amplitude above 100 pA to reach threshold",
            "RU": "→ Увеличьте амплитуду стимула выше 100 пА для достижения порога"
        },
        "silent_high_stim": {
            "EN": "→ Check rheobase: reduce gK or increase gNa if neuron won't fire",
            "RU": "→ Проверьте реобаз: уменьшите gK или увеличьте gNa если нейрон не разряжается"
        },
        "high_freq": {
            "EN": "→ Reduce stimulus or increase adaptation (gIM, gSK) to lower frequency",
            "RU": "→ Уменьшите стимул или увеличьте адаптацию (gIM, gSK) для снижения частоты"
        },
        "low_atp": {
            "EN": "→ Increase ATP synthesis rate or reduce stimulation duration",
            "RU": "→ Увеличьте скорость синтеза АТФ или уменьшите длительность стимуляции"
        },
        "high_ca": {
            "EN": "→ Reduce gCa or gTCa, or increase calcium buffering",
            "RU": "→ Уменьшите gCa или gTCa, или увеличьте буферизацию кальция"
        },
        "high_chaos": {
            "EN": "→ High chaos: consider parameter scan for bifurcation points",
            "RU": "→ Высокий хаос: рассмотрите сканирование параметров для точек бифуркации"
        },
        "high_temp": {
            "EN": "→ Reduce temperature to 37°C for physiological kinetics",
            "RU": "→ Снизьте температуру до 37°C для физиологической кинетики"
        },
        "low_temp": {
            "EN": "→ Increase temperature to 37°C for physiological firing rates",
            "RU": "→ Повысьте температуру до 37°C для физиологических частот разрядки"
        },
        "balanced": {
            "EN": "→ Parameters appear well-balanced for this simulation",
            "RU": "→ Параметры хорошо сбалансированы для этой симуляции"
        }
    }
    
    def _get(key):
        return _REC_TEXTS[key][lang]
    
    # Firing-related recommendations
    if stats.get('n_spikes', 0) == 0:
        if stats.get('stim_amplitude_pA', 0) < 100:
            recs.append(_get("silent_low_stim"))
        else:
            recs.append(_get("silent_high_stim"))
    
    if stats.get('firing_rate_hz', 0) > 200:
        recs.append(_get("high_freq"))
    
    # Metabolic recommendations
    if stats.get('atp_min_mM', 2) < 0.5:
        recs.append(_get("low_atp"))
    
    # Calcium recommendations
    if stats.get('ca_i_max_nM', 0) > 1000:
        recs.append(_get("high_ca"))
    
    # Stability recommendations
    if stats.get('lle_per_ms', -1) > 0.1:
        recs.append(_get("high_chaos"))
    
    # Temperature recommendations
    if stats.get('temperature_celsius', 37) > 40:
        recs.append(_get("high_temp"))
    elif stats.get('temperature_celsius', 37) < 25:
        recs.append(_get("low_temp"))
    
    return recs if recs else [_get("balanced")]


def generate_full_report(stats: Dict[str, Any], language: Optional[str] = None) -> Dict[str, Any]:
    """Generate comprehensive expert report with insights and recommendations.
    
    Research-grade summary combining insights, recommendations, and severity summary.
    
    Parameters
    ----------
    stats : dict
        Simulation statistics
    language : str, optional
        Language code: "EN" or "RU". Defaults to global _CURRENT_LANGUAGE.
        
    Returns
    -------
    dict
        Report with keys: insights, recommendations, severity_counts, summary
    """
    lang = (language or _CURRENT_LANGUAGE).upper()
    
    insights_raw = generate_expert_insights(stats, language=lang)
    recommendations = get_quick_recommendations(stats, language=lang)
    
    # Count severities (re-evaluate rules for counting)
    severity_counts = {"critical": 0, "warning": 0, "info": 0}
    for rule in DEFAULT_EXPERT_RULES:
        try:
            if rule["condition"](stats):
                severity_counts[rule["severity"]] += 1
        except Exception:
            pass
    
    # Bilingual summary templates
    _SUMMARIES = {
        "critical": {
            "EN": lambda n: f"⚠️ {n} critical issues detected - review recommended",
            "RU": lambda n: f"⚠️ Обнаружено {n} критических проблем - требуется проверка"
        },
        "warning": {
            "EN": lambda n: f"ℹ️ {n} warnings - minor adjustments suggested",
            "RU": lambda n: f"ℹ️ {n} предупреждений - рекомендуются небольшие корректировки"
        },
        "ok": {
            "EN": "✅ Simulation appears well-parameterized",
            "RU": "✅ Симуляция хорошо параметризована"
        }
    }
    
    # Generate summary text
    if severity_counts["critical"] > 0:
        summary = _SUMMARIES["critical"][lang](severity_counts["critical"])
    elif severity_counts["warning"] > 0:
        summary = _SUMMARIES["warning"][lang](severity_counts["warning"])
    else:
        summary = _SUMMARIES["ok"][lang]
    
    return {
        "insights": insights_raw,
        "recommendations": recommendations,
        "severity_counts": severity_counts,
        "summary": summary,
        "n_insights": len(insights_raw),
        "n_recommendations": len(recommendations),
        "language": lang
    }
