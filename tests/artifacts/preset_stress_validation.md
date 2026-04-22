# Preset Stress Validation Report

- Total cases: **1134**
- Overall status: **WARN**
- PASS: **987**
- WARN: **147**
- FAIL: **0**

## By Preset

| Preset | PASS | WARN | FAIL |
|---|---:|---:|---:|
| A: Squid Giant Axon (HH 1952) | 48 | 6 | 0 |
| B: Pyramidal L5 (Mainen 1996) | 48 | 6 | 0 |
| C: FS Interneuron (Wang-Buzsaki) | 54 | 0 | 0 |
| D: alpha-Motoneuron (Powers 2001) | 54 | 0 | 0 |
| E: Cerebellar Purkinje (De Schutter) | 54 | 0 | 0 |
| F: Multiple Sclerosis (Demyelination) | 54 | 0 | 0 |
| G: Local Anesthesia (gNa Block) | 54 | 0 | 0 |
| H: Severe Hyperkalemia (High EK) | 45 | 9 | 0 |
| I: In Vitro Slice (Mammalian 23°C) | 46 | 8 | 0 |
| J: C-Fiber (Pain / Unmyelinated) | 54 | 0 | 0 |
| K: Thalamic Relay (Ih + ITCa + Burst) | 48 | 6 | 0 |
| L: Hippocampal CA1 Pyramidal (Adapting) | 54 | 0 | 0 |
| M: Epilepsy (v10 SCN1A mutation) | 48 | 6 | 0 |
| N: Alzheimer's (v10 Calcium Toxicity) | 52 | 2 | 0 |
| O: Hypoxia (v10 ATP-pump failure) | 54 | 0 | 0 |
| P: Thalamic Reticular Nucleus (TRN Spindles) | 0 | 54 | 0 |
| Q: Striatal Spiny Projection (SPN) | 54 | 0 | 0 |
| R: Cholinergic Neuromodulation (ACh) | 10 | 44 | 0 |
| S: Pathology: Dravet Syndrome (SCN1A LOF) | 48 | 6 | 0 |
| T: Passive Cable (Linear Decay) | 54 | 0 | 0 |
| U: I-Clamp Threshold Explorer | 54 | 0 | 0 |

## WARN/FAIL samples

- `A: Squid Giant Axon (HH 1952)` | status=WARN | Iext×1.5, T=22.0°C, t_sim=300.0 ms | firing_hz outside expected range [0.0, 250.0]
- `A: Squid Giant Axon (HH 1952)` | status=WARN | Iext×1.5, T=22.0°C, t_sim=300.0 ms | firing_hz outside expected range [0.0, 250.0]
- `A: Squid Giant Axon (HH 1952)` | status=WARN | Iext×1.5, T=22.0°C, t_sim=800.0 ms | firing_hz outside expected range [0.0, 250.0]
- `A: Squid Giant Axon (HH 1952)` | status=WARN | Iext×1.5, T=22.0°C, t_sim=800.0 ms | firing_hz outside expected range [0.0, 250.0]
- `A: Squid Giant Axon (HH 1952)` | status=WARN | Iext×1.5, T=22.0°C, t_sim=1500.0 ms | firing_hz outside expected range [0.0, 250.0]
- `A: Squid Giant Axon (HH 1952)` | status=WARN | Iext×1.5, T=22.0°C, t_sim=1500.0 ms | firing_hz outside expected range [0.0, 250.0]
- `B: Pyramidal L5 (Mainen 1996)` | status=WARN | Iext×0.5, T=37.0°C, t_sim=300.0 ms | v_max outside expected range [0.0, 120.0]
- `B: Pyramidal L5 (Mainen 1996)` | status=WARN | Iext×0.5, T=37.0°C, t_sim=300.0 ms | v_max outside expected range [0.0, 120.0]
- `B: Pyramidal L5 (Mainen 1996)` | status=WARN | Iext×0.5, T=37.0°C, t_sim=800.0 ms | v_max outside expected range [0.0, 120.0]
- `B: Pyramidal L5 (Mainen 1996)` | status=WARN | Iext×0.5, T=37.0°C, t_sim=800.0 ms | v_max outside expected range [0.0, 120.0]
- `B: Pyramidal L5 (Mainen 1996)` | status=WARN | Iext×0.5, T=37.0°C, t_sim=1500.0 ms | v_max outside expected range [0.0, 120.0]
- `B: Pyramidal L5 (Mainen 1996)` | status=WARN | Iext×0.5, T=37.0°C, t_sim=1500.0 ms | v_max outside expected range [0.0, 120.0]
- `H: Severe Hyperkalemia (High EK)` | status=WARN | Iext×0.5, T=22.0°C, t_sim=300.0 ms | firing_hz outside expected range [0.0, 200.0]
- `H: Severe Hyperkalemia (High EK)` | status=WARN | Iext×0.5, T=22.0°C, t_sim=300.0 ms | firing_hz outside expected range [0.0, 200.0]
- `H: Severe Hyperkalemia (High EK)` | status=WARN | Iext×0.5, T=22.0°C, t_sim=800.0 ms | firing_hz outside expected range [0.0, 200.0]
- `H: Severe Hyperkalemia (High EK)` | status=WARN | Iext×0.5, T=22.0°C, t_sim=800.0 ms | firing_hz outside expected range [0.0, 200.0]
- `H: Severe Hyperkalemia (High EK)` | status=WARN | Iext×0.5, T=22.0°C, t_sim=1500.0 ms | firing_hz outside expected range [0.0, 200.0]
- `H: Severe Hyperkalemia (High EK)` | status=WARN | Iext×0.5, T=22.0°C, t_sim=1500.0 ms | firing_hz outside expected range [0.0, 200.0]
- `H: Severe Hyperkalemia (High EK)` | status=WARN | Iext×1.0, T=22.0°C, t_sim=300.0 ms | firing_hz outside expected range [0.0, 200.0]
- `H: Severe Hyperkalemia (High EK)` | status=WARN | Iext×1.0, T=22.0°C, t_sim=800.0 ms | firing_hz outside expected range [0.0, 200.0]
- `H: Severe Hyperkalemia (High EK)` | status=WARN | Iext×1.0, T=22.0°C, t_sim=1500.0 ms | firing_hz outside expected range [0.0, 200.0]
- `I: In Vitro Slice (Mammalian 23°C)` | status=WARN | Iext×0.5, T=30.0°C, t_sim=300.0 ms | v_max outside expected range [0.0, 120.0]
- `I: In Vitro Slice (Mammalian 23°C)` | status=WARN | Iext×0.5, T=37.0°C, t_sim=300.0 ms | v_max outside expected range [0.0, 120.0]
- `I: In Vitro Slice (Mammalian 23°C)` | status=WARN | Iext×0.5, T=37.0°C, t_sim=300.0 ms | v_max outside expected range [0.0, 120.0]
- `I: In Vitro Slice (Mammalian 23°C)` | status=WARN | Iext×0.5, T=37.0°C, t_sim=800.0 ms | v_max outside expected range [0.0, 120.0]
- `I: In Vitro Slice (Mammalian 23°C)` | status=WARN | Iext×0.5, T=37.0°C, t_sim=800.0 ms | v_max outside expected range [0.0, 120.0]
- `I: In Vitro Slice (Mammalian 23°C)` | status=WARN | Iext×0.5, T=30.0°C, t_sim=1500.0 ms | v_max outside expected range [0.0, 120.0]
- `I: In Vitro Slice (Mammalian 23°C)` | status=WARN | Iext×0.5, T=37.0°C, t_sim=1500.0 ms | v_max outside expected range [0.0, 120.0]
- `I: In Vitro Slice (Mammalian 23°C)` | status=WARN | Iext×0.5, T=37.0°C, t_sim=1500.0 ms | v_max outside expected range [0.0, 120.0]
- `K: Thalamic Relay (Ih + ITCa + Burst)` | status=WARN | Iext×0.5, T=37.0°C, t_sim=300.0 ms | v_max outside expected range [0.0, 120.0]
- `K: Thalamic Relay (Ih + ITCa + Burst)` | status=WARN | Iext×0.5, T=37.0°C, t_sim=300.0 ms | v_max outside expected range [0.0, 120.0]
- `K: Thalamic Relay (Ih + ITCa + Burst)` | status=WARN | Iext×0.5, T=37.0°C, t_sim=800.0 ms | v_max outside expected range [0.0, 120.0]
- `K: Thalamic Relay (Ih + ITCa + Burst)` | status=WARN | Iext×0.5, T=37.0°C, t_sim=800.0 ms | v_max outside expected range [0.0, 120.0]
- `K: Thalamic Relay (Ih + ITCa + Burst)` | status=WARN | Iext×0.5, T=37.0°C, t_sim=1500.0 ms | v_max outside expected range [0.0, 120.0]
- `K: Thalamic Relay (Ih + ITCa + Burst)` | status=WARN | Iext×0.5, T=37.0°C, t_sim=1500.0 ms | v_max outside expected range [0.0, 120.0]
- `M: Epilepsy (v10 SCN1A mutation)` | status=WARN | Iext×1.5, T=37.0°C, t_sim=300.0 ms | firing_hz outside expected range [0.0, 250.0]
- `M: Epilepsy (v10 SCN1A mutation)` | status=WARN | Iext×1.5, T=37.0°C, t_sim=300.0 ms | firing_hz outside expected range [0.0, 250.0]
- `M: Epilepsy (v10 SCN1A mutation)` | status=WARN | Iext×1.5, T=37.0°C, t_sim=800.0 ms | firing_hz outside expected range [0.0, 250.0]
- `M: Epilepsy (v10 SCN1A mutation)` | status=WARN | Iext×1.5, T=37.0°C, t_sim=800.0 ms | firing_hz outside expected range [0.0, 250.0]
- `M: Epilepsy (v10 SCN1A mutation)` | status=WARN | Iext×1.5, T=37.0°C, t_sim=1500.0 ms | firing_hz outside expected range [0.0, 250.0]
- `M: Epilepsy (v10 SCN1A mutation)` | status=WARN | Iext×1.5, T=37.0°C, t_sim=1500.0 ms | firing_hz outside expected range [0.0, 250.0]
- `N: Alzheimer's (v10 Calcium Toxicity)` | status=WARN | Iext×1.5, T=37.0°C, t_sim=1500.0 ms | firing_hz outside expected range [0.0, 250.0]
- `N: Alzheimer's (v10 Calcium Toxicity)` | status=WARN | Iext×1.5, T=37.0°C, t_sim=1500.0 ms | firing_hz outside expected range [0.0, 250.0]
- `P: Thalamic Reticular Nucleus (TRN Spindles)` | status=WARN | Iext×0.5, T=22.0°C, t_sim=300.0 ms | v_max outside expected range [0.0, 120.0]
- `P: Thalamic Reticular Nucleus (TRN Spindles)` | status=WARN | Iext×0.5, T=22.0°C, t_sim=300.0 ms | v_max outside expected range [0.0, 120.0]
- `P: Thalamic Reticular Nucleus (TRN Spindles)` | status=WARN | Iext×0.5, T=30.0°C, t_sim=300.0 ms | v_max outside expected range [0.0, 120.0]
- `P: Thalamic Reticular Nucleus (TRN Spindles)` | status=WARN | Iext×0.5, T=30.0°C, t_sim=300.0 ms | v_max outside expected range [0.0, 120.0]
- `P: Thalamic Reticular Nucleus (TRN Spindles)` | status=WARN | Iext×0.5, T=37.0°C, t_sim=300.0 ms | v_max outside expected range [0.0, 120.0]
- `P: Thalamic Reticular Nucleus (TRN Spindles)` | status=WARN | Iext×0.5, T=37.0°C, t_sim=300.0 ms | v_max outside expected range [0.0, 120.0]
- `P: Thalamic Reticular Nucleus (TRN Spindles)` | status=WARN | Iext×0.5, T=22.0°C, t_sim=800.0 ms | v_max outside expected range [0.0, 120.0]