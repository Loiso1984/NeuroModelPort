# 📚 NeuroModelPort v10.1 — Documentation Index

**Last Updated:** 2026-04-01  
**Status:** Phase 7 Complete (Dual Stimulation Integrated)

---

## 🚀 START HERE

- **[README.md](../README.md)** — Project overview & quick start
- **[USER_GUIDE.md](USER_GUIDE.md)** — How to use the GUI (with screenshots)
- **[QUICK_START.md](QUICK_START.md)** — 5-minute tutorial for new users

---

## 🏗️ ARCHITECTURE & DESIGN

| Document | Purpose |
|----------|---------|
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | System design, module structure, data flow |
| **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)** | How to extend / modify the code |
| **[API_REFERENCE.md](API_REFERENCE.md)** | Classes, functions, signatures |

---

## 🧠 SCIENTIFIC CONTENT

| Document | Topic |
|----------|-------|
| **[PRESETS_GUIDE.md](PRESETS_GUIDE.md)** | Understanding all neuron presets (8+ types) |
| **[DUAL_STIMULATION.md](DUAL_STIMULATION.md)** | Multi-site voltage clamp (Phase 7) |
| **[LITERATURE_VALUES.md](LITERATURE_VALUES.md)** | Channel parameters from published papers |

---

## 📊 ANALYSIS & FEATURES

| Feature | Documentation |
|---------|---------------|
| **Dendritic Filtering** | See [ARCHITECTURE.md](ARCHITECTURE.md#dendritic-compartments) |
| **Strength-Duration Curve** | See [USER_GUIDE.md](USER_GUIDE.md#sd-curve) |
| **Excitability Mapping** | See [USER_GUIDE.md](USER_GUIDE.md#excitability-map) |
| **Stochastic Simulation** | See [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md#stochastic-mode) |

---

## 📁 DOCUMENT ORGANIZATION

```
docs/
├── INDEX.md                    ← You are here
├── README_DOCS.md              ← Documentation guide
├── ARCHITECTURE.md             ← System design
├── USER_GUIDE.md              ← How to use (bilingual RU/EN)
├── DEVELOPER_GUIDE.md         ← For contributors  
├── QUICK_START.md             ← 5-minute tutorial
├── API_REFERENCE.md           ← Classes & functions
├── PRESETS_GUIDE.md           ← All neuron types
├── DUAL_STIMULATION.md        ← Phase 7 feature
├── LITERATURE_VALUES.md       ← Research references
│
├── guides/                     ← Topic-specific guides
│   ├── GUI_TOUR.md
│   ├── ANALYSIS_PLOTS.md
│   ├── PARAMETER_TUNING.md
│   └── TROUBLESHOOTING.md
│
└── archive/                    ← Old documentation
    ├── PHASE_6_COMPLETION.md
    ├── AUDIT_*.md
    └── v10_1_ADDITIONS.md
```

---

## ✅ PHASE 7 COMPLETE

**What's New in v10.1:**
- ✅ Dual-site stimulation (soma, AIS, dendritic)
- ✅ Dendritic filtering with frequency attenuation
- ✅ Full GUI integration with widget controls
- ✅ S-D curve analysis (binary search method)
- ✅ 2-D excitability mapping (I × duration)
- ✅ Bilingual interface (Russian + English)

**See:** [DUAL_STIMULATION.md](DUAL_STIMULATION.md)

---

## 🛠️ FOR DEVELOPERS

- **Adding a new preset:** [DEVELOPER_GUIDE.md#new-preset](DEVELOPER_GUIDE.md)
- **Modifying the RHS kernel:** [DEVELOPER_GUIDE.md#rhs-kernel](DEVELOPER_GUIDE.md)
- **Adding a new analysis tab:** [DEVELOPER_GUIDE.md#new-analysis](DEVELOPER_GUIDE.md)
- **Testing framework:** See [tests/](../tests/README.md)

---

## 📈 SUPPORTED NEURON MODELS

| Neuron | Reference | Status |
|--------|-----------|--------|
| Squid Giant Axon | Hodgkin & Huxley 1952 | ✅ Baseline |
| L5 Pyramidal | Mainen & Sejnowski 1996 | ✅ Complete |
| FS Interneuron | Wang & Buzsáki 1996 | ✅ Complete |
| Purkinje Cell | De Schutter & Bower 1994 | ✅ Complete |
| Thalamic Relay | McCormick & Huguenard 1992 | ✅ Complete |
| Alpha-Motoneuron | Powers 2001 | ✅ Complete |

---

## 🔗 QUICK LINKS

- **GitHub:** https://github.com/[repo]/NeuroModelPort
- **Issues:** Report bugs in GitHub Issues
- **Discussions:** Community Q&A in Discussions

---

## 📧 SUPPORT

- **Questions?** Check [guides/TROUBLESHOOTING.md](guides/TROUBLESHOOTING.md)
- **Bug reports?** Use GitHub Issues with this template: [.github/ISSUE_TEMPLATE/bug.md]
- **Feature requests?** Open GitHub Discussion

---

**Happy modeling!** 🧠⚡
