# 🌍 Alemayhu — Roadmap

**Mission:**  
Build a system that can see, remember, and reason about the world using self-supervised learning.

---

## ✅ Phase 1: Foundation

- [x] Define project scope and core principles
- [x] Set up codebase with modular architecture (LeCun 2022: perception, world model, cost, memory, actor, configurator)
- [ ] Choose primary data sources (e.g. video, audio, text)
- [ ] Implement basic data ingestion & preprocessing

---

## 🚧 Phase 2: Perception

- [ ] Train vision encoder (e.g. ConvNet, ViT, etc.)
- [ ] Add self-supervised learning objectives (e.g. contrastive, predictive)
- [ ] Evaluate representation quality on downstream tasks

---

## 🧠 Phase 3: Memory

- [ ] Design memory module (e.g. recurrent state, transformer-based)
- [ ] Integrate memory into perception loop
- [ ] Test temporal consistency and recall

---

## 🧩 Phase 4: Reasoning

- [x] Implement forward world model (JEPA: predict next state in representation space)
- [x] Add planning module (Mode-2 latent imagination, beam-search MPC)
- [x] Evaluate in a simulated environment (CodeWorld: agent writes code by planning)
- [ ] Evaluate in richer environments (e.g. BabyAI, Crafter)

---

## 🚀 Phase 5: Applications

- [ ] Visual question answering
- [ ] Embodied agent control (in simulation)
- [ ] Transfer to real-world tasks or datasets

---

## 🧪 Research Directions

- Open-ended learning from raw sensory data
- Continual learning with no supervision
- Hierarchical world models

---

## 📬 Contributions

Issues and pull requests welcome. Let's build a world model together.
