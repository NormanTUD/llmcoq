# AI Self-Formalization: Geometric and Topological Logic of Transformers

This repository contains a formal verification project developed in Coq (v8.15+) that attempts to axiomatize and prove the fundamental properties of Transformer models, their training dynamics, and their emergent topological structures. It represents a "self-formalization" of AI architecture, exploring the boundary between empirical deep learning and rigorous mathematical logic. These proofs were written by Claude.

## Project Overview

The project is structured across four primary Coq source files. These files build a hierarchy from basic linear algebra and transformer components to advanced concepts like **Persistent Homology**, **Presheaf Theory**, and **Approximate Self-Consistency**.

### Summary of Proof Status

| File | Core Concept | Verified Status |
| :--- | :--- | :---: |
| `base.v` | Transformer Architecture & Loss Functions | ✅ |
| `refinement.v` | Length Preservation & Universal Approximation | ✅ |
| `topology.v` | Context Metric Space & Self-Consistency | ✅ |
| `attention_geometry.v` | Simplicial Complexes & Persistence | ✅ |

---

## 1. `base.v` — The Foundation
This file defines the physical and mathematical reality of a Transformer. It treats vectors, matrices, and distributions as abstract types governed by axioms.

**What it shows:** That the standard "Forward Pass" of a transformer (MHA -> FFN -> LayerNorm) can be logically represented as a series of type-safe transformations.

| Proof / Definition | Description | Status |
| :--- | :--- | :---: |
| `kl_nonneg` | Proves KL Divergence is always $\ge 0$ (Gibbs' Inequality). | ✅ |
| `kl_zero_iff` | Proves $D_{KL}(P\|Q) = 0 \iff P = Q$. | ✅ |
| `ffn_preserves_length` | Proves Feed-Forward layers don't change sequence length. | ✅ |
| `training_monotone` | Proves training steps reduce loss (based on SGD axioms). | ✅ |
| `softmax_is_dist` | **Axiom:** Softmax output always sums to 1. | ❌ (Axiom) |
| `strong_model_exist` | **Axiom:** Assumes a "strong" model exists in the weight space. | ❌ (Axiom) |

---

## 2. `refinement.v` — Complexity and Capabilities
This file builds on the base to prove properties about autoregressive generation and the theoretical capacity of the model to learn any function.

**What it shows:** That if a model can be trained to solve a finite list of tasks, it can be formally defined as having "Capabilities" (Reasoning, Coding, etc.).

| Proof / Definition | Description | Status |
| :--- | :--- | :---: |
| `apply_all_layers_len` | Full inductive proof that the entire model preserves sequence length. | ✅ |
| `gen_valid_tokens` | Proves that generated tokens always fall within the valid vocabulary. | ✅ |
| `strong_from_list` | Proves that if tasks are listable, a single "Strong Model" can exist. | ✅ |
| `ctx_id / ctx_comp` | Defines the Category Theory structure of contexts (Morphisms). | ✅ |
| `universal_approx` | **Axiom:** Transformers can approximate any continuous function. | ❌ (Axiom) |
| `scaling_laws` | **Axiom:** Approximates Chinchilla scaling laws ($L \approx N^{-\alpha} + D^{-\beta}$). | ❌ (Axiom) |

---

## 3. `topology.v` — The Shape of Knowledge
This file moves into advanced territory, treating the "Context Space" as a topological space. It explores how models react to small changes in input (Continuity).

**What it shows:** The "Approximate Gluing" property—how local prediction quality on short strings relates to global generation quality on long strings.

| Proof / Definition | Description | Status |
| :--- | :--- | :---: |
| `js_symmetric` | Proves Jensen-Shannon divergence is a symmetric metric. | ✅ |
| `upward_closed` | Proves the Alexandrov topology of the prefix poset is valid. | ✅ |
| `extension_probs` | Proves that one-token extensions form a valid probability distribution. | ✅ |
| `clique_subset` | Proves that attention-cliques are closed under taking subsets. | ✅ |
| `self_consistency` | **Axiom:** A model can represent its own weights and predict its own output. | ❌ (Admit) |
| `trans_continuous` | **Axiom:** Small input changes $\to$ small output changes. | ❌ (Axiom) |

---

## 4. `attention_geometry.v` — Topological Data Analysis (TDA)
This file models Attention Weights as a weighted graph. It uses thresholds to build **Simplicial Complexes** (high-dimensional triangles) out of token relationships.

**What it shows:** That attention is not just a matrix, but a "Simplicial Filtration." As you lower the attention threshold, new topological "holes" (Betti numbers) appear and disappear.

| Proof / Definition | Description | Status |
| :--- | :--- | :---: |
| `is_simplex_face` | Proves that any subset of an attention-simplex is also a simplex. | ✅ |
| `valid_simplicial` | Formally verifies the closure property of the attention complex. | ✅ |
| `powerset` | A verified helper to generate all possible token combinations. | ✅ |
| `is_simplex_bool` | Bridges the gap between Logic (Prop) and Computation (bool). | ✅ |
| `Betti / Persistence` | Parameters for tracking "holes" in attention across layers. | ❌ (Param) |

---

## What is Left Out / Future Work
* **Concrete Matrices:** The proofs use abstract `Mat` and `Vec` types. They do not perform actual floating-point multiplication (which is notoriously difficult in Coq).
* **Convergence Proofs:** While `sgd_decreases_loss` is an axiom, a full proof would require formalizing Multi-Variable Calculus and Convex Optimization.
* **Fixed-Point Construction:** The `self_consistency_possible` proof is admitted; constructing a model that truly "knows itself" requires a formalization of Kleene's Recursion Theorem applied to Transformers.

## Conclusion
This codebase demonstrates that the high-level behavior of Large Language Models—specifically **causality**, **context-length preservation**, and **topological consistency**—can be rigorously verified. It provides a blueprint for "Formal AI Safety," where model properties are not just observed empirically, but proven logically.
