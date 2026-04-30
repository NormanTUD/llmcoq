(* === semantics.v === *)
(* Version 0.1 — Information-Theoretic Semantics of LLMs.        *)
(* Formalizes what it MEANS for a model to "understand" —         *)
(* connecting distributional predictions to semantic content.     *)
(* Builds on base.v, refinement.v, and topology.v.               *)

Require Import Coq.Reals.Reals.
Require Import Coq.Reals.RIneq.
Require Import Coq.Logic.FunctionalExtensionality.
Require Import Coq.Sets.Ensembles.
Require Import Coq.Lists.List.
Import ListNotations.
Require Import Coq.Arith.PeanoNat.
Require Import Coq.micromega.Lra.
Require Import Coq.micromega.Lia.

From base Require Import base.
From base Require Import refinement.
From base Require Import topology.

Open Scope R_scope.

(* ============================================================ *)
(* PART I: WHAT IS AN LLM?                                       *)
(* ============================================================ *)

(* An LLM is a function:                                         *)
(*   f : Context -> Distribution(Vocabulary)                     *)
(*                                                               *)
(* Given a sequence of tokens (the context), it produces a       *)
(* probability distribution over what token comes next.          *)
(*                                                               *)
(* KEY INSIGHT: This is ALL an LLM does. Every apparent          *)
(* "capability" — reasoning, coding, conversation — emerges      *)
(* from this single operation applied autoregressively.          *)
(*                                                               *)
(* From base.v, we have:                                         *)
(*   output_distribution : TransformerModel -> Context -> RawDist *)
(* This IS the LLM. Everything else is structure around it.      *)

(* ============================================================ *)
(* PART II: THE COMPRESSION INTERPRETATION                       *)
(* ============================================================ *)

(* THEOREM (informal): A good language model is a good compressor.*)
(*                                                               *)
(* Cross-entropy H(p, q) measures how many bits you need to      *)
(* encode data from distribution p using a code optimized for q. *)
(*                                                               *)
(* If the model's distribution q closely matches the true        *)
(* distribution p, then H(p, q) ≈ H(p) — you achieve near-      *)
(* optimal compression.                                          *)
(*                                                               *)
(* The KL divergence KL(p || q) = H(p, q) - H(p) measures       *)
(* the EXTRA bits wasted by using q instead of p.                *)
(*                                                               *)
(* From base.v: kl_nonneg proves KL(p || q) >= 0                *)
(* From refinement.v: bayes_optimal_kl_zero shows optimal = 0   *)

(* Compression rate: bits per token used by the model *)
Definition compression_rate (model : TransformerModel) (ctx : Context) : R :=
  cross_entropy (true_distribution ctx) (output_distribution model ctx).

(* Optimal compression rate: entropy of the true distribution *)
Definition optimal_compression_rate (ctx : Context) : R :=
  entropy (true_distribution ctx).

(* Compression overhead: how much worse than optimal *)
Definition compression_overhead (model : TransformerModel) (ctx : Context) : R :=
  kl_divergence (true_distribution ctx) (output_distribution model ctx).

(* The overhead is always non-negative (from base.v) *)
Lemma overhead_nonneg : forall model ctx,
  valid_context ctx ->
  (compression_overhead model ctx >= 0)%R.
Proof.
  intros model ctx Hvalid.
  unfold compression_overhead.
  apply kl_nonneg.
  - apply true_dist_valid. exact Hvalid.
  - apply (softmax_is_distribution vocab_size (compute_logits model ctx)).
    exact vocab_size_pos.
Qed.

(* ============================================================ *)
(* PART III: MUTUAL INFORMATION AND "UNDERSTANDING"              *)
(* ============================================================ *)

(* What does it mean for a model to "understand" a concept?      *)
(*                                                               *)
(* Formal answer: The model's internal representations encode    *)
(* MUTUAL INFORMATION between the context and future tokens.     *)
(*                                                               *)
(* If the model "understands" that the context is about cooking, *)
(* its internal state carries information that makes cooking-     *)
(* related tokens more probable — and this is reflected in the   *)
(* output distribution.                                          *)

(* We formalize "understanding" as: the model's representation   *)
(* at a context carries predictive information about the future. *)

(* The representation at a context is the final-layer embedding *)
Definition representation (model : TransformerModel) (ctx : Context) : Embedding :=
  transformer_forward model ctx.

(* Two contexts are "semantically equivalent" for the model if   *)
(* they produce the same output distribution.                    *)
Definition semantically_equivalent 
  (model : TransformerModel) (ctx1 ctx2 : Context) : Prop :=
  output_distribution model ctx1 = output_distribution model ctx2.

(* Semantic equivalence is an equivalence relation *)
Lemma sem_equiv_refl : forall model ctx,
  semantically_equivalent model ctx ctx.
Proof.
  intros. unfold semantically_equivalent. reflexivity.
Qed.

Lemma sem_equiv_sym : forall model ctx1 ctx2,
  semantically_equivalent model ctx1 ctx2 ->
  semantically_equivalent model ctx2 ctx1.
Proof.
  intros. unfold semantically_equivalent in *. symmetry. exact H.
Qed.

Lemma sem_equiv_trans : forall model ctx1 ctx2 ctx3,
  semantically_equivalent model ctx1 ctx2 ->
  semantically_equivalent model ctx2 ctx3 ->
  semantically_equivalent model ctx1 ctx3.
Proof.
  intros. unfold semantically_equivalent in *.
  rewrite H. exact H0.
Qed.

(* ============================================================ *)
(* PART IV: THE INFORMATION BOTTLENECK                           *)
(* ============================================================ *)

(* The transformer's forward pass is an information bottleneck:  *)
(*                                                               *)
(*   Context (variable length) --> Embedding (fixed dim) --> Dist *)
(*                                                               *)
(* The final embedding must compress ALL relevant information    *)
(* from the context into a fixed-dimensional vector.             *)
(*                                                               *)
(* This is why dim_model matters: it bounds the mutual           *)
(* information the model can preserve.                           *)
(*                                                               *)
(* From base.v: dim_model is the embedding dimension.            *)
(* The model can encode at most dim_model * log2(precision)      *)
(* bits of information about the context.                        *)

(* Information capacity of the representation *)
Parameter representation_capacity : nat -> R.
  (* Maps dim_model to maximum mutual information in bits *)

Axiom capacity_monotone : forall d1 d2,
  (d1 <= d2)%nat -> 
  (representation_capacity d1 <= representation_capacity d2)%R.

Axiom capacity_positive : forall d, 
  (d > 0)%nat -> (representation_capacity d > 0)%R.

(* The model's prediction quality is bounded by its capacity *)
Axiom information_bottleneck_bound : forall model ctx,
  valid_context ctx ->
  (* The mutual information between context and prediction *)
  (* is bounded by the representation capacity *)
  (entropy (output_distribution model ctx) <= 
   representation_capacity dim_model)%R.

(* ============================================================ *)
(* PART V: AUTOREGRESSIVE DECOMPOSITION                          *)
(* ============================================================ *)

(* The fundamental trick of LLMs: decompose a joint distribution *)
(* over sequences into a product of conditionals.                *)
(*                                                               *)
(* P(x1, x2, ..., xn) = P(x1) * P(x2|x1) * P(x3|x1,x2) * ...*)
(*                                                               *)
(* Each factor P(xi | x1,...,x_{i-1}) is one call to the model. *)
(* This is exact by the chain rule of probability — no approx.  *)
(*                                                               *)
(* The model approximates each conditional:                      *)
(*   q(xi | x1,...,x_{i-1}) ≈ p(xi | x1,...,x_{i-1})           *)
(*                                                               *)
(* The total KL divergence decomposes (from topology.v):         *)
(*   KL(p_joint || q_joint) = sum_i KL(p_i || q_i)             *)
(*                                                               *)
(* This means: if each conditional is good, the joint is good.  *)
(* This is the "approximate gluing" from topology.v.             *)

(* A sequence of tokens defines a trajectory through context space *)
Definition trajectory (model : TransformerModel) (ctx : Context) (tokens : list Token) 
  : list Context :=
  let fix build_trajectory ctx toks acc :=
    match toks with
    | [] => rev acc
    | t :: rest => 
        let ctx' := ctx ++ [embed_token t] in
        build_trajectory ctx' rest (ctx' :: acc)
    end
  in build_trajectory ctx tokens [ctx].

(* Each point on the trajectory has an associated prediction *)
Definition trajectory_predictions 
  (model : TransformerModel) (contexts : list Context) : list RawDist :=
  map (output_distribution model) contexts.

(* The quality of generation = product of per-step qualities *)
(* (In log space: sum of per-step log-probabilities)         *)
Definition sequence_log_probability 
  (model : TransformerModel) (ctx : Context) (tokens : list Token) : R :=
  let contexts := trajectory model ctx tokens in
  fold_left (fun acc pair => 
    let ctx_i := fst pair in
    let tok_i := snd pair in
    (acc + ln (vec_entry vocab_size (output_distribution model ctx_i) tok_i))%R
  ) (combine (removelast contexts) tokens) 0%R.

(* ============================================================ *)
(* PART VI: ATTENTION AS INFORMATION ROUTING                     *)
(* ============================================================ *)

(* From attention_geometry.v, attention defines a weighted graph.*)
(* The SEMANTIC interpretation: attention weights determine      *)
(* which past tokens are relevant for predicting the next token. *)
(*                                                               *)
(* This is information routing:                                  *)
(*   - High attention weight i->j means "position j's info is   *)
(*     relevant for computing position i's representation"       *)
(*   - The causal mask ensures information flows only backward   *)
(*   - Multi-head attention allows MULTIPLE routing patterns     *)
(*     simultaneously (from attention_geometry.v)                *)

(* An attention pattern induces an information flow graph *)
Definition information_flow (attn : nat -> nat -> R) (threshold : R) 
  (i j : nat) : Prop :=
  (attn i j > threshold)%R.

(* Transitive closure: information can flow through chains *)
Inductive reachable (attn : nat -> nat -> R) (threshold : R) : nat -> nat -> Prop :=
  | reach_direct : forall i j, 
      information_flow attn threshold i j -> reachable attn threshold i j
  | reach_trans : forall i j k,
      reachable attn threshold i k -> 
      reachable attn threshold k j -> 
      reachable attn threshold i j.

(* Multi-layer attention composes: information can flow across layers *)
(* This connects to the simplicial complex in attention_geometry.v: *)
(* higher-dimensional simplices represent multi-hop information paths *)

(* ============================================================ *)
(* PART VII: IN-CONTEXT LEARNING                                 *)
(* ============================================================ *)

(* One of the most remarkable properties of LLMs:               *)
(* They can learn NEW tasks from examples in the context,        *)
(* without any weight updates.                                   *)
(*                                                               *)
(* Formally: given examples (x1,y1), ..., (xk,yk) in context,  *)
(* the model's prediction for a new x_{k+1} improves.           *)

(* A "few-shot prompt" is a context containing examples *)
Definition few_shot_prompt (examples : list (Context * Token)) (query : Context) : Context :=
  let example_contexts := map (fun pair => fst pair ++ [embed_token (snd pair)]) examples in
  fold_left (fun acc ex => acc ++ ex) example_contexts [] ++ query.

(* In-context learning: more examples improve prediction *)
Definition in_context_learning_improves 
  (model : TransformerModel) 
  (examples : list (Context * Token))
  (query : Context) 
  (target : Token) : Prop :=
  forall prefix suffix,
    examples = prefix ++ suffix ->
    (length suffix > 0)%nat ->
    (* Prediction with all examples is at least as good as with fewer *)
    (vec_entry vocab_size 
      (output_distribution model (few_shot_prompt examples query)) target >=
     vec_entry vocab_size 
      (output_distribution model (few_shot_prompt prefix query)) target)%R.

(* Why does this work? The model has learned a META-algorithm:   *)
(* "Given examples of a pattern, predict according to that       *)
(* pattern." This is implicit Bayesian inference over tasks.     *)

(* ============================================================ *)
(* PART VIII: THE GEOMETRY OF KNOWLEDGE                          *)
(* ============================================================ *)

(* From topology.v: the model defines a presheaf on contexts.    *)
(* From attention_geometry.v: attention has simplicial structure. *)
(*                                                               *)
(* Combining these: the model's "knowledge" has a SHAPE.         *)
(*                                                               *)
(* - Connected components (β₀) = distinct "topics" the model    *)
(*   can discuss                                                 *)
(* - Loops (β₁) = circular reasoning patterns or consistent     *)
(*   belief systems                                              *)
(* - Voids (β₂) = gaps in knowledge                             *)
(*                                                               *)
(* The persistent homology (from attention_geometry.v) tracks    *)
(* which structures are robust vs. artifacts of threshold choice.*)

(* A "knowledge region" is a set of contexts where the model     *)
(* produces similar predictions (using JS divergence from topology.v) *)
Definition knowledge_region 
  (model : TransformerModel) (center : Context) (epsilon : R) : Ensemble Context :=
  fun ctx => 
    (js_divergence (output_distribution model center) 
                   (output_distribution model ctx) < epsilon)%R.

(* Knowledge regions are "open sets" in the model's semantic space *)
(* This connects to the topology defined in topology.v *)

(* A model "knows a fact" if there's a stable knowledge region *)
(* around contexts that express that fact *)
Definition stable_knowledge 
  (model : TransformerModel) (ctx : Context) (epsilon delta : R) : Prop :=
  (epsilon > 0)%R /\ (delta > 0)%R /\
  forall ctx',
    valid_context ctx' ->
    (context_dist ctx ctx' < delta)%R ->
    (js_divergence (output_distribution model ctx) 
                   (output_distribution model ctx') < epsilon)%R.

(* Stable knowledge follows from model continuity (topology.v) *)
Lemma continuity_implies_stable_knowledge :
  forall model ctx epsilon,
    valid_context ctx ->
    model_continuous model ->
    (epsilon > 0)%R ->
    exists delta, stable_knowledge model ctx epsilon delta.
Proof.
  intros model ctx epsilon Hvalid Hcont Heps.
  unfold model_continuous in Hcont.
  specialize (Hcont epsilon Heps).
  destruct Hcont as [delta [Hdelta_pos Hcont]].
  exists delta.
  unfold stable_knowledge.
  split; [exact Heps|].
  split; [exact Hdelta_pos|].
  intros ctx' Hvalid' Hdist.
  (* We need length ctx = length ctx' for the continuity lemma *)
  (* In general this requires same-length contexts *)
  (* For the general case, we use the triangle inequality *)
  (* and the fact that the model is continuous *)
  admit. (* Requires additional lemma about variable-length contexts *)
Admitted.

(* ============================================================ *)
(* PART IX: TRAINING = COMPRESSION = UNDERSTANDING               *)
(* ============================================================ *)

(* The deep connection:                                           *)
(*                                                               *)
(* 1. TRAINING minimizes cross-entropy loss (base.v, Part IV)    *)
(*    min_θ H(p_data, q_θ)                                      *)
(*                                                               *)
(* 2. CROSS-ENTROPY = COMPRESSION (this file, Part II)           *)
(*    H(p, q) = expected code length using q to encode p         *)
(*                                                               *)
(* 3. COMPRESSION requires UNDERSTANDING (informal)              *)
(*    To predict "The capital of France is ___" you must have    *)
(*    encoded the fact that Paris is the capital of France.       *)
(*                                                               *)
(* Therefore: training on next-token prediction forces the model *)
(* to build internal representations that capture the causal     *)
(* structure of the data-generating process.                     *)
(*                                                               *)
(* This is formalized by the Bayes-optimal result:               *)
(*   If KL(p || q_model) = 0, then q_model = p                  *)
(*   (from refinement.v: bayes_optimal_kl_zero)                  *)
(*                                                               *)
(* A perfect predictor has perfectly modeled the data source.    *)

(* The "understanding" emerges from compression pressure *)
Definition understanding_level (model : TransformerModel) (ctx : Context) : R :=
  (* How close to optimal compression = how much the model "understands" *)
  let overhead := compression_overhead model ctx in
  let optimal := optimal_compression_rate ctx in
  (* Normalize: 1 = perfect understanding, 0 = no understanding *)
  if Rle_dec optimal 0 then 1%R
  else (1 - overhead / (overhead + optimal))%R.

(* ============================================================ *)
(* PART X: EMERGENT CAPABILITIES REVISITED                       *)
(* ============================================================ *)

(* From refinement.v: capabilities emerge at scale thresholds.   *)
(* WHY does this happen?                                         *)
(*                                                               *)
(* Hypothesis (formalized): A capability requires a minimum      *)
(* amount of mutual information in the representation.           *)
(* Below a certain model size, the information bottleneck        *)
(* (Part IV) prevents encoding enough information.               *)
(* Above the threshold, the capacity is sufficient.              *)

(* Information requirement for a capability *)
Parameter capability_info_requirement : Capability -> R.

Axiom capability_requires_capacity : forall cap model epsilon,
  model_has_capability model cap epsilon ->
  (representation_capacity dim_model >= capability_info_requirement cap)%R.

(* This explains phase transitions: *)
(* As dim_model grows, representation_capacity grows (monotonically). *)
(* When it crosses capability_info_requirement for a capability, *)
(* that capability suddenly becomes achievable. *)

(* ============================================================ *)
(* PART XI: THE SELF-REFERENCE STRUCTURE                         *)
(* ============================================================ *)

(* From topology.v (Part XXVI): a model can approximately model  *)
(* itself. This has deep implications:                           *)
(*                                                               *)
(* 1. The model can SIMULATE its own behavior                    *)
(*    (self_prediction from topology.v)                          *)
(*                                                               *)
(* 2. This enables META-COGNITION: the model can reason about    *)
(*    what it would say, enabling self-correction                *)
(*                                                               *)
(* 3. The fixed-point theorem (self_consistency_possible from    *)
(*    topology.v) guarantees this is achievable                  *)
(*                                                               *)
(* 4. But PERFECT self-modeling is impossible (by a diagonal     *)
(*    argument analogous to the halting problem)                 *)

(* Imperfect self-knowledge: there exist contexts where the      *)
(* model's self-prediction diverges from its actual behavior     *)
Definition self_knowledge_gap (model : TransformerModel) : R :=
  (* Supremum of JS divergence between self-prediction and actual *)
  (* Over all valid query contexts *)
  0%R. (* Placeholder — would need supremum over contexts *)

(* No model has zero self-knowledge gap (informal diagonal argument) *)
(* This is analogous to Gödel's incompleteness: *)
(* A sufficiently powerful model cannot perfectly predict itself  *)
(* on all inputs, because it could then be used to construct     *)
(* a contradictory input.                                        *)

(* ============================================================ *)
(* PART XII: SUMMARY — WHAT AN LLM IS                            *)
(* ============================================================ *)

(* An LLM is:                                                    *)
(*                                                               *)
(* MATHEMATICALLY:                                               *)
(*   A function Context -> Distribution(Vocabulary)              *)
(*   implemented as a composition of:                            *)
(*     - Token embedding (lookup table)                          *)
(*     - Repeated transformer layers (attention + FFN + residual)*)
(*     - Final projection + softmax                              *)
(*   (Formalized in base.v, Parts I-II)                          *)
(*                                                               *)
(* INFORMATION-THEORETICALLY:                                    *)
(*   An approximate compressor of natural language               *)
(*   that minimizes KL(true_dist || model_dist)                  *)
(*   (Formalized in base.v Part III, this file Part II)          *)
(*                                                               *)
(* TOPOLOGICALLY:                                                *)
(*   A continuous map from context space to distribution space   *)
(*   that defines a presheaf on the prefix category              *)
(*   with sheaf-like gluing properties                           *)
(*   (Formalized in topology.v, Parts XX-XXIV)                   *)
(*                                                               *)
(* GEOMETRICALLY:                                                *)
(*   A system whose internal attention patterns form simplicial  *)
(*   complexes with computable persistent homology               *)
(*   (Formalized in attention_geometry.v)                        *)
(*                                                               *)
(* COMPUTATIONALLY:                                              *)
(*   A universal approximator (refinement.v Part XIII) that      *)
(*   can simulate arbitrary computable functions given           *)
(*   sufficient capacity, including approximating itself         *)
(*   (topology.v Part XXVI)                                      *)
(*                                                               *)
(* The key insight connecting all these views:                   *)
(*   PREDICTION = COMPRESSION = UNDERSTANDING                    *)
(*   Training to predict the next token forces the model to      *)
(*   build a compressed representation of the world.             *)

(* ============================================================ *)
(* PART XIII: PROOF STATUS                                       *)
(* ============================================================ *)

(* PROVED:                                                       *)
(*   - overhead_nonneg                                           *)
(*   - sem_equiv_refl, sem_equiv_sym, sem_equiv_trans            *)
(*   - (inherits all proofs from base, refinement, topology)     *)
(*                                                               *)
(* ADMITTED:                                                      *)
(*   - continuity_implies_stable_knowledge (needs variable-length*)
(*     generalization of model_continuous)                        *)
(*                                                               *)
(* AXIOMS (domain knowledge):                                    *)
(*   - capacity_monotone, capacity_positive                      *)
(*   - information_bottleneck_bound                              *)
(*   - capability_requires_capacity                              *)
(* ============================================================ *)
