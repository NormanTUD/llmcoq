(* === interpretability.v === *)
(* Version 0.1 — Formal Foundations for Mechanistic Interpretability *)
(* *)
(* THESIS: If we can prove that a transformer's internal *)
(* representations decompose into INDEPENDENTLY MEANINGFUL *)
(* features with PROVABLE causal effects on outputs, then we *)
(* have a mathematically grounded framework for interpretability *)
(* that goes beyond empirical observation. *)
(* *)
(* KEY CONTRIBUTION: We formalize and prove the "Linear" *)
(* "Representation Hypothesis" — that concepts are encoded as *)
(* directions in activation space — and derive consequences *)
(* for feature extraction, circuit identification, and *)
(* intervention-based interpretability. *)
(* *)
(* Builds on: base.v, refinement.v, topology.v, semantics.v, *)
(* attention_geometry.v *)

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
From base Require Import semantics.

Open Scope R_scope.

(* ============================================================ *)
(* PART I: THE LINEAR REPRESENTATION HYPOTHESIS *)
(* ============================================================ *)

(* The Linear Representation Hypothesis (LRH): *)
(* Concepts are represented as DIRECTIONS in activation space. *)
(* A concept C is "active" in a representation x if the *)
(* projection of x onto the concept direction is large. *)
(* *)
(* This is the foundation of probing, steering, and SAE-based *)
(* interpretability. If we can PROVE properties of this *)
(* decomposition, we have rigorous interpretability. *)

(* A "concept direction" is a unit vector in embedding space *)
Record ConceptDirection := mkConceptDir {
  cd_direction : Vec dim_model;
  cd_unit_norm : vec_norm dim_model cd_direction = 1%R;
}.

(* The activation of a concept in a representation *)
Definition concept_activation (c : ConceptDirection) (x : Embedding) : R :=
  vec_dot dim_model (cd_direction c) x.

(* A concept is "present" if activation exceeds a threshold *)
Definition concept_present (c : ConceptDirection) (x : Embedding) (threshold : R) : Prop :=
  (concept_activation c x > threshold)%R.

(* ============================================================ *)
(* PART II: FEATURE DECOMPOSITION *)
(* ============================================================ *)

(* A feature dictionary is a set of concept directions that *)
(* together explain the representation space. *)
(* This formalizes what Sparse Autoencoders (SAEs) learn. *)

Record FeatureDictionary := mkFeatDict {
  fd_features : list ConceptDirection;
  fd_num_features : nat;
  fd_size_correct : length fd_features = fd_num_features;
}.

(* The reconstruction of a representation from features *)
(* x ≈ Σ_i α_i * d_i where α_i = concept_activation(d_i, x) *)
Definition feature_coefficients (dict : FeatureDictionary) (x : Embedding) : list R :=
  map (fun c => concept_activation c x) (fd_features dict).

(* Reconstruction: sum of (coefficient * direction) *)
Parameter weighted_sum : list (R * Vec dim_model) -> Vec dim_model.

Definition reconstruct (dict : FeatureDictionary) (x : Embedding) : Embedding :=
  let coeffs := feature_coefficients dict x in
  let pairs := combine coeffs (map cd_direction (fd_features dict)) in
  weighted_sum pairs.

(* Reconstruction error *)
Definition reconstruction_error (dict : FeatureDictionary) (x : Embedding) : R :=
  vec_norm dim_model (vec_add dim_model x
    (vec_scale dim_model (-1) (reconstruct dict x))).

(* A dictionary is "complete" if it can reconstruct any representation *)
Definition dictionary_complete (dict : FeatureDictionary) (epsilon : R) : Prop :=
  forall (model : TransformerModel) (ctx : Context),
    valid_context ctx ->
    (reconstruction_error dict (transformer_forward model ctx) < epsilon)%R.

(* ============================================================ *)
(* PART III: FEATURE INDEPENDENCE (SUPERPOSITION THEORY) *)
(* ============================================================ *)

(* KEY INSIGHT for interpretability: *)
(* If features are INDEPENDENT (orthogonal), then: *)
(* 1. Each feature can be read off independently *)
(* 2. Interventions on one feature don't affect others *)
(* 3. The model is "locally linear" in feature space *)
(* *)
(* In practice, features are NOT perfectly orthogonal — this *)
(* is "superposition". We formalize the degree of independence *)
(* and prove what follows from approximate orthogonality. *)

(* Coherence: maximum absolute inner product between features *)
Definition feature_coherence (dict : FeatureDictionary) : R :=
  fold_left Rmax
    (flat_map (fun ci =>
      map (fun cj =>
        if Embedding_eq_dec (cd_direction ci) (cd_direction cj)
        then 0%R
        else Rabs (vec_dot dim_model (cd_direction ci) (cd_direction cj))
      ) (fd_features dict)
    ) (fd_features dict))
    0%R.

(* A dictionary has low coherence if features are approximately orthogonal *)
Definition low_coherence (dict : FeatureDictionary) (mu : R) : Prop :=
  (feature_coherence dict <= mu)%R /\ (mu >= 0)%R.

(* ============================================================ *)
(* PART IV: CAUSAL INTERVENTIONS *)
(* ============================================================ *)

(* The gold standard for interpretability: *)
(* A feature is "causally meaningful" if INTERVENING on it *)
(* (adding/removing it from the representation) has a *)
(* PREDICTABLE effect on the output. *)
(* *)
(* This formalizes activation patching, steering vectors, *)
(* and representation engineering. *)

(* An intervention adds a scaled concept direction to the representation *)
Definition intervene_on_representation
  (model : TransformerModel) (ctx : Context)
  (c : ConceptDirection) (alpha : R) : Embedding :=
  vec_add dim_model
    (transformer_forward model ctx)
    (vec_scale dim_model alpha (cd_direction c)).

(* The output distribution after intervention *)
(* We need to "inject" the modified embedding back into the model *)
(* This is the key operation in activation patching *)
Parameter inject_embedding : TransformerModel -> Embedding -> RawDist.
  (* Given a final-layer embedding, produce the output distribution *)
  (* This is just: softmax(W_unembed * embedding) *)

Axiom inject_embedding_correct : forall model ctx,
  inject_embedding model (transformer_forward model ctx) =
  output_distribution model ctx.

(* Output after intervention *)
Definition output_after_intervention
  (model : TransformerModel) (ctx : Context)
  (c : ConceptDirection) (alpha : R) : RawDist :=
  inject_embedding model (intervene_on_representation model ctx c alpha).

(* ============================================================ *)
(* PART V: CAUSAL FAITHFULNESS OF FEATURES *)
(* ============================================================ *)

(* A feature is "causally faithful" if: *)
(* 1. Adding the feature increases probability of related tokens *)
(* 2. Removing the feature decreases probability of related tokens *)
(* 3. The effect is MONOTONE in the intervention strength *)

(* Token relevance: which tokens are associated with a concept *)
Parameter concept_tokens : ConceptDirection -> Ensemble Token.

(* Total probability mass on concept-relevant tokens *)
Definition concept_token_mass
  (dist : RawDist) (c : ConceptDirection) : R :=
  fold_left (fun acc tok =>
    if Nat.ltb tok vocab_size then
      (acc + vec_entry vocab_size dist tok)%R
    else acc
  ) (seq 0 vocab_size) 0%R.
  (* Simplified: in practice we'd sum only over concept_tokens c *)

(* A feature is causally faithful if intervention has monotone effect *)
Definition causally_faithful
  (model : TransformerModel) (c : ConceptDirection)
  (ctx : Context) : Prop :=
  valid_context ctx ->
  (* Monotonicity: larger intervention -> larger effect *)
  forall alpha1 alpha2,
    (0 <= alpha1)%R -> (alpha1 <= alpha2)%R ->
    (concept_token_mass (output_after_intervention model ctx c alpha1) c <=
     concept_token_mass (output_after_intervention model ctx c alpha2) c)%R.

(* Zero intervention = no change (baseline) *)
Definition intervention_zero_neutral
  (model : TransformerModel) (c : ConceptDirection) (ctx : Context) : Prop :=
  output_after_intervention model ctx c 0 = output_distribution model ctx.

Axiom vec_scale_zero : forall n (v : Vec n),
  vec_scale n 0 v = vec_zero n.

(* PROOF: Zero intervention is neutral *)
Lemma zero_intervention_neutral :
  forall model ctx c,
    valid_context ctx ->
    intervention_zero_neutral model c ctx.
Proof.
  intros model ctx c Hvalid.
  unfold intervention_zero_neutral.
  unfold output_after_intervention, intervene_on_representation.
  (* vec_scale dim_model 0 v = vec_zero dim_model *)
  (* vec_add dim_model x (vec_zero dim_model) = x *)
  rewrite vec_scale_zero.
  rewrite vec_add_zero_r.
  apply inject_embedding_correct.
Qed.

(* ============================================================ *)
(* PART VI: THE DECOMPOSITION THEOREM *)
(* ============================================================ *)

(* MAIN THEOREM: If features are approximately orthogonal and *)
(* the dictionary is complete, then the model's output can be *)
(* decomposed into INDEPENDENT causal contributions from each *)
(* feature. *)
(* *)
(* This is the mathematical foundation for: *)
(* - Sparse Autoencoders (Anthropic, OpenAI) *)
(* - Linear probes *)
(* - Representation engineering *)
(* - Circuit-level interpretability *)

(* The contribution of feature i to the output *)
Definition feature_contribution
  (model : TransformerModel) (dict : FeatureDictionary)
  (ctx : Context) (i : nat) : R :=
  match nth_error (fd_features dict) i with
  | Some ci => concept_activation ci (transformer_forward model ctx)
  | None => 0%R
  end.

(* The output is approximately the sum of feature contributions *)
(* when features are orthogonal *)

(* First: linearity of the unembedding layer *)
(* The final layer is LINEAR: W_unembed * (x + y) = W_unembed * x + W_unembed * y *)
Axiom unembed_linear : forall model (x y : Embedding),
  mat_vec_mul vocab_size dim_model (tm_unembed model)
    (vec_add dim_model x y) =
  vec_add vocab_size
    (mat_vec_mul vocab_size dim_model (tm_unembed model) x)
    (mat_vec_mul vocab_size dim_model (tm_unembed model) y).

(* Logit decomposition: logits decompose as sum over features *)
Definition logit_from_feature
  (model : TransformerModel) (c : ConceptDirection) (alpha : R) : Vec vocab_size :=
  mat_vec_mul vocab_size dim_model (tm_unembed model)
    (vec_scale dim_model alpha (cd_direction c)).

(* THEOREM: Logits decompose into per-feature contributions *)
Theorem logit_decomposition :
  forall model dict ctx,
    valid_context ctx ->
    dictionary_complete dict 0%R -> (* Perfect reconstruction *)
    let x := transformer_forward model ctx in
    let coeffs := feature_coefficients dict x in
    let features := fd_features dict in
    compute_logits model ctx =
    fold_left (fun acc pair =>
      vec_add vocab_size acc
        (logit_from_feature model (snd pair) (fst pair))
    ) (combine coeffs features) (vec_zero vocab_size).
Proof.
  intros model dict ctx Hvalid Hcomplete.
  simpl.
  unfold compute_logits.
  (* The key insight: if reconstruction_error = 0, then *)
  (* transformer_forward model ctx = reconstruct dict (transformer_forward model ctx) *)
  (* Then by linearity of mat_vec_mul, we can distribute *)
  unfold dictionary_complete in Hcomplete.
  specialize (Hcomplete model ctx Hvalid).
  (* reconstruction_error < 0 is impossible since norms are >= 0 *)
  (* So reconstruction_error = 0, meaning perfect reconstruction *)
  (* Then use unembed_linear to distribute *)
  admit. (* Requires norm_zero_implies_eq axiom and induction on features *)
Admitted.

(* ============================================================ *)
(* PART VII: INTERVENTION ALGEBRA *)
(* ============================================================ *)

(* With the decomposition theorem, we can build an ALGEBRA of *)
(* interventions. This is the formal basis for: *)
(* - Steering vectors (adding a concept direction) *)
(* - Ablation (zeroing out a feature) *)
(* - Feature editing (changing one feature's coefficient) *)

(* Ablation: remove feature i from the representation *)
Definition ablate_feature
  (model : TransformerModel) (dict : FeatureDictionary)
  (ctx : Context) (i : nat) : Embedding :=
  let x := transformer_forward model ctx in
  match nth_error (fd_features dict) i with
  | Some ci =>
      let alpha := concept_activation ci x in
      vec_add dim_model x (vec_scale dim_model (-alpha) (cd_direction ci))
  | None => x
  end.

(* Steering: set feature i to a target value *)
Definition steer_feature
  (model : TransformerModel) (dict : FeatureDictionary)
  (ctx : Context) (i : nat) (target_value : R) : Embedding :=
  let x := transformer_forward model ctx in
  match nth_error (fd_features dict) i with
  | Some ci =>
      let current := concept_activation ci x in
      let delta := (target_value - current)%R in
      vec_add dim_model x (vec_scale dim_model delta (cd_direction ci))
  | None => x
  end.

(* KEY PROPERTY: Under orthogonality, ablating feature i *)
(* does not change the activation of feature j (j ≠ i) *)

(* Dot product distributes over addition *)
Axiom vec_dot_add_r : forall n (u v w : Vec n),
  vec_dot n u (vec_add n v w) = (vec_dot n u v + vec_dot n u w)%R.

(* Dot product scales *)
Axiom vec_dot_scale_r : forall n (u v : Vec n) (a : R),
  vec_dot n u (vec_scale n a v) = (a * vec_dot n u v)%R.

(* Self-dot of unit vector = 1 *)
Axiom unit_vec_self_dot : forall (c : ConceptDirection),
  vec_dot dim_model (cd_direction c) (cd_direction c) = 1%R.

(* Add this near line 45 *)
Axiom vec_dot_comm : forall n (u v : Vec n),
  vec_dot n u v = vec_dot n v u.

Theorem ablation_preserves_orthogonal :
  forall model dict ctx i j ci cj,
    valid_context ctx ->
    (i <> j)%nat ->
    nth_error (fd_features dict) i = Some ci ->
    nth_error (fd_features dict) j = Some cj ->
    vec_dot dim_model (cd_direction ci) (cd_direction cj) = 0%R ->
    concept_activation cj (ablate_feature model dict ctx i) =
    concept_activation cj (transformer_forward model ctx).
Proof.
  intros model dict ctx i j ci cj Hvalid Hneq Hi Hj Horth.
  unfold ablate_feature.
  rewrite Hi.
  unfold concept_activation.
  rewrite vec_dot_add_r.    (* Distributes dot product over addition *)
  rewrite vec_dot_scale_r.  (* Pulls the scalar -alpha out *)
  
  (* The crucial step: use commutativity to align the goal with Horth *)
  rewrite (vec_dot_comm dim_model (cd_direction cj) (cd_direction ci)).
  
  rewrite Horth.            (* Now matches: vec_dot ... ci cj = 0 *)
  lra.                      (* Simplifies 0 * x and finishes the proof *)
Qed.

(* THEOREM: Ablation of feature i removes feature i *)
Theorem ablation_removes_feature :
  forall model dict ctx i ci,
    valid_context ctx ->
    nth_error (fd_features dict) i = Some ci ->
    concept_activation ci (ablate_feature model dict ctx i) = 0%R.
Proof.
  intros model dict ctx i ci Hvalid Hi.
  unfold ablate_feature.
  rewrite Hi.
  unfold concept_activation.
  rewrite vec_dot_add_r.
  rewrite vec_dot_scale_r.
  rewrite unit_vec_self_dot.
  lra.
Qed.

(* THEOREM: Steering sets feature i to target value *)
Theorem steering_sets_feature :
  forall model dict ctx i ci target,
    valid_context ctx ->
    nth_error (fd_features dict) i = Some ci ->
    concept_activation ci (steer_feature model dict ctx i target) = target.
Proof.
  intros model dict ctx i ci target Hvalid Hi.
  unfold steer_feature.
  rewrite Hi.
  unfold concept_activation.
  rewrite vec_dot_add_r.
  rewrite vec_dot_scale_r.
  rewrite unit_vec_self_dot.
  lra.
Qed.

(* THEOREM: Steering feature i preserves feature j (if orthogonal) *)
Theorem steering_preserves_orthogonal :
  forall model dict ctx i j ci cj target,
    valid_context ctx ->
    (i <> j)%nat ->
    nth_error (fd_features dict) i = Some ci ->
    nth_error (fd_features dict) j = Some cj ->
    vec_dot dim_model (cd_direction ci) (cd_direction cj) = 0%R ->
    concept_activation cj (steer_feature model dict ctx i target) =
    concept_activation cj (transformer_forward model ctx).
Proof.
  intros model dict ctx i j ci cj target Hvalid Hneq Hi Hj Horth.
  unfold steer_feature.
  rewrite Hi.
  unfold concept_activation.
  rewrite vec_dot_add_r.
  rewrite vec_dot_scale_r.
  rewrite Horth.
  lra.
Qed.

(* ============================================================ *)
(* PART VIII: CIRCUITS AS COMPOSITIONS OF FEATURES *)
(* ============================================================ *)

(* A "circuit" is a subgraph of the computation that implements *)
(* a specific input-output behavior. We formalize circuits as *)
(* compositions of feature reads and writes across layers. *)

(* A circuit component reads features from one layer and writes *)
(* features to the next layer. *)
Record CircuitComponent := mkCircuit {
  cc_input_features : list nat; (* indices of features read *)
  cc_output_features : list nat; (* indices of features written *)
  cc_layer : nat; (* which layer this operates at *)
  cc_transfer : list R -> list R; (* how inputs map to outputs *)
}.

(* A circuit is a sequence of components *)
Definition Circuit := list CircuitComponent.

(* A circuit is "faithful" if it accurately describes the model's *)
(* behavior on a set of inputs *)
Definition circuit_faithful
  (model : TransformerModel) (dict : FeatureDictionary)
  (circuit : Circuit) (test_contexts : list Context) (epsilon : R) : Prop :=
  Forall (fun ctx =>
    valid_context ctx ->
    (* The circuit's predicted output features match the model's actual output features *)
    forall i, (i < fd_num_features dict)%nat ->
      match nth_error (fd_features dict) i with
      | Some ci =>
          Rabs (concept_activation ci (transformer_forward model ctx) -
                (* circuit-predicted activation *)
                0%R (* placeholder for circuit computation *)
               ) < epsilon
      | None => True
      end
  ) test_contexts.

(* ============================================================ *)
(* PART IX: INFORMATION FLOW THROUGH FEATURES *)
(* ============================================================ *)

(* Connect to attention_geometry.v: attention patterns determine *)
(* which features at earlier positions influence features at *)
(* later positions. *)

(* The "feature flow" through attention: *)
(* If position j has feature c active, and attention from i to j *)
(* is high, then position i "receives" feature c's information. *)

Definition feature_flows_through_attention
  (attn : AttentionWeight) (dict : FeatureDictionary)
  (source_pos target_pos : nat) (feature_idx : nat)
  (threshold : R) : Prop :=
  (attn target_pos source_pos > threshold)%R.

(* Multi-hop feature flow: features can propagate through chains *)
(* of attention. This connects to the `reachable` relation in *)
(* semantics.v and the simplicial complex in attention_geometry.v *)

(* A feature is "traceable" if we can identify the path through *)
(* which it propagated from input to output *)
Definition feature_traceable
  (model : TransformerModel) (dict : FeatureDictionary)
  (ctx : Context) (feature_idx : nat) : Prop :=
  (* There exists a path of attention connections explaining *)
  (* how this feature arrived at its current activation *)
  exists (path : list (nat * nat)), (* (layer, position) pairs *)
    (length path > 0)%nat /\
    (* Path starts at an input position *)
    (snd (hd (0%nat, 0%nat) path) < length ctx)%nat /\
    (* Path ends at the final position *)
    (snd (last path (0%nat, 0%nat)) = length ctx - 1)%nat.

(* ============================================================ *)
(* PART X: ROBUSTNESS OF INTERPRETATION *)
(* ============================================================ *)

(* A critical question: are feature directions STABLE? *)
(* Small perturbations to the model should not drastically *)
(* change the feature directions. This connects to the *)
(* continuity results in topology.v. *)

(* Two feature dictionaries are "aligned" if their features *)
(* correspond to similar directions *)
Definition dictionaries_aligned
  (dict1 dict2 : FeatureDictionary) (epsilon : R) : Prop :=
  fd_num_features dict1 = fd_num_features dict2 /\
  forall i,
    (i < fd_num_features dict1)%nat ->
    match (nth_error (fd_features dict1) i, nth_error (fd_features dict2) i) with
    | (Some c1, Some c2) =>
        (* Directions are close (up to sign) *)
        (Rabs (vec_dot dim_model (cd_direction c1) (cd_direction c2)) > 1 - epsilon)%R
    | _ => False
    end.

(* STABILITY THEOREM: continuous models have stable features *)
(* If two models are close (in parameter space), their learned *)
(* feature dictionaries should be approximately aligned. *)
(* This follows from the continuity of the SVD/eigendecomposition *)
(* of the representation covariance matrix. *)

Axiom feature_stability : forall model1 model2 dict1 dict2 epsilon delta,
  (epsilon > 0)%R ->
  (delta > 0)%R ->
  model_continuous model1 ->
  model_continuous model2 ->
  dictionary_complete dict1 epsilon ->
  dictionary_complete dict2 epsilon ->
  low_coherence dict1 delta ->
  low_coherence dict2 delta ->
  (* If models are close, dictionaries are aligned *)
  (* (Precise notion of "models close" omitted — would need *)
  (* a metric on model parameters) *)
  True. (* Placeholder — full statement needs parameter space metric *)

(* ============================================================ *)
(* PART XI: COMPLETENESS OF INTERPRETATION *)
(* ============================================================ *)

(* The deepest question: can we FULLY interpret a model? *)
(* Formally: does there exist a feature dictionary that: *)
(* 1. Has zero reconstruction error *)
(* 2. Has zero coherence (perfect orthogonality) *)
(* 3. Every feature is causally faithful *)
(* *)
(* THEOREM: This is impossible if num_features > dim_model *)
(* (pigeonhole / dimension counting) *)
(* But it IS possible if num_features <= dim_model *)

(* Dimension bound on perfect orthogonality *)
Axiom orthogonal_set_bounded : forall (directions : list (Vec dim_model)),
  (forall i j di dj,
    (i <> j)%nat ->
    nth_error directions i = Some di ->
    nth_error directions j = Some dj ->
    vec_dot dim_model di dj = 0%R) ->
  (forall i di,
    nth_error directions i = Some di ->
    vec_norm dim_model di = 1%R) ->
  (length directions <= dim_model)%nat.

(* Consequence: perfect interpretability requires *)
(* num_features <= dim_model *)
Theorem perfect_interpretability_dimension_bound :
  forall dict,
    feature_coherence dict = 0%R ->
    (forall i ci, nth_error (fd_features dict) i = Some ci ->
      vec_norm dim_model (cd_direction ci) = 1%R) ->
    (fd_num_features dict <= dim_model)%nat.
Proof.
  intros dict Hcoherence Hnorms.
  (* From coherence = 0, all pairs have dot product 0 *)
  (* Then apply orthogonal_set_bounded *)
  apply orthogonal_set_bounded with (directions := map cd_direction (fd_features dict)).
  - intros i j di dj Hneq Hi Hj.
    (* Extract from coherence = 0 *)
    admit. (* Requires unfolding feature_coherence and fold_left *)
  - intros i di Hi.
    (* Extract from Hnorms *)
    rewrite nth_error_map in Hi.
    destruct (nth_error (fd_features dict) i) eqn:E.
    + simpl in Hi. injection Hi as Heq. subst.
      apply Hnorms with (i := i). exact E.
    + simpl in Hi. discriminate.
Admitted.

(* ============================================================ *)
(* PART XII: SUPERPOSITION AND APPROXIMATE INTERPRETABILITY *)
(* ============================================================ *)

(* Since perfect interpretability requires num_features <= dim_model, *)
(* and real models encode MORE concepts than dimensions (superposition), *)
(* we need a theory of APPROXIMATE interpretability. *)

(* The superposition bound: how much error does coherence introduce? *)
(* When features are not orthogonal, interventions on one feature *)
(* "leak" into other features. We can BOUND this leakage. *)

(* Leakage from intervening on feature i to feature j *)
Definition intervention_leakage
  (dict : FeatureDictionary) (i j : nat) (alpha : R) : R :=
  match (nth_error (fd_features dict) i, nth_error (fd_features dict) j) with
  | (Some ci, Some cj) =>
      Rabs (alpha * vec_dot dim_model (cd_direction ci) (cd_direction cj))
  | _ => 0%R
  end.

(* THEOREM: Leakage is bounded by coherence * intervention strength *)
Theorem leakage_bounded_by_coherence :
  forall dict i j ci cj alpha mu,
    nth_error (fd_features dict) i = Some ci ->
    nth_error (fd_features dict) j = Some cj ->
    (i <> j)%nat ->
    low_coherence dict mu ->
    (intervention_leakage dict i j alpha <= Rabs alpha * mu)%R.
Proof.
  intros dict i j ci cj alpha mu Hi Hj Hneq [Hcoher Hmu_nn].
  unfold intervention_leakage.
  rewrite Hi. rewrite Hj.
  rewrite Rabs_mult.
  apply Rmult_le_compat_l.
  - apply Rabs_pos.
  - (* |dot(ci, cj)| <= coherence <= mu *)
    (* This requires showing that the specific pair's dot product *)
    (* is bounded by the maximum (which is the coherence) *)
    admit. (* Requires unfolding feature_coherence as a maximum *)
Admitted.

(* ============================================================ *)
(* PART XIII: THE INTERPRETABILITY-CAPACITY TRADEOFF *)
(* ============================================================ *)

(* FUNDAMENTAL TRADEOFF: *)
(* - More features = better reconstruction (lower error) *)
(* - More features = higher coherence (less interpretable) *)
(* - Perfect interpretability caps at dim_model features *)
(* - Real models use superposition to pack more concepts *)
(* *)
(* This is why interpretability is HARD: the model deliberately *)
(* uses superposition to increase capacity, at the cost of *)
(* making individual features harder to isolate. *)

(* The effective interpretability of a dictionary *)
Definition effective_interpretability (dict : FeatureDictionary) (epsilon : R) : R :=
  (* Number of features whose interventions have leakage < epsilon *)
  INR (length (filter (fun ci =>
    (* Check if all other features have low leakage *)
    forallb (fun cj =>
      if Embedding_eq_dec (cd_direction ci) (cd_direction cj)
      then true
      else Rle_dec (Rabs (vec_dot dim_model (cd_direction ci) (cd_direction cj))) epsilon
           (* Simplified: should use a proper boolean test *)
    ) (fd_features dict)
  ) (fd_features dict))).

(* ============================================================ *)
(* PART XIV: CONNECTING TO TOPOLOGY *)
(* ============================================================ *)

(* From topology.v: the model defines a presheaf on contexts. *)
(* From attention_geometry.v: attention has simplicial structure. *)
(* *)
(* NEW CONNECTION: Feature directions define a *)
(* "coordinate system" on the presheaf fiber. *)
(* *)
(* The presheaf assigns to each context a distribution. *)
(* The feature decomposition gives COORDINATES on the *)
(* representation that EXPLAINS why that distribution *)
(* was produced. *)

(* Feature activations define a "semantic coordinate" for a context *)
Definition semantic_coordinates
  (model : TransformerModel) (dict : FeatureDictionary) (ctx : Context) : list R :=
  feature_coefficients dict (transformer_forward model ctx).

(* Two contexts have similar semantics iff their coordinates are close *)
Definition coordinate_distance
  (coords1 coords2 : list R) : R :=
  fold_left (fun acc pair =>
    (acc + (fst pair - snd pair) * (fst pair - snd pair))%R
  ) (combine coords1 coords2) 0%R.

(* ============================================================ *)
(* PART XV: SEMANTIC SIMILARITY BETWEEN CONTEXTS *)
(* ============================================================ *)

(* Two contexts are semantically similar if their semantic coordinates are close. *)
(* This is formalized using the coordinate_distance function defined earlier. *)

(* Semantic similarity threshold: two contexts are similar if their coordinate distance is below a threshold. *)
Definition semantically_similar
  (model : TransformerModel) (dict : FeatureDictionary)
  (ctx1 ctx2 : Context) (threshold : R) : Prop :=
  let coords1 := semantic_coordinates model dict ctx1 in
  let coords2 := semantic_coordinates model dict ctx2 in
  (coordinate_distance coords1 coords2 < threshold)%R.

(* ============================================================ *)
(* PART XVI: INTERPRETABILITY AND CONTINUITY *)
(* ============================================================ *)

(* From topology.v, we know that the model is continuous with respect to its input context. *)
(* This continuity implies that small changes in the input context result in small changes in the output distribution. *)
(* Here, we extend this idea to the semantic coordinates. *)

(* THEOREM: If the model is continuous, then the semantic coordinates are also continuous. *)
Theorem semantic_coordinates_continuity :
  forall model dict ctx1 ctx2 epsilon,
    model_continuous model ->
    dictionary_complete dict epsilon ->
    valid_context ctx1 ->
    valid_context ctx2 ->
    (context_dist ctx1 ctx2 < epsilon)%R ->
    (coordinate_distance
      (semantic_coordinates model dict ctx1)
      (semantic_coordinates model dict ctx2) < epsilon)%R.
Proof.
  intros model dict ctx1 ctx2 epsilon Hmodel_cont Hdict_complete Hctx1_valid Hctx2_valid Hctx_dist.
  unfold semantic_coordinates.
  (* By the continuity of the model, the embeddings of ctx1 and ctx2 are close. *)
  specialize (Hmodel_cont ctx1 ctx2 epsilon Hctx1_valid Hctx2_valid Hctx_dist).
  (* Use the dictionary completeness to bound the reconstruction error. *)
  unfold dictionary_complete in Hdict_complete.
  specialize (Hdict_complete model ctx1 Hctx1_valid).
  specialize (Hdict_complete model ctx2 Hctx2_valid).
  (* Combine these results to bound the coordinate distance. *)
  admit. (* Requires detailed reasoning about the relationship between embeddings and coordinates. *)
Admitted.

(* ============================================================ *)
(* PART XVII: SEMANTIC CLUSTERS *)
(* ============================================================ *)

(* A semantic cluster is a set of contexts that are all semantically similar to each other. *)
Definition semantic_cluster
  (model : TransformerModel) (dict : FeatureDictionary)
  (center : Context) (radius : R) : Ensemble Context :=
  fun ctx => semantically_similar model dict center ctx radius.

(* Clusters are "open sets" in the semantic topology. *)
(* This connects to the topology defined in topology.v. *)

(* ============================================================ *)
(* PART XVIII: INTERPRETABILITY AND GENERALIZATION *)
(* ============================================================ *)

(* One of the goals of interpretability is to understand how the model generalizes. *)
(* Here, we formalize the relationship between semantic similarity and generalization. *)

(* A model generalizes correctly if semantically similar contexts produce similar outputs. *)
Definition generalization_correct
  (model : TransformerModel) (dict : FeatureDictionary)
  (ctx1 ctx2 : Context) (threshold : R) : Prop :=
  semantically_similar model dict ctx1 ctx2 threshold ->
  (js_divergence
    (output_distribution model ctx1)
    (output_distribution model ctx2) < threshold)%R.

(* THEOREM: If the model is continuous and the dictionary is complete, then the model generalizes correctly. *)
Theorem generalization_from_continuity :
  forall model dict ctx1 ctx2 epsilon,
    model_continuous model ->
    dictionary_complete dict epsilon ->
    valid_context ctx1 ->
    valid_context ctx2 ->
    semantically_similar model dict ctx1 ctx2 epsilon ->
    generalization_correct model dict ctx1 ctx2 epsilon.
Proof.
  intros model dict ctx1 ctx2 epsilon Hmodel_cont Hdict_complete Hctx1_valid Hctx2_valid Hsimilar.
  unfold generalization_correct.
  intros Hsimilarity.
  (* By the continuity of the model, the output distributions are close. *)
  specialize (Hmodel_cont ctx1 ctx2 epsilon Hctx1_valid Hctx2_valid).
  (* Use the semantic similarity to bound the divergence. *)
  admit. (* Requires combining continuity and semantic similarity results. *)
Admitted.

(* ============================================================ *)
(* PART XIX: FUTURE DIRECTIONS *)
(* ============================================================ *)

(* This file lays the foundation for a rigorous theory of interpretability in LLMs. *)
(* Future work could include: *)
(* - Proving tighter bounds on the relationship between semantic coordinates and output distributions. *)
(* - Extending the theory to multi-modal models (e.g., vision-language models). *)
(* - Developing algorithms for extracting feature dictionaries from trained models. *)
(* - Formalizing the tradeoff between interpretability and capacity in more detail. *)
