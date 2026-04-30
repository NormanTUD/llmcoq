(* === StrongAIModel_v03.v === *)
(* Version 0.3 — Full detail pass.                              *)
(* Builds on the compiled v0.2 skeleton.                         *)
(* Every section is now fleshed out with concrete structure.      *)

Require Import Coq.Reals.Reals.
Require Import Coq.Reals.RIneq.
Require Import Coq.Logic.FunctionalExtensionality.
Require Import Coq.Sets.Ensembles.
Require Import Coq.Lists.List.
Require Import Coq.Arith.PeanoNat.
Require Import Coq.micromega.Lra.

Import ListNotations.
Open Scope R_scope.

(* ============================================================ *)
(* PART I: FOUNDATIONAL TYPES                                    *)
(* ============================================================ *)

(* --- 1.1 Vocabulary and Tokens --- *)
(* A vocabulary is a finite set. We model token IDs as nat. *)
Parameter vocab_size : nat.
Axiom vocab_size_pos : (vocab_size > 0)%nat.

Definition Token := nat.

(* A token is valid if it's within vocabulary bounds *)
Definition valid_token (t : Token) : Prop := (t < vocab_size)%nat.

(* --- 1.2 Vectors and Matrices (abstract linear algebra) --- *)
(* We parameterize by a dimension rather than building full LA. *)

Parameter dim_model : nat.       (* d_model: embedding dimension *)
Parameter dim_ff : nat.          (* d_ff: feedforward hidden dim *)
Parameter num_heads : nat.       (* number of attention heads *)
Parameter dim_head : nat.        (* d_k = d_model / num_heads *)

Axiom dim_model_pos : (dim_model > 0)%nat.
Axiom num_heads_pos : (num_heads > 0)%nat.

(* Abstract vector type parameterized by dimension *)
Parameter Vec : nat -> Type.

(* Abstract matrix type: rows x cols *)
Parameter Mat : nat -> nat -> Type.

(* Core linear algebra operations — all postponed *)
Parameter vec_zero : forall n, Vec n.
Parameter vec_add : forall n, Vec n -> Vec n -> Vec n.
Parameter vec_scale : forall n, R -> Vec n -> Vec n.
Parameter mat_vec_mul : forall m n, Mat m n -> Vec n -> Vec m.
Parameter vec_dot : forall n, Vec n -> Vec n -> R.
Parameter vec_norm : forall n, Vec n -> R.

(* Properties of vec operations *)
Axiom vec_add_comm : forall n (u v : Vec n), vec_add n u v = vec_add n v u.
Axiom vec_add_assoc : forall n (u v w : Vec n),
  vec_add n u (vec_add n v w) = vec_add n (vec_add n u v) w.
Axiom vec_add_zero_r : forall n (v : Vec n), vec_add n v (vec_zero n) = v.
Axiom vec_norm_nonneg : forall n (v : Vec n), (vec_norm n v >= 0)%R.
Axiom vec_norm_zero : forall n, vec_norm n (vec_zero n) = 0%R.

(* --- 1.3 Probability Distributions --- *)
(* A distribution over the vocabulary is a Vec vocab_size *)
(* whose entries are non-negative and sum to 1. *)

Definition RawDist := Vec vocab_size.

Parameter vec_entry : forall n, Vec n -> nat -> R.
Parameter vec_sum : forall n, Vec n -> R.

Definition is_distribution (d : RawDist) : Prop :=
  (forall i, (i < vocab_size)%nat -> vec_entry vocab_size d i >= 0) /\
  vec_sum vocab_size d = 1%R.

Record Dist := mkDist {
  dist_vec : RawDist;
  dist_valid : is_distribution dist_vec
}.

(* --- 1.4 Contexts (sequences of embedded tokens) --- *)
(* A context is a list of token embeddings. *)

Definition Embedding := Vec dim_model.

Definition Context := list Embedding.

Definition context_length (ctx : Context) : nat := length ctx.

(* Maximum context length *)
Parameter max_seq_len : nat.
Axiom max_seq_len_pos : (max_seq_len > 0)%nat.

Definition valid_context (ctx : Context) : Prop :=
  (context_length ctx <= max_seq_len)%nat /\ (context_length ctx > 0)%nat.

(* Token embedding matrix *)
Parameter embedding_matrix : Mat dim_model vocab_size.

(* Embed a token *)
Parameter one_hot : nat -> Vec vocab_size.
Definition embed_token (t : Token) : Embedding :=
  mat_vec_mul dim_model vocab_size embedding_matrix (one_hot t).

(* ============================================================ *)
(* PART II: TRANSFORMER ARCHITECTURE                             *)
(* ============================================================ *)

(* --- 2.1 Softmax --- *)
Parameter softmax : forall n, Vec n -> Vec n.

Axiom softmax_is_distribution : forall n (v : Vec n),
  (n > 0)%nat ->
  (forall i, (i < n)%nat -> vec_entry n (softmax n v) i >= 0) /\
  vec_sum n (softmax n v) = 1%R.

(* --- 2.2 Layer Normalization --- *)
Parameter layer_norm : Vec dim_model -> Vec dim_model.

(* LayerNorm produces unit-variance output (simplified axiom) *)
Axiom layer_norm_normalized : forall v : Vec dim_model,
  exists mu sigma,
    (sigma > 0)%R /\
    layer_norm v = vec_scale dim_model (1 / sigma) (vec_add dim_model v (vec_scale dim_model (-1 * mu) (vec_zero dim_model))).
    (* This is a simplification — real LN subtracts mean and divides by std *)

(* --- 2.3 Attention Head --- *)
Record AttentionHead := mkAttentionHead {
  W_Q : Mat dim_head dim_model;   (* Query projection *)
  W_K : Mat dim_head dim_model;   (* Key projection *)
  W_V : Mat dim_head dim_model;   (* Value projection *)
}.

(* Compute query, key, value for a single position *)
Definition compute_query (h : AttentionHead) (x : Embedding) : Vec dim_head :=
  mat_vec_mul dim_head dim_model (W_Q h) x.

Definition compute_key (h : AttentionHead) (x : Embedding) : Vec dim_head :=
  mat_vec_mul dim_head dim_model (W_K h) x.

Definition compute_value (h : AttentionHead) (x : Embedding) : Vec dim_head :=
  mat_vec_mul dim_head dim_model (W_V h) x.

(* Attention score between query position i and key position j *)
(* score(i,j) = Q_i . K_j / sqrt(d_k) *)
Parameter sqrt_dim_head : R.
Axiom sqrt_dim_head_pos : (sqrt_dim_head > 0)%R.
Axiom sqrt_dim_head_correct : (sqrt_dim_head * sqrt_dim_head = INR dim_head)%R.

Definition attention_score (h : AttentionHead) (qi kj : Embedding) : R :=
  (vec_dot dim_head (compute_query h qi) (compute_key h kj)) / sqrt_dim_head.

(* Causal mask: position i can only attend to positions j <= i *)
Definition causal_mask (i j : nat) : R :=
  if (j <=? i)%nat then 1%R else 0%R.

(* --- 2.4 Multi-Head Attention --- *)
Record MultiHeadAttention := mkMHA {
  heads : list AttentionHead;
  W_O : Mat dim_model (num_heads * dim_head);  (* Output projection *)
  mha_num_heads_correct : length heads = num_heads;
}.

(* We abstract the full MHA computation *)
Parameter mha_apply : MultiHeadAttention -> Context -> Context.

(* Key property: MHA preserves sequence length *)
Axiom mha_preserves_length : forall (mha : MultiHeadAttention) (ctx : Context),
  length (mha_apply mha ctx) = length ctx.

(* Key property: MHA respects causality — output at position i *)
(* depends only on inputs at positions 0..i *)
Axiom mha_causal : forall (mha : MultiHeadAttention) (ctx : Context) (i : nat),
  (i < length ctx)%nat ->
  forall ctx',
    length ctx' = length ctx ->
    (forall j, (j <= i)%nat -> (j < length ctx)%nat -> 
      nth_error ctx j = nth_error ctx' j) ->
    nth_error (mha_apply mha ctx) i = nth_error (mha_apply mha ctx') i.

(* --- 2.5 Feed-Forward Network --- *)
Record FFN := mkFFN {
  W1 : Mat dim_ff dim_model;
  b1 : Vec dim_ff;
  W2 : Mat dim_model dim_ff;
  b2 : Vec dim_model;
}.

(* ReLU activation (or GELU — we abstract) *)
Parameter activation : forall n, Vec n -> Vec n.

(* FFN applied to a single vector: W2 * activation(W1 * x + b1) + b2 *)
Definition ffn_apply_single (f : FFN) (x : Embedding) : Embedding :=
  vec_add dim_model
    (mat_vec_mul dim_model dim_ff (W2 f)
      (activation dim_ff
        (vec_add dim_ff
          (mat_vec_mul dim_ff dim_model (W1 f) x)
          (b1 f))))
    (b2 f).

(* FFN applied position-wise to a context *)
Definition ffn_apply (f : FFN) (ctx : Context) : Context :=
  map (ffn_apply_single f) ctx.

Lemma ffn_preserves_length : forall (f : FFN) (ctx : Context),
  length (ffn_apply f ctx) = length ctx.
Proof.
  intros f ctx.
  unfold ffn_apply.
  apply map_length.
Qed.

(* --- 2.6 Residual Connection --- *)
(* residual(x, sublayer) = layer_norm(x + sublayer(x)) *)
(* We need pointwise addition on contexts *)

(* Note: |> might not be available. Let's use a let binding instead. *)
(* Actually, let me rewrite without pipe operator *)

Definition context_add' (ctx1 ctx2 : Context) : Context :=
  map (fun p => vec_add dim_model (fst p) (snd p)) (combine ctx1 ctx2).

Definition context_layer_norm (ctx : Context) : Context :=
  map layer_norm ctx.

Definition residual_connection (sublayer : Context -> Context) (ctx : Context) : Context :=
  context_layer_norm (context_add' ctx (sublayer ctx)).

(* --- 2.7 Transformer Layer --- *)
Record TransformerLayer := mkTransformerLayer {
  tl_mha : MultiHeadAttention;
  tl_ffn : FFN;
}.

Definition transformer_layer_apply (l : TransformerLayer) (ctx : Context) : Context :=
  let after_attn := residual_connection (mha_apply (tl_mha l)) ctx in
  let after_ffn := residual_connection (ffn_apply (tl_ffn l)) after_attn in
  after_ffn.

(* --- 2.8 Full Transformer Model --- *)
Record TransformerModel := mkTransformerModel {
  tm_layers : list TransformerLayer;
  tm_unembed : Mat vocab_size dim_model;  (* Final linear layer to logits *)
  tm_num_layers : nat;
  tm_layers_correct : length tm_layers = tm_num_layers;
}.

(* Apply all layers sequentially *)
Fixpoint apply_all_layers (layers : list TransformerLayer) (ctx : Context) : Context :=
  match layers with
  | [] => ctx
  | l :: rest => apply_all_layers rest (transformer_layer_apply l ctx)
  end.

(* Helper: last element of a list *)
(* Actually, let's define last_error since it might not exist *)
Definition last_error {A : Type} (l : list A) : option A :=
  match rev l with
  | [] => None
  | x :: _ => Some x
  end.

(* The full forward pass *)
Definition transformer_forward (model : TransformerModel) (ctx : Context) : Embedding :=
  let final_ctx := apply_all_layers (tm_layers model) ctx in
  match last_error final_ctx with
  | Some last_emb => last_emb
  | None => vec_zero dim_model
  end.

(* Logits: project final embedding to vocabulary space *)
Definition compute_logits (model : TransformerModel) (ctx : Context) : Vec vocab_size :=
  mat_vec_mul vocab_size dim_model (tm_unembed model) (transformer_forward model ctx).

(* Output distribution: softmax of logits *)
Definition output_distribution (model : TransformerModel) (ctx : Context) : RawDist :=
  softmax vocab_size (compute_logits model ctx).

(* ============================================================ *)
(* PART III: LOSS FUNCTIONS                                      *)
(* ============================================================ *)

(* --- 3.1 Cross-Entropy Loss --- *)
Parameter ln : R -> R.  (* Natural logarithm *)
Axiom ln_pos : forall x, (x > 0)%R -> (ln x <= 0)%R \/ (ln x > 0)%R.
Axiom ln_one : ln 1 = 0%R.
Axiom ln_monotone : forall x y, (0 < x)%R -> (x <= y)%R -> (ln x <= ln y)%R.

(* Cross-entropy: H(p, q) = - sum_i p_i * ln(q_i) *)
Parameter cross_entropy : RawDist -> RawDist -> R.

Axiom cross_entropy_nonneg : forall p q,
  is_distribution p -> is_distribution q ->
  (cross_entropy p q >= 0)%R.

(* Gibbs' inequality: cross-entropy is minimized when q = p *)
Axiom gibbs_inequality : forall p q,
  is_distribution p -> is_distribution q ->
  (cross_entropy p q >= cross_entropy p p)%R.

(* Cross-entropy equals entropy when p = q *)
Definition entropy (p : RawDist) : R := cross_entropy p p.

Axiom cross_entropy_eq_iff_kl_zero : forall p q,
  is_distribution p -> is_distribution q ->
  cross_entropy p q = cross_entropy p p <-> p = q.

(* --- 3.2 KL Divergence --- *)
Definition kl_divergence (p q : RawDist) : R :=
  (cross_entropy p q - entropy p)%R.

Lemma kl_nonneg : forall p q,
  is_distribution p -> is_distribution q ->
  (kl_divergence p q >= 0)%R.
Proof.
  intros p q Hp Hq.
  unfold kl_divergence, entropy.
  pose proof (gibbs_inequality p q Hp Hq).
  lra.
Qed.

Lemma kl_zero_iff : forall p q,
  is_distribution p -> is_distribution q ->
  kl_divergence p q = 0%R <-> p = q.
Proof.
  intros p q Hp Hq.
  unfold kl_divergence, entropy.
  split.
  - intro H.
    assert (cross_entropy p q = cross_entropy p p) by lra.
    apply cross_entropy_eq_iff_kl_zero; assumption.
  - intro H. subst. lra.
Qed.

(* ============================================================ *)
(* PART IV: TRAINING DYNAMICS                                    *)
(* ============================================================ *)

(* --- 4.1 Dataset --- *)
Record DataPoint := mkDataPoint {
  dp_context : Context;
  dp_target : RawDist;
  dp_context_valid : valid_context dp_context;
  dp_target_valid : is_distribution dp_target;
}.

Definition Dataset := list DataPoint.

(* --- 4.2 Empirical Loss --- *)
(* Average cross-entropy over a dataset *)
Parameter sum_losses : TransformerModel -> Dataset -> R.

Axiom sum_losses_def : forall model ds,
  sum_losses model ds = 
    fold_left (fun acc dp => 
      (acc + cross_entropy (dp_target dp) (output_distribution model (dp_context dp)))%R
    ) ds 0%R.

Definition empirical_loss (model : TransformerModel) (ds : Dataset) : R :=
  match ds with
  | [] => 0%R
  | _ => (sum_losses model ds / INR (length ds))%R
  end.

(* --- 4.3 Gradient Descent (Abstract) --- *)
(* We abstract the gradient as an operation on the full model *)
Parameter model_gradient : TransformerModel -> DataPoint -> TransformerModel.
Parameter model_update : TransformerModel -> TransformerModel -> R -> TransformerModel.
  (* model_update current grad learning_rate *)

Parameter learning_rate : R.
Axiom lr_pos : (learning_rate > 0)%R.
Axiom lr_small : (learning_rate < 1)%R.

Definition sgd_step (model : TransformerModel) (dp : DataPoint) : TransformerModel :=
  model_update model (model_gradient model dp) learning_rate.

(* --- 4.4 Training Loop --- *)
Fixpoint train (model : TransformerModel) (data : list DataPoint) : TransformerModel :=
  match data with
  | [] => model
  | dp :: rest => train (sgd_step model dp) rest
  end.

(* --- 4.5 Convergence Axiom (to be refined) --- *)
(* After sufficient training, empirical loss decreases *)
Axiom sgd_decreases_loss : forall (model : TransformerModel) (dp : DataPoint),
  (cross_entropy (dp_target dp) (output_distribution (sgd_step model dp) (dp_context dp)) <=
   cross_entropy (dp_target dp) (output_distribution model (dp_context dp)))%R.

(* ============================================================ *)
(* PART V: GENERALIZATION                                        *)
(* ============================================================ *)

(* --- 5.1 True Distribution --- *)
(* There exists a "true" data-generating distribution *)
Parameter true_distribution : Context -> RawDist.
Axiom true_dist_valid : forall ctx, valid_context ctx -> is_distribution (true_distribution ctx).

(* --- 5.2 Expected Loss (Population Risk) --- *)
(* We can't compute integrals in Coq easily, so we axiomatize *)
Parameter expected_loss : TransformerModel -> R.

Axiom expected_loss_nonneg : forall model, (expected_loss model >= 0)%R.

(* --- 5.3 Generalization Bound --- *)
(* PAC-style bound: with high probability, empirical loss *)
(* approximates expected loss. Inspired by MLCERT. *)

Axiom generalization_bound : forall (model : TransformerModel) (ds : Dataset) (delta : R),
  (delta > 0)%R ->
  (INR (length ds) > 0)%R ->
  (* With probability at least 1 - delta: *)
  exists (complexity_term : R),
    (complexity_term >= 0)%R /\
    (Rabs (expected_loss model - empirical_loss model ds) <= 
     complexity_term / sqrt (INR (length ds)) + sqrt (ln (1 / delta) / (2 * INR (length ds))))%R.

(* ============================================================ *)
(* PART VI: STRONG MODEL DEFINITION (REFINED)                   *)
(* ============================================================ *)

(* --- 6.1 Task Formalization --- *)
Record Task := mkTask {
  task_description : Context;  (* prompt / input *)
  task_ideal_response : RawDist;  (* ideal next-token distribution *)
  task_valid_ctx : valid_context task_description;
  task_valid_dist : is_distribution task_ideal_response;
}.

(* --- 6.2 Capability Levels --- *)
Inductive Capability : Type :=
  | LanguageModeling    (* predict next token well *)
  | Reasoning           (* multi-step logical inference *)
  | Instruction         (* follow instructions *)
  | Coding              (* generate correct code *)
  | Factual             (* recall factual knowledge *)
  | Mathematical.       (* mathematical problem solving *)

Parameter capability_tasks : Capability -> Ensemble Task.

(* --- 6.3 Strong Model: Formal Definition --- *)
Definition model_solves_task (model : TransformerModel) (t : Task) (epsilon : R) : Prop :=
  (cross_entropy (task_ideal_response t) 
    (output_distribution model (task_description t)) < epsilon)%R.

Definition model_has_capability (model : TransformerModel) (cap : Capability) (epsilon : R) : Prop :=
  forall t : Task, Ensembles.In Task (capability_tasks cap) t -> model_solves_task model t epsilon.

Definition strong_model (model : TransformerModel) (epsilon : R) : Prop :=
  forall cap : Capability, model_has_capability model cap epsilon.

(* ============================================================ *)
(* PART VII: PROVABLE THEOREMS                                   *)
(* ============================================================ *)

(* --- 7.1 KL divergence is non-negative (proved above) --- *)
(* See kl_nonneg and kl_zero_iff *)

(* --- 7.2 FFN preserves context length (proved above) --- *)
(* See ffn_preserves_length *)

(* --- 7.3 Training monotonically decreases pointwise loss --- *)
Lemma training_monotone_pointwise :
  forall (model : TransformerModel) (dp : DataPoint),
  (cross_entropy (dp_target dp) (output_distribution (sgd_step model dp) (dp_context dp)) <=
   cross_entropy (dp_target dp) (output_distribution model (dp_context dp)))%R.
Proof.
  intros model dp.
  apply sgd_decreases_loss.
Qed.

(* --- 7.4 Multi-step training monotonicity --- *)
Lemma training_multi_step_monotone :
  forall (model : TransformerModel) (dp : DataPoint) (n : nat),
  let trained := Nat.iter n (fun m => sgd_step m dp) model in
  (cross_entropy (dp_target dp) (output_distribution trained (dp_context dp)) <=
   cross_entropy (dp_target dp) (output_distribution model (dp_context dp)))%R.
Proof.
  intros model dp n.
  simpl.
  induction n.
  - simpl. lra.
  - simpl.
    pose proof (sgd_decreases_loss 
      (Nat.iter n (fun m => sgd_step m dp) model) dp) as Hstep.
    lra.
Qed.

(* --- 7.5 A strong model solves all tasks of every capability --- *)
Lemma strong_implies_all_tasks :
  forall model epsilon,
    strong_model model epsilon ->
    forall cap t,
      Ensembles.In Task (capability_tasks cap) t ->
      model_solves_task model t epsilon.
Proof.
  intros model epsilon Hstrong cap t Hin.
  unfold strong_model in Hstrong.
  unfold model_has_capability in Hstrong.
  exact (Hstrong cap t Hin).
Qed.

(* --- 7.6 Composition of layers preserves context length --- *)
Lemma apply_all_layers_length :
  forall layers ctx,
  (forall l c, length (transformer_layer_apply l c) = length c) ->
  length (apply_all_layers layers ctx) = length ctx.
Proof.
  intros layers.
  induction layers as [| l rest IH].
  - intros ctx Hpres. simpl. reflexivity.
  - intros ctx Hpres. simpl.
    rewrite IH.
    + apply Hpres.
    + exact Hpres.
Qed.

(* --- 7.7 Existence of a strong model (main theorem, axiomatized) --- *)
Theorem strong_model_existence :
  exists (model : TransformerModel) (epsilon : R),
    (epsilon > 0)%R /\ strong_model model epsilon.
Proof.
  (* This is the fundamental claim. We cannot construct the model *)
  (* within Coq — it requires training on data. We axiomatize. *)
  (* In a full development, this would follow from:              *)
  (*   1. Universal approximation for transformers               *)
  (*   2. Sufficient training data                               *)
  (*   3. Convergence of SGD                                     *)
  (* For now, we admit it. *)
Admitted.

(* ============================================================ *)
(* PART VIII: AUTOREGRESSIVE GENERATION                          *)
(* ============================================================ *)

Parameter sample_from_dist : RawDist -> Token.
Axiom sample_valid : forall d, is_distribution d -> valid_token (sample_from_dist d).

Fixpoint autoregressive_generate 
  (model : TransformerModel) (ctx : Context) (n : nat) : list Token :=
  match n with
  | O => []
  | S n' =>
      let dist := output_distribution model ctx in
      let tok := sample_from_dist dist in
      let new_emb := embed_token tok in
      let ctx' := ctx ++ [new_emb] in
      tok :: autoregressive_generate model ctx' n'
  end.

(* Generation produces valid tokens *)
Lemma generation_valid_tokens :
  forall model ctx n,
    valid_context ctx ->
    Forall valid_token (autoregressive_generate model ctx n).
Proof.
  intros model ctx n Hvalid.
  revert ctx Hvalid.
  induction n as [| n' IH].
  - intros. simpl. constructor.
  - intros ctx Hvalid. simpl.
    constructor.
    + apply sample_valid.
      apply (softmax_is_distribution vocab_size (compute_logits model ctx)).
      exact vocab_size_pos.
    + apply IH.
      (* We need: appending one embedding to a valid context is valid *)
      (* This requires max_seq_len to be large enough — we admit for now *)
      admit.
Admitted.

(* ============================================================ *)
(* PART IX: CATEGORICAL STRUCTURE (Presheaf Perspective)         *)
(* ============================================================ *)

(* The "presheaf" perspective: a model's knowledge can be viewed *)
(* as a functor from contexts (a category) to distributions.     *)

(* Objects: contexts (ordered by prefix relation) *)
Definition is_prefix (ctx1 ctx2 : Context) : Prop :=
  exists suffix, ctx2 = ctx1 ++ suffix.

Lemma prefix_refl : forall ctx, is_prefix ctx ctx.
Proof.
  intro ctx. exists []. rewrite app_nil_r. reflexivity.
Qed.

Lemma prefix_trans : forall c1 c2 c3,
  is_prefix c1 c2 -> is_prefix c2 c3 -> is_prefix c1 c3.
Proof.
  intros c1 c2 c3 [s1 H1] [s2 H2].
  exists (s1 ++ s2). subst. rewrite app_assoc. reflexivity.
Qed.

(* The model defines a "presheaf": for each context, a distribution *)
(* Restriction maps: if ctx1 is a prefix of ctx2, the model's *)
(* prediction at ctx1 is "consistent" with its prediction at ctx2 *)
(* in a sense we can formalize. *)

(* Coherence: the model's output at a prefix is the marginal *)
(* of its output at any extension. This is an idealized property. *)
Axiom model_coherence : forall (model : TransformerModel) (ctx : Context) (ext : list Embedding),
  valid_context ctx ->
  valid_context (ctx ++ ext) ->
  (* The distribution at ctx "predicts" the path through ext *)
  (* This is a deep property — we state it abstractly *)
  True. (* Placeholder — to be refined with actual marginalization *)

(* ============================================================ *)
(* END Version 0.3                                               *)
(* ============================================================ *)
