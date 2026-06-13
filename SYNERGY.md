# Synergy: latent world models for regime-aware time series

Oneiros is a generic idea wearing a toy's clothes. Its core claim — *predict the
next state in a learned latent space, then plan in that space* — transfers
directly to sequential decision problems far from a 2D point-mass. One natural
target is regime-aware modeling of financial time series. This note sketches the
connection at a conceptual level; it deliberately contains no internals of any
specific trading system.

## A hidden-state model is already a discrete latent world model

A Hidden Markov Model (HMM) over market regimes is, structurally, a world model
with a *discrete* latent. The observable returns are emissions; the latent
regime (for example "trending", "mean-reverting", "high-volatility") is the
state that actually drives behavior. Inference gives a posterior over the current
latent regime, and the transition matrix is a one-step predictor of the *next*
latent — `P(regime_{t+1} | regime_t)`. That is exactly the JEPA shape, only with
a categorical latent and a linear (matrix) predictor:

| Oneiros (this repo) | Regime model |
|---|---|
| encoder `f`: obs -> continuous latent `z` | filtering: returns -> posterior over discrete regime |
| predictor `g`: `(z_t, a_t)` -> `z_hat_{t+1}` | transition matrix: `regime_t` -> `regime_{t+1}` |
| plan in latent space (MPC over `g`) | act on the *predicted next regime*, not raw price |
| anti-collapse keeps latents informative | regimes must stay distinct to be useful |

The shared principle in both columns: **do not predict the raw, noisy
observable. Predict the compact latent that governs it, and make decisions on the
latent.** Predicting raw next-tick price is the time-series analogue of
pixel reconstruction — high variance, low signal, and mostly modeling noise the
decision does not care about.

## The research extension: continuous JEPA-style regime dynamics

A discrete HMM regime is a coarse, hand-specified latent. The natural extension,
mirroring exactly what this repo does, is to **learn a continuous latent regime
representation and a learned dynamics model over it**:

1. **Encoder `f`.** Map a window of features (returns, realized volatility,
   cross-asset spreads, order-flow summaries) to a continuous latent `z_t` —
   the analogue of Oneiros encoding an observation. This is richer than a
   one-hot regime: regimes can blend and drift continuously rather than
   snapping between discrete buckets.

2. **Predictor `g`.** Learn `z_hat_{t+1} = g(z_t, context_t)` to predict the
   *next latent regime embedding* (not the next price), trained against an EMA
   target encoder with the same VICReg-style variance/covariance penalty used
   here. The anti-collapse machinery is not optional decoration — without it a
   latent regime predictor collapses to "tomorrow looks like today", which is
   precisely the degenerate one-step persistence an HMM transition matrix can
   already express.

3. **Planning / decisioning.** Where Oneiros runs MPC to reach a spatial goal,
   a regime model rolls `g` forward to anticipate regime *transitions* — e.g.
   the probability that a trending latent is about to drift toward a
   high-volatility one — and conditions position sizing or strategy selection on
   that predicted latent trajectory rather than on a point price forecast.

## Why the latent framing helps here specifically

* **Robustness to non-stationarity.** A latent that captures *regime* changes
  more slowly and more meaningfully than raw price, so a predictor over it
  generalizes across the many price paths a single regime can produce.
* **Decision-relevant compression.** Position sizing, hedging, and
  strategy-switching are functions of regime, not of the next tick. Predicting
  the decision-relevant latent aligns the learning objective with the actual
  use.
* **Honest evaluation carries over.** The same three gates this repo enforces —
  beat a no-op baseline, do not collapse, beat a naive policy — are exactly the
  checks a learned regime-dynamics model needs before it is trusted to inform
  any decision.

This remains a *research direction*, not a deployed component: the value of the
analogy is that the small, fully-verified artifact in this repository
demonstrates, end to end, that learned latent dynamics plus latent-space planning
is sound — and that the anti-collapse discipline is what makes it real rather
than a degenerate identity.
