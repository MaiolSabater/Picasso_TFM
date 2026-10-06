"""Section 3.3 -- Group-Theme Optimization.

We compute a shared group theme T^G of m colours from the individual palettes,
AND the assignment g^i_j of every palette colour to a theme colour (or to 0 =
"unassociated / leave unchanged").

The objective (paper Eq. 1), minimised over theme colours and assignments:

  sum_i sum_j  (1 - delta(g^i_j)) * w^i_j * || P^i_j - T^G_{g^i_j} ||^2      (data term)
  + gamma * sum_i sum_{j1} sum_{j2 != j1} h(g^i_j1, g^i_j2)                  (palette-reduction penalty)
  + eta   * sum_i sum_j w^i_j * delta(g^i_j)                                 (unassociated-colours cost)

  delta(x) = 1 if x == 0 else 0
  h(x, y)  = 1 if x == y and x != 0 and y != 0 else 0

Assignments are labels in {0, 1, ..., m}, where 0 means "no theme colour"
(colour is left unchanged). Theme colours are indexed 1..m internally.

Parameter presets (paper "Parameters and Medoid vs. Mean"):
  MODE_KMEANS            gamma=0,    eta=1e10   (plain weighted k-means)
  MODE_MIN_REDUCTION     gamma=1e2,  eta=1e10
  MODE_ALLOW_UNASSIGNED  gamma=0,    eta=1e3
  MODE_BOTH              gamma=1e2,  eta=25e3

After optimisation each theme colour is replaced by its MEDOID -- the actual
palette colour nearest the cluster mean -- to avoid inventing new colours.
"""

from dataclasses import dataclass, field

import numpy as np

from .palette import Palette

MODE_KMEANS = dict(gamma=0.0, eta=1e10)
MODE_MIN_REDUCTION = dict(gamma=1e2, eta=1e10)
MODE_ALLOW_UNASSIGNED = dict(gamma=0.0, eta=1e3)
MODE_BOTH = dict(gamma=1e2, eta=25e3)


@dataclass
class GroupThemeResult:
    theme_ab: np.ndarray            # (m, 2) group theme colours in ab
    theme_L: np.ndarray             # (m,)   representative L per theme colour
    assignments: list               # list of (k_i,) int arrays, labels in 0..m
    palettes: list = field(default_factory=list)

    @property
    def m(self):
        return self.theme_ab.shape[0]

    def theme_lab(self):
        return np.concatenate([self.theme_L[:, None], self.theme_ab], axis=1)


def _init_theme_kmeans(all_ab, all_w, m, seed):
    from sklearn.cluster import KMeans
    m_eff = min(m, len(np.unique(all_ab, axis=0)))
    km = KMeans(n_clusters=m_eff, n_init=4, random_state=seed)
    km.fit(all_ab, sample_weight=all_w)
    centers = km.cluster_centers_
    if len(centers) < m:  # pad if fewer unique colours than requested
        pad = np.repeat(centers[-1:], m - len(centers), axis=0)
        centers = np.vstack([centers, pad])
    return centers


def _assign_one_palette(P_ab, P_w, theme_ab, gamma, eta):
    """Assignment step for a single palette (paper's brute-force per image).

    Returns labels in {0..m} minimising this palette's contribution to Eq. 1.
    We handle the pairwise palette-reduction term with a greedy pass, which is
    exact when gamma is large enough to forbid collisions (the intended regime)
    and a good approximation otherwise. Palettes are tiny (~5 colours).
    """
    k = len(P_ab)
    m = len(theme_ab)

    # Cost of assigning colour j to theme t (t = 1..m): weighted sq. distance.
    # Cost of assigning to 0 (unassigned): eta * w_j.
    d2 = np.sum((P_ab[:, None, :] - theme_ab[None, :, :]) ** 2, axis=2)  # (k, m)
    data_cost = P_w[:, None] * d2                                        # (k, m)
    unassigned_cost = eta * P_w                                         # (k,)

    # Per-colour cost of each possible label 0..m:
    #   label 0        -> unassigned_cost[j]
    #   label t (1..m) -> data_cost[j, t-1]
    # Full per-colour cost table, shape (k, m+1).
    cost = np.empty((k, m + 1))
    cost[:, 0] = unassigned_cost
    cost[:, 1:] = data_cost

    if gamma <= 0:
        # No collision penalty: each colour chooses independently (exact).
        return np.argmin(cost, axis=1).astype(int)

    # With the palette-reduction penalty we must account for interactions
    # between this palette's colours, so we solve Eq. 1 EXACTLY for this
    # palette by enumerating all label combinations. Palettes are tiny
    # (k ~ 5-7, m ~ 5), so this is fast and, crucially, cannot loop.
    #
    # If the space is unexpectedly large, fall back to the independent choice
    # (collision-agnostic), which is still a valid assignment.
    n_combos = (m + 1) ** k
    if n_combos > 2_000_000:
        return np.argmin(cost, axis=1).astype(int)

    best_labels = np.argmin(cost, axis=1).astype(int)
    best_cost = np.inf
    # Iterate combinations via a mixed-radix counter (base m+1, k digits).
    labels = np.zeros(k, dtype=int)
    for _ in range(n_combos):
        total = cost[np.arange(k), labels].sum()
        # palette-reduction penalty: gamma per ORDERED pair sharing a nonzero
        # theme colour (matches h() summed over j1 != j2 in Eq. 1).
        for t in range(1, m + 1):
            c = np.count_nonzero(labels == t)
            if c > 1:
                total += gamma * c * (c - 1)
        if total < best_cost:
            best_cost = total
            best_labels = labels.copy()
        # increment mixed-radix counter
        for d in range(k):
            labels[d] += 1
            if labels[d] <= m:
                break
            labels[d] = 0
    return best_labels


def optimize_group_theme(palettes, m, mode=MODE_KMEANS, gamma=None, eta=None,
                         max_iter=50, seed=0, use_medoid=True):
    """Run the EM optimisation of Eq. 1 over a list of Palette objects.

    m     : requested number of group-theme colours.
    mode  : one of the MODE_* preset dicts (overridden by explicit gamma/eta).
    """
    if gamma is None:
        gamma = mode["gamma"]
    if eta is None:
        eta = mode["eta"]

    all_ab = np.vstack([p.ab for p in palettes])
    all_w = np.concatenate([p.weights for p in palettes])
    all_L = np.concatenate([p.L for p in palettes])

    theme_ab = _init_theme_kmeans(all_ab, all_w, m, seed)

    assignments = [np.zeros(p.k, dtype=int) for p in palettes]
    prev = None
    for _ in range(max_iter):
        # ----- Assignment step (fix theme, solve g^i per palette) -----
        for i, p in enumerate(palettes):
            assignments[i] = _assign_one_palette(
                p.ab, p.weights, theme_ab, gamma, eta)

        flat = np.concatenate(assignments)
        if prev is not None and np.array_equal(flat, prev):
            break
        prev = flat

        # ----- Update step (fix g, recompute theme colours) -----
        for t in range(1, len(theme_ab) + 1):
            num = np.zeros(2)
            den = 0.0
            for i, p in enumerate(palettes):
                sel = assignments[i] == t
                if sel.any():
                    w = p.weights[sel]
                    num += np.sum(w[:, None] * p.ab[sel], axis=0)
                    den += w.sum()
            if den > 0:
                theme_ab[t - 1] = num / den
            # empty clusters keep their previous position

    # ----- Medoid replacement + representative L -----
    theme_L = np.full(len(theme_ab), 50.0)
    if use_medoid:
        for t in range(1, len(theme_ab) + 1):
            best_d = np.inf
            best_ab = theme_ab[t - 1]
            best_L = 50.0
            for i, p in enumerate(palettes):
                sel = assignments[i] == t
                if sel.any():
                    d = np.sum((p.ab[sel] - theme_ab[t - 1]) ** 2, axis=1)
                    j = np.argmin(d)
                    if d[j] < best_d:
                        best_d = d[j]
                        best_ab = p.ab[sel][j]
                        best_L = p.L[sel][j]
            theme_ab[t - 1] = best_ab
            theme_L[t - 1] = best_L
    else:
        for t in range(1, len(theme_ab) + 1):
            Ls, ws = [], []
            for i, p in enumerate(palettes):
                sel = assignments[i] == t
                if sel.any():
                    Ls.append(p.L[sel]); ws.append(p.weights[sel])
            if Ls:
                theme_L[t - 1] = np.average(np.concatenate(Ls),
                                            weights=np.concatenate(ws))

    return GroupThemeResult(theme_ab=theme_ab, theme_L=theme_L,
                            assignments=assignments, palettes=list(palettes))
