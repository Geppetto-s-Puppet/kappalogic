import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from kappalogic import (
    or_misclassification_boundary_sum,
    not_tower_threshold,
)

mpl.rcParams['font.family'] = 'DejaVu Sans'

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

# ---- Panel 1: OR(a,b) misclassification heatmap ----
xi = 1e-3
def reg(x, xi): return np.tanh(x/xi)**2
def NOT(x, xi): return 1 - reg(x, xi)
def OR(a, b, xi): return NOT(NOT(a, xi)*NOT(b, xi), xi)

u = np.linspace(0.01, 8, 400)
v = np.linspace(0.01, 8, 400)
U, V = np.meshgrid(u, v, indexing='ij')
Z = OR(U*xi, V*xi, xi)

ax = axes[0]
im = ax.pcolormesh(u, v, Z.T, cmap='RdBu_r', vmin=0, vmax=1, shading='auto')
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('OR(a, b)', fontsize=10)

# overlay the exact boundary line u+v = const (Prop 8)
K = or_misclassification_boundary_sum(xi) - 0.5*np.log(1/xi)
total = 0.5*np.log(1/xi) + K
u_line = np.linspace(0.01, 8, 100)
v_line = total - u_line
mask = (v_line > 0) & (v_line < 8)
ax.plot(u_line[mask], v_line[mask], color='black', lw=2, ls='--',
        label=f'u+v = const (Prop. 8)')

ax.set_xlabel('u = a / ξ', fontsize=11)
ax.set_ylabel('v = b / ξ', fontsize=11)
ax.set_title('OR(a, b) misclassifies a wide region\n(both a, b clearly nonzero, yet OR ≈ 0)', fontsize=11)
ax.legend(loc='upper right', fontsize=9)

# ---- Panel 2: gate threshold scaling laws (NOT-tower + OR family) ----
ax2 = axes[1]
xis = np.logspace(-2, -8, 30)

# NOT-tower thresholds for n=1 (NAND-like), n=2 (XOR b=0), n=3 (XNOR b=0)
for n, label, color in [(1, 'NAND-type: n=1 (~ξ^1.5)', 'tab:orange'),
                          (2, 'XOR b=0: n=2', 'tab:green'),
                          (3, 'XNOR b=0: n=3', 'tab:red')]:
    vals = [not_tower_threshold(n, xi) for xi in xis]
    ax2.loglog(xis, vals, 'o-', color=color, label=label, markersize=3)

# AND's own resonance x* = xi * u_star (linear in xi, for comparison)
u_star = np.arctanh(1/np.sqrt(3))
ax2.loglog(xis, xis*u_star, 'o-', color='tab:blue', label='AND resonance: O(ξ)', markersize=3)

ax2.set_xlabel('ξ', fontsize=11)
ax2.set_ylabel('threshold value', fontsize=11)
ax2.set_title('Different gates have genuinely different\nξ→0 scaling laws (log-log)', fontsize=11)
ax2.legend(loc='upper left', fontsize=8.5)
ax2.invert_xaxis()

plt.tight_layout()
plt.savefig('kappalogic_overview.png', dpi=150, bbox_inches='tight')
print("saved kappalogic_overview.png")
