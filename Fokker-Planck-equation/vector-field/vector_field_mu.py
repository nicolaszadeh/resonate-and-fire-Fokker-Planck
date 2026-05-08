# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 15:35:04 2026

@author: Nicolas Zadeh
"""

import numpy as np
import matplotlib.pyplot as plt

# Numerical parameters
u_F=9

x_min=-10
x_max=u_F

v_min=-60
v_max=60

b=1
nu=1
omega_0=2
tau=2
N_t=0.0

if (1/tau**2-4*omega_0**2<0):
    print("Oscillatory framework")
else:
    print("Non-oscillatory framework")
    exit()

# Fonts
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})

fig, ax = plt.subplots()

# added room to the right
right_margin = 0.7
ax.set_xlim(x_min, u_F + right_margin)
ax.set_ylim(v_min*0.8, v_max*0.8)

ax.set_xlabel(r"$x$")
ax.xaxis.set_label_coords(0.5, -0.02)
ax.set_ylabel(r"$v$", rotation=0, labelpad=10)
ax.yaxis.set_label_coords(-0.022, 0.72)

# definition of x and v

x = np.linspace(x_min, u_F*0.93, 9)
v=np.linspace(v_min, v_max, 10)
# main lines
ax.axvline(x=u_F, color="blue", linestyle="--", lw=0.4,label=r"$x=u_F$")
ax.axhline(y=0, color="black", label=r"$v=0$")
ax.plot(x, (-x * omega_0**2+b*(nu+N_t)) * tau, label=r"$v=-x\tau\omega_0^2+b\tau(\nu_{\rm ext}+N(t))$")

# no tick marks
ax.tick_params(axis='both', which='both', length=0)
# keep only the threshold on x
ax.set_xticks([u_F])
ax.set_xticklabels([r"$u_{\rm F}$"])

# keep only v = 0 on y
ax.set_yticks([0])
ax.set_yticklabels([r"$0$"])

# point P
xP = b * (nu + N_t) / omega_0**2
vP = 0.0

ax.plot(xP, vP, 'ko', markersize=6, zorder=5)
ax.annotate(
    r"$P(t)$",
    xy=(xP, vP),
    xytext=(-6, 22),
    textcoords="offset points",
    fontsize=14
)

# add a node for P, the center of the spirals

x_nodes = np.sort(np.append(x, xP))   # insert x_P

v_nodes = np.linspace(v_min, v_max, 10)

X_full, V_full = np.meshgrid(x_nodes, v_nodes)

# The arrows cannot go further than the threshold barrier
mask = X_full+ 0.02*V_full< u_F

# Plotting the field
ax.quiver(
    X_full[mask], V_full[mask],
    V_full[mask],
    -V_full[mask]/tau - X_full[mask]*omega_0**2 + b*(nu + N_t),
    color="red",
    width=0.003
)

ax.legend(loc='lower left')

plt.savefig("vector_field.pdf", format="pdf", bbox_inches="tight")

plt.tight_layout()
plt.show()


print("Done!")


