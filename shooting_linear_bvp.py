import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import os

# ======================================================
# Page configuration
# ======================================================

st.set_page_config(
    page_title="Shooting Method for a Linear BVP",
    layout="wide"
)

st.title("Shooting Method for a Boundary Value Problem")

st.latex(r"""
x'(t) = \lambda x(t), \qquad x(T) = x_T
""")

st.markdown(
    "The shooting method treats the unknown initial value "
    r"$\theta = x(0)$ as a free parameter and iteratively adjusts it "
    "until the terminal condition is satisfied."
)

# ======================================================
# Sidebar: parameters
# ======================================================

with st.sidebar:
    st.header("Model parameters")

    lam = st.number_input("λ (lambda < 0)", value=-1.0, format="%.6f")
    T = st.number_input("Terminal time T", value=5.0, format="%.6f")
    x_T = st.number_input("Terminal value x_T", value=1.0, format="%.6f")

    tol = st.number_input("Tolerance", value=1e-6, format="%.1e")
    max_iter = st.number_input("Maximum iterations", value=20, step=1)

    st.markdown("---")
    st.markdown(
        "The initial bracket for the shooting parameter is constructed "
        "automatically around the theoretical solution."
    )

# ======================================================
# Exact structure (used only to build bracket)
# ======================================================

theta_star_exact = x_T * np.exp(-lam * T)

theta_left = 0.2 * theta_star_exact
theta_right = 2.0 * theta_star_exact

# ======================================================
# Model functions
# ======================================================

def exact_solution(t, theta, lam):
    return theta * np.exp(lam * t)

def F(theta):
    return theta * np.exp(lam * T) - x_T

# ======================================================
# Shooting method (bisection)
# ======================================================

theta_vals = []
F_vals = []

F_left = F(theta_left)
F_right = F(theta_right)

for k in range(max_iter):
    theta_mid = 0.5 * (theta_left + theta_right)
    F_mid = F(theta_mid)

    theta_vals.append(theta_mid)
    F_vals.append(F_mid)

    if abs(F_mid) < tol:
        break

    if F_left * F_mid < 0:
        theta_right = theta_mid
        F_right = F_mid
    else:
        theta_left = theta_mid
        F_left = F_mid

theta_star = theta_mid

# ======================================================
# Iteration table
# ======================================================

st.subheader("Shooting iterations")

st.dataframe(
    {
        "k": list(range(len(theta_vals))),
        r"$\theta_k$": theta_vals,
        r"$F(\theta_k)$": F_vals,
        "sign": ["+" if f > 0 else "-" for f in F_vals],
    },
    use_container_width=True,
)

# ======================================================
# Plot
# ======================================================

t = np.linspace(0, T, 400)

plt.figure(figsize=(8, 5))

# exact solution
plt.plot(
    t,
    exact_solution(t, theta_star, lam),
    color="black",
    linewidth=2,
    label="Exact solution"
)

# shooting trajectories
for k, theta in enumerate(theta_vals):
    plt.plot(
        t,
        exact_solution(t, theta, lam),
        linestyle="--",
        alpha=0.6,
        label=f"Shot k={k}"
    )

plt.scatter(T, x_T, color="red", zorder=5, label=r"$x(T)=x_T$")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.legend()
plt.grid(True)
plt.tight_layout()

os.makedirs("figures", exist_ok=True)
plt.savefig("figures/2.5.png", dpi=300)

st.pyplot(plt.gcf())

# ======================================================
# Summary
# ======================================================

st.success(
    rf"Converged to $\theta^\* \approx {theta_star:.6f}$ "
    rf"with $|F(\theta^\*)| < {tol:.1e}$."
)

st.markdown(
    "Each curve corresponds to a trial solution obtained for a different guess "
    "of the initial value. The shooting method iteratively refines this guess "
    "until the terminal condition is met."
)