"""Quick diagnostic: where does the mass blow up in the C/30 pickle?"""
import numpy as np, pickle

with open("ccpm_subdomains_C30_full.pkl", "rb") as f:
    d = pickle.load(f)

time = d["time"]
m_a, m_b, m_c, m_tot = d["m_a"], d["m_b"], d["m_c"], d["m_tot"]
theta = d["theta"]
V = d["V"]

# Find when mass drift exceeds thresholds
print("=== Mass drift milestones ===")
for thr in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
    idx = np.argmax(m_tot - 1 > thr) if np.any(m_tot - 1 > thr) else -1
    if idx > 0:
        print(f"  drift>{thr}: t={time[idx]/3600:.2f}h  theta={theta[idx]:.4f}  mA={m_a[idx]:.3f}  mB={m_b[idx]:.3f}  mC={m_c[idx]:.3f}")

print("\n=== Timeline ===")
n = len(time)
for frac in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    i = min(int(frac * (n - 1)), n - 1)
    print(f"  t={time[i]/3600:6.2f}h  V={V[i]:6.4f}  theta={theta[i]:.6f}  "
          f"m_tot={m_tot[i]:.4f}  mA={m_a[i]:.4f}  mB={m_b[i]:.4f}  mC={m_c[i]:.4f}")

# PDF snapshots in domain A
gA = d["gA"]; cp_a = d["cp_a"]
print(f"\ngA shape: {gA.shape},  cp_a: {len(cp_a)} pts [{cp_a[0]:.1f}, {cp_a[-1]:.1f}]")
for frac in [0, 0.3, 0.5, 0.7, 0.9, 1.0]:
    i = min(int(frac * (n - 1)), n - 1)
    peak = np.argmax(gA[:, i])
    total_a = np.trapezoid(gA[:, i], cp_a)
    print(f"  t={time[i]/3600:5.1f}h: peak@cell{peak:3d} (cp={cp_a[peak]:7.0f})  "
          f"int(gA)={total_a:.4f}  max={gA[peak,i]:.3e}  "
          f"g[0]={gA[0,i]:.3e}  g[-1]={gA[-1,i]:.3e}")

# PDF snapshots in domain C
gC = d["gC"]; cp_c = d["cp_c"]
print(f"\ngC shape: {gC.shape},  cp_c: {len(cp_c)} pts [{cp_c[0]:.1f}, {cp_c[-1]:.1f}]")
for frac in [0, 0.3, 0.5, 0.7, 0.9, 1.0]:
    i = min(int(frac * (n - 1)), n - 1)
    peak = np.argmax(np.abs(gC[:, i]))
    total_c = np.trapezoid(gC[:, i], cp_c)
    print(f"  t={time[i]/3600:5.1f}h: peak@cell{peak:3d} (cp={cp_c[peak]:7.0f})  "
          f"int(gC)={total_c:.4f}  max={gC[peak,i]:.3e}  "
          f"g[0]={gC[0,i]:.3e}  g[-1]={gC[-1,i]:.3e}")

# Check if gA or gC have negative values
print(f"\nmin(gA)={np.min(gA):.3e}  min(gB)={np.min(d['gB']):.3e}  min(gC)={np.min(gC):.3e}")
