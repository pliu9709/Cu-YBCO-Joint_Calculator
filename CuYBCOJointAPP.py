
import os
import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import streamlit as st

# ===== 强制加载仓库里的中文字体（解决 Streamlit Cloud 乱码）=====
FONT_PATH = os.path.join(os.path.dirname(__file__), "fonts", "NotoSansSC-Regular.ttf")

if os.path.exists(FONT_PATH):
    font_manager.fontManager.addfont(FONT_PATH)
    font_name = font_manager.FontProperties(fname=FONT_PATH).get_name()
    plt.rcParams["font.family"] = font_name

plt.rcParams["axes.unicode_minus"] = False
# ==========================
# Matplotlib 中文支持
# ==========================

# ==========================
# 物性函数（原样保留）
# ==========================
T_MIN = 4.0
T_MAX = 200.0

def k_cu_rrr50(T):
    coeffs = {
        "a": 1.8743, "b": -0.41538, "c": -0.6018, "d": 0.13294, "e": 0.26426,
        "f": -0.0219, "g": -0.051276, "h": 0.0014871, "i": 0.003723,
    }
    T = np.asarray(T)
    T_clamp = np.clip(T, T_MIN, T_MAX)
    a, b, c, d, e, f, g, h, i = coeffs.values()
    num = a + c*T_clamp**0.5 + e*T_clamp + g*T_clamp**1.5 + i*T_clamp**2
    den = 1.0 + b*T_clamp**0.5 + d*T_clamp + f*T_clamp**1.5 + h*T_clamp**2
    log10k = num / den
    return 10.0**log10k

def k_hastelloy(T):
    T = np.asarray(T)
    T_clamp = np.clip(T, T_MIN, T_MAX)
    a = 0.58856
    b = 0.23494
    c = -0.00292
    d = 1.679e-5
    e = -3.432e-8
    return (a + b*T_clamp + c*T_clamp**2 + d*T_clamp**3 + e*T_clamp**4) * 0.85

def k_ss304(T):
    ss_coeffs = {
        "a": -1.4087, "b": 1.3982, "c": 0.2543, "d": 0.02406, "e": 0.0,
        "f": 0.4256, "g": -0.4858, "h": 0.1650, "i": -0.01159,
    }
    T = np.asarray(T)
    T_clamp = np.clip(T, T_MIN, T_MAX)
    x_log = np.log10(T_clamp)
    a,b,c,d,e,f,g,h,i = ss_coeffs.values()
    log10k = (a + b*x_log + c*x_log**2 + d*x_log**3 + e*x_log**4
              + f*x_log**5 + g*x_log**6 + h*x_log**7 + i*x_log**8)
    return 10.0**log10k

def rho_cu_rrr50(T):
    T = np.asarray(T)
    T_clamp = np.clip(T, T_MIN, T_MAX)
    rho = (
        0.06948
        - 0.00434 * T_clamp
        + 1.17e-4 * T_clamp**2
        - 5.135e-7 * T_clamp**3
        + 7.55179e-10 * T_clamp**4
    ) * 1e-8
    return rho

# ==========================
# 三对角求解器（Thomas）
# ==========================
def solve_tridiag(a, b, c, d):
    a = a.astype(float).copy()
    b = b.astype(float).copy()
    c = c.astype(float).copy()
    d = d.astype(float).copy()
    n = len(b)

    for i in range(1, n):
        w = a[i] / b[i-1]
        b[i] -= w * c[i-1]
        d[i] -= w * d[i-1]

    xsol = np.zeros_like(d)
    xsol[-1] = d[-1] / b[-1]
    for i in range(n-2, -1, -1):
        xsol[i] = (d[i] - c[i] * xsol[i+1]) / b[i]
    return xsol

# ==========================
# 核心计算封装
# ==========================
def run_model(
    # 边界/电流
    T_H: float, T_C: float, I: float,
    # 铜段
    A_cu: float, L_cu: float,
    # YBCO段
    L_ybco: float, A_shunt: float, n_ybco: int,
    # 接头
    R_joint: float, L_joint: float = 0.02,
    # 数值
    N: int = 501,
):
    # ===== 带材几何：随 n_ybco 变化 =====
    tape_width = 4.0e-3
    t_hast_single = 45e-6
    t_cu_single = 16e-6

    t_hast = t_hast_single * n_ybco
    t_cu_tape = t_cu_single * n_ybco

    A_hast = tape_width * t_hast
    A_cu_tape = tape_width * t_cu_tape
    A_ybco_total = A_hast + A_cu_tape + A_shunt

    # 总长度与网格
    L_total = L_cu + L_ybco
    x = np.linspace(0.0, L_total, N)
    dx = x[1] - x[0]

    # 接头热（分布到 [L_cu-L_joint, L_cu]，完全在铜侧）
    Q_joint = I**2 * R_joint
    q_joint_per_m = Q_joint / L_joint
    x1 = L_cu - L_joint
    x2 = L_cu

    # 分段函数
    def A_func(x_pos):
        x_pos = np.asarray(x_pos)
        return np.where(x_pos <= L_cu, A_cu, A_ybco_total)

    def k_eq_ybco(T):
        T = np.asarray(T)
        numerator = k_hastelloy(T)*A_hast + k_cu_rrr50(T)*A_cu_tape + k_ss304(T)*A_shunt
        return numerator / A_ybco_total

    def k_func(T, x_pos):
        T = np.asarray(T)
        x_pos = np.asarray(x_pos)
        return np.where(x_pos <= L_cu, k_cu_rrr50(T), k_eq_ybco(T))

    def rho_func(T, x_pos):
        T = np.asarray(T)
        x_pos = np.asarray(x_pos)
        return np.where(x_pos <= L_cu, rho_cu_rrr50(T), 0.0)

    def qv_func(T, x_pos):
        A_local = A_func(x_pos)
        J = I / A_local
        return (J**2) * rho_func(T, x_pos)

    # 解温度场
    def solve_temperature(max_iter=300, tol=1e-6):
        T = np.linspace(T_H, T_C, N)

        for _ in range(max_iter):
            T_old = T.copy()

            k_nodes = k_func(T, x)
            A_nodes = A_func(x)
            kA_nodes = k_nodes * A_nodes
            kA_face = 0.5 * (kA_nodes[:-1] + kA_nodes[1:])  # N-1

            # 节点线热源（W/m）
            qprime = qv_func(T, x) * A_nodes

            # 接头区间叠加线热源（W/m）
            mask_joint = (x >= x1) & (x <= x2) & (x > 0) & (x < L_total)
            qprime = qprime.copy()
            qprime[mask_joint] += q_joint_per_m

            # 组装三对角
            a = np.zeros(N)
            b = np.zeros(N)
            c = np.zeros(N)
            rhs = np.zeros(N)

            # Dirichlet
            b[0] = 1.0;  rhs[0] = T_H
            b[-1] = 1.0; rhs[-1] = T_C

            for i in range(1, N-1):
                kA_w = kA_face[i-1]
                kA_e = kA_face[i]
                a[i] =  kA_w / dx**2
                c[i] =  kA_e / dx**2
                b[i] = -(kA_w + kA_e) / dx**2
                rhs[i] = -qprime[i]

            T = solve_tridiag(a, b, c, rhs)

            if np.max(np.abs(T - T_old)) < tol:
                break

        return T

    T_sol = solve_temperature()

    # 热流（用 face 统一定义）
    A_vals = A_func(x)
    k_nodes = k_func(T_sol, x)
    kA_nodes = k_nodes * A_vals
    kA_face = 0.5 * (kA_nodes[:-1] + kA_nodes[1:])
    Q_face = -kA_face * (T_sol[1:] - T_sol[:-1]) / dx
    Q_hot = Q_face[0]
    Q_cold = Q_face[-1]

    # 总焦耳热
    qv_vals = qv_func(T_sol, x)
    qvA_vals = qv_vals * A_vals
    Q_joule_cu = np.trapezoid(qvA_vals, x)
    Q_joule_total = Q_joule_cu + Q_joint

    # 占比图数据（YBCO段）
    x_half = 0.5*(x[1:] + x[:-1])
    T_half = 0.5*(T_sol[1:] + T_sol[:-1])
    mask_y = x_half > L_cu
    xh = x_half[mask_y]
    Th = T_half[mask_y]

    G_h = k_hastelloy(Th) * A_hast
    G_c = k_cu_rrr50(Th)  * A_cu_tape
    G_s = k_ss304(Th)     * A_shunt
    G_sum = G_h + G_c + G_s
    f_h = G_h / G_sum
    f_c = G_c / G_sum
    f_s = G_s / G_sum

    # ==========================
    # 图1：温度（统一排版参数，保证两图视觉一致）
    # ==========================
    fig1, ax1 = plt.subplots(figsize=(7.5, 4.2), dpi=150, constrained_layout=False)
    ax1.plot(x, T_sol, linewidth=2)
    ax1.axvline(L_cu, linestyle="--", color="gray", label="Cu–YBCO 接头")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("T (K)")
    ax1.set_title("沿程温度分布 T(x)")
    ax1.grid(True)

    # legend 固定位置/大小，避免挤压绘图区导致两张图比例不一致
    ax1.legend(loc="upper right", frameon=True)

    # 关键：两张图用同一组边距（强制一致）
    fig1.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.18)

    # ==========================
    # 图2：占比（同样的边距 + legend 固定）
    # ==========================
    fig2, ax2 = plt.subplots(figsize=(7.5, 4.2), dpi=150, constrained_layout=False)
    ax2.plot(xh, f_h, label="Hastelloy")
    ax2.plot(xh, f_c, label="Cu in tape")
    ax2.plot(xh, f_s, label="SS304 shunt")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("Heat conduction fraction")
    ax2.set_ylim(0, 1)
    ax2.set_title("YBCO 段各材料导热占比（沿长度）")
    ax2.grid(True)

    ax2.legend(loc="upper left", frameon=True)

    # 同一组边距（强制一致）
    fig2.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.18)

    # ===== 固定像素导出（保证两张图最终显示大小完全一致）=====
    def fig_to_png_bytes(fig, w_px=1200, h_px=650, dpi=150):
        buf = io.BytesIO()
        fig.set_dpi(dpi)
        fig.set_size_inches(w_px / dpi, h_px / dpi)  # 强制同宽同高
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches=None, pad_inches=0.0)
        buf.seek(0)
        return buf.getvalue()

    img_T = fig_to_png_bytes(fig1)
    img_frac = fig_to_png_bytes(fig2)

    # 释放 matplotlib 资源，避免越跑越慢/串台
    plt.close(fig1)
    plt.close(fig2)

    return {
        "x": x,
        "T_sol": T_sol,
        "Q_hot": Q_hot,
        "Q_cold": Q_cold,
        "Q_joint": Q_joint,
        "Q_joule_total": Q_joule_total,
        "img_T": img_T,
        "img_frac": img_frac,
    }


# ==========================
# Streamlit UI
# ==========================
st.set_page_config(page_title="Cu–YBCO 热分析", layout="wide")
st.title("Cu–YBCO 引线稳态导热 + 焦耳热（含接头电阻 & 带材并联数）")
st.write("BUILD: 2025-12-29-01")


with st.sidebar:
    st.header("输入参数")

    # 边界条件
    st.subheader("边界条件")
    T_H = st.number_input("热端温度 T_H (K)", value=100.0, step=1.0)
    T_C = st.number_input("冷端温度 T_C (K)", value=7.0, step=1.0)
    I = st.number_input("电流 I (A)", value=240.0, step=10.0)

    st.divider()

    # 铜段
    st.subheader("铜段（Cu）")
    A_cu_mm2 = st.number_input("铜截面积 A_cu (mm²)", value=50.0, step=1.0)
    L_cu = st.number_input("铜长度 L_cu (m)", value=0.14, step=0.01, format="%.3f")

    st.divider()

    # YBCO段
    st.subheader("YBCO 段")
    L_ybco = st.number_input("YBCO 段长度 L_ybco (m)", value=0.10, step=0.01, format="%.3f")
    n_ybco = st.number_input("带材并联数量 n_ybco (整数)", value=3, step=1, min_value=1)
    A_shunt_mm2 = st.number_input("分流器截面积 A_shunt (mm²)", value=20.8, step=0.1, format="%.2f")

    st.divider()

    # 接头
    st.subheader("接头（Cu–YBCO）")
    R_joint_uohm = st.number_input("接头电阻 R_joint (μΩ)", value=39.11, step=0.1, format="%.3f")
    L_joint_cm = st.number_input("接头长度 L_joint (cm)", value=2.0, step=0.1, format="%.2f")

    st.divider()

    # 数值
    st.subheader("数值设置")
    N = st.slider("网格点数 N", min_value=201, max_value=2001, value=501, step=100)

    run_btn = st.button("开始计算", type="primary")

# 单位换算
A_cu = A_cu_mm2 * 1e-6
A_shunt = A_shunt_mm2 * 1e-6
R_joint = R_joint_uohm * 1e-6
L_joint = L_joint_cm / 100.0

if run_btn:
    # 简单输入检查：接头长度不能超过铜长度
    if L_joint >= L_cu:
        st.error("接头长度 L_joint 不能大于或等于铜长度 L_cu（否则接头区间会跑到铜段左边界以外）。")
    else:
        with st.spinner("计算中..."):
            out = run_model(
                T_H=T_H, T_C=T_C, I=I,
                A_cu=A_cu, L_cu=L_cu,
                L_ybco=L_ybco, A_shunt=A_shunt, n_ybco=int(n_ybco),
                R_joint=R_joint, L_joint=L_joint,
                N=N,
            )

        # ==========================
        # 页面展示：统一宽度（关键：不要把图1放在列里）
        # ==========================

        # 结果区（更紧凑：两列metric）
        with st.container(border=True):
            st.subheader("计算结果")
            m1, m2 = st.columns(2)
            with m1:
                st.metric("冷端漏热 Q_cold (W)", f"{out['Q_cold']:.6f}")
                st.metric("总焦耳热 Q_joule_total (W)", f"{out['Q_joule_total']:.6f}")
            with m2:
                st.metric("接头焦耳热 Q_joint (W)", f"{out['Q_joint']:.6f}")
                st.metric("热端热流 Q_hot (W)", f"{out['Q_hot']:.6f}")

        # 图1（全宽）
        with st.container(border=True):
            st.subheader("图 1：沿程温度分布 T(x)")
            st.image(out["img_T"], use_container_width=True)

        # 图2（全宽）
        with st.container(border=True):
            st.subheader("图 2：YBCO 段材料导热占比（沿长度）")
            st.image(out["img_frac"], use_container_width=True)

        st.caption("输入单位：A_cu/A_shunt 用 mm²，R_joint 用 μΩ，L_joint 用 cm。内部自动换算为 SI 单位。")

else:
    st.info("在左侧输入参数，然后点 **开始计算**。")
