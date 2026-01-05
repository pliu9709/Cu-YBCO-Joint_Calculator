import os
import io
import datetime as dt
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import streamlit as st

# ===== PDF (ReportLab) =====
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ================= 登录配置 =================
USERNAME = "GWCDLHGCZX"
PASSWORD = "GWCDLHGCZX"

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False


def login_page():
    st.set_page_config(page_title="登录", layout="centered")

    st.markdown("## 🔐 系统登录")

    username = st.text_input("用户名")
    password = st.text_input("密码", type="password")

    if st.button("登录"):
        if username == USERNAME and password == PASSWORD:
            st.session_state.authenticated = True
            st.success("登录成功，正在进入系统…")
            st.rerun()
        else:
            st.error("用户名或密码错误")


# 如果未登录 → 只显示登录页
if not st.session_state.authenticated:
    login_page()
    st.stop()
# =========================================================
# 0) 基础配置：页面 + 字体
# =========================================================
st.set_page_config(
    page_title="Cu–YBCO Lead Thermal Analysis",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT_DIR = os.path.dirname(__file__)
FONT_PATH = os.path.join(ROOT_DIR, "fonts", "NotoSansSC-Regular.ttf")

# 示意图文件：请将图片命名为 schematic.png，并放到项目目录 assets/ 下
# 例如：
#   your_app/
#     app_joint_check_tabbed.py
#     assets/
#       schematic.png
SCHEMATIC_PATH = os.path.join(ROOT_DIR, "assets", "schematic.png")

# Matplotlib 中文
if os.path.exists(FONT_PATH):
    try:
        font_manager.fontManager.addfont(FONT_PATH)
        font_name = font_manager.FontProperties(fname=FONT_PATH).get_name()
        plt.rcParams["font.family"] = font_name
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False


def register_pdf_font() -> str:
    """ReportLab 中文字体注册"""
    if os.path.exists(FONT_PATH):
        try:
            pdfmetrics.getFont("NotoSansSC")
        except KeyError:
            pdfmetrics.registerFont(TTFont("NotoSansSC", FONT_PATH))
        return "NotoSansSC"
    return "Helvetica"


PDF_FONT = register_pdf_font()


# =========================================================
# 1) UI 美化：CSS（产品化、卡片阴影、去掉描边红框感）
# =========================================================
def inject_css():
    st.markdown(
        """
<style>
:root{
  --bg1: rgba(255,255,255,0.75);
  --txt0: rgba(17, 17, 17, 0.96);
  --txt1: rgba(17, 17, 17, 0.65);
  --bd0: rgba(0,0,0,0.08);
  --shadow2: 0 6px 18px rgba(0,0,0,0.06);
  --r16: 16px;
  --r20: 20px;
}
.block-container { padding-top: 1.0rem; padding-bottom: 2.0rem; }
*:focus { outline: none !important; box-shadow: none !important; }
button:focus, button:active, [role="button"]:focus { outline: none !important; box-shadow: none !important; }

section[data-testid="stSidebar"] { border-right: 1px solid var(--bd0); }
section[data-testid="stSidebar"] .block-container{ padding-top: 1.0rem; }

.hero {
  border: 1px solid var(--bd0);
  background: linear-gradient(135deg, rgba(240,248,255,0.85), rgba(255,255,255,0.75));
  border-radius: var(--r20);
  padding: 18px 18px 14px 18px;
  box-shadow: var(--shadow2);
}
.hero-title{
  font-size: 30px;
  font-weight: 900;
  letter-spacing: 0.3px;
  margin: 0;
  color: var(--txt0);
}
.hero-sub{
  margin-top: 6px;
  color: var(--txt1);
  font-size: 13px;
  line-height: 1.45;
}
.card{
  border: 1px solid var(--bd0);
  background: var(--bg1);
  border-radius: var(--r20);
  padding: 14px 16px 12px 16px;
  box-shadow: var(--shadow2);
}
.hint{
  color: var(--txt1);
  font-size: 13px;
}
button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] p{ font-weight: 700; }
div[data-testid="stMetricValue"]{ font-size: 22px; }
div[data-testid="stMetricLabel"]{ color: rgba(0,0,0,0.65); }
hr{ border-color: rgba(0,0,0,0.06); }
.stDownloadButton button, .stButton button{ border-radius: 14px !important; }
.stButton button[kind="primary"]{ font-weight: 800; }

.badge{
  display:inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  background: rgba(0, 128, 0, 0.08);
  color: rgba(0, 90, 0, 0.8);
  font-size: 12px;
  font-weight: 800;
}
.badge-idle{
  background: rgba(0,0,0,0.05);
  color: rgba(0,0,0,0.65);
}

/* ✅用 Streamlit 原生 container(border=True) 做卡片（避免空白透明bar） */
div[data-testid="stVerticalBlockBorderWrapper"]{
  border-radius: var(--r20) !important;
  border: 1px solid var(--bd0) !important;
  background: var(--bg1) !important;
  box-shadow: var(--shadow2) !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div{
  padding: 14px 16px 12px 16px !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )


inject_css()

# =========================================================
# 2) 双语文案（关键：别写反）
# =========================================================
LANG_OPTIONS = ["中文", "English"]

if "lang" not in st.session_state:
    st.session_state.lang = "中文"


def is_cn() -> bool:
    return st.session_state.lang == "中文"


TEXT = {
    # General
    "language": {"cn": "语言 / Language", "en": "Language / 语言"},
    "not_run": {"cn": "未运行", "en": "Not run"},
    "ran": {"cn": "已运行", "en": "Ran"},
    "calculating": {"cn": "计算中...", "en": "Computing..."},
    "done": {"cn": "计算完成 ✅", "en": "Done ✅"},
    "run": {"cn": "🚀 开始计算", "en": "🚀 Run"},
    "params": {"cn": "参数输入", "en": "Inputs"},
    "units_note_sidebar": {
        "cn": "单位输入为工程常用单位，内部自动换算 SI。",
        "en": "Inputs use engineering units; internally converted to SI.",
    },
    # Hero
    "hero_title": {
        "cn": "Cu–YBCO 引线稳态导热 + 焦耳热分析",
        "en": "Cu–YBCO Lead: Steady Conduction + Joule Heating",
    },
    "hero_sub": {
        "cn": "一维稳态导热 + 焦耳热（含接头电阻），铜段与 YBCO 段串联，采用有限差分法，基于Picard+三对角矩阵求解",
        "en": "1D steady heat conduction with Joule heating (incl. joint resistance). Cu segment in series with YBCO segment.",
    },
    # Tabs
    "tab_overview": {"cn": "📌 总览", "en": "📌 Overview"},
    "tab_plots": {"cn": "📈 图表", "en": "📈 Plots"},
    "tab_report": {"cn": "📄 报告导出", "en": "📄 Report"},
    "tab_history": {"cn": "🧾 历史", "en": "🧾 History"},
    "tab_about": {"cn": "ℹ️ 说明", "en": "ℹ️ About"},
    # Overview
    "key_results": {"cn": "关键结果", "en": "Key Results"},
    "overview_tip": {
        "cn": "先在左侧输入参数并点击「开始计算」，结果会保留在页面中。",
        "en": "Set inputs on the left and click “Run”. Results will stay on the page.",
    },
    "no_result": {
        "cn": "尚未计算。请先点击左侧 **🚀 开始计算**。",
        "en": "Not computed yet. Please click **🚀 Run** on the left.",
    },
    "units_box_title": {"cn": "单位说明", "en": "Units"},
    "units_box": {
        "cn": "A_cu / A_shunt：mm²<br/>R_joint：μΩ<br/>L_joint：cm<br/>内部自动换算 SI。",
        "en": "A_cu / A_shunt: mm²<br/>R_joint: μΩ<br/>L_joint: cm<br/>Converted to SI internally.",
    },
    # Sidebar groups
    "bc": {"cn": "边界条件", "en": "Boundary Conditions"},
    "cu": {"cn": "铜段（Cu）", "en": "Cu Segment"},
    "ybco": {"cn": "YBCO 段", "en": "YBCO Segment"},
    "joint": {"cn": "接头（Cu–YBCO）", "en": "Joint (Cu–YBCO)"},
    "num": {"cn": "一维节点数", "en": "Numerics"},
    "num_tip": {"cn": "建议：401~1201 比较稳。更高 N 会更慢，但曲线更平滑。",
                "en": "Suggestion: 401–1201 is robust. Larger N is slower but smoother."},
    # Plots
    "plots_title": {"cn": "图表", "en": "Plots"},
    "plots_no": {
        "cn": "尚未计算，暂无图表。请先在左侧点击 **🚀 开始计算**。",
        "en": "No plots yet. Please click **🚀 Run** on the left.",
    },
    "fig1": {"cn": "图 1：沿程温度分布 T(x)", "en": "Figure 1: Temperature profile T(x)"},
    "fig2": {"cn": "图 2：YBCO 段材料导热占比（沿长度）", "en": "Figure 2: Conduction fraction in YBCO segment"},
    "explain_hint": {"cn": "🔎 参数含义", "en": "🔎 Parameter meaning "},
    # Report
    "report_title": {"cn": "导出 PDF 计算报告", "en": "Export PDF Report"},
    "report_no": {"cn": "请先完成一次计算，然后这里会出现「导出 PDF」按钮。",
                  "en": "Run once first; then the “Export PDF” button will appear here."},
    "report_btn": {"cn": "📄 导出 PDF 报告", "en": "📄 Export PDF"},
    "report_note": {"cn": "报告仅作参考", "en": "For reference only"},
    "font_warn": {"cn": "未找到 fonts/NotoSansSC-Regular.ttf，PDF 中文可能无法显示。请检查项目目录。",
                  "en": "fonts/NotoSansSC-Regular.ttf not found. Chinese may not render in PDF."},
    # History
    "hist_title": {"cn": "运行历史（对比/导出）", "en": "Run History (Compare/Export)"},
    "hist_no": {"cn": "还没有历史记录。每次点击「开始计算」都会自动记录到这里。",
                "en": "No history yet. Each run will be recorded here."},
    "hist_tip": {"cn": "提示：你可以在外部用 Excel / Python 对这张表做更复杂的对比与统计。",
                 "en": "Tip: Use Excel/Python for deeper comparison and statistics."},
    "export_all": {"cn": "⬇️ 导出历史 CSV", "en": "⬇️ Export history CSV"},
    "export_key": {"cn": "⬇️ 仅导出关键结果 CSV", "en": "⬇️ Export key results CSV"},
    "clear_hist": {"cn": "🧹 清空历史", "en": "🧹 Clear history"},
    "hist_cleared": {"cn": "历史已清空", "en": "History cleared"},
    # About
    "about_title": {"cn": "说明 / 模型 / 版本信息", "en": "About / Model / Version"},
    "materials": {"cn": "物性参数", "en": "Materials"},
    "tape": {"cn": "YBCO 带材", "en": "YBCO Tape"},
    # Footer
    "copyright": {"cn": "© 版权所有", "en": "© Copyright"},
    "center": {"cn": "高温超导联合工程中心@广东东莞松山湖",
               "en": "HTS Joint Engineering Center @ Songshan Lake, Dongguan, Guangdong"},
}


def t(key: str) -> str:
    return TEXT[key]["cn"] if is_cn() else TEXT[key]["en"]


# =========================================================
# 3) 数据结构
# =========================================================
@dataclass
class ModelInputs:
    T_H: float
    T_C: float
    I: float
    A_cu_mm2: float
    L_cu: float
    L_ybco: float
    n_ybco: int
    A_shunt_mm2: float
    R_joint_uohm: float
    L_joint_cm: float
    N: int

    def to_si(self) -> Dict[str, float]:
        return {
            "T_H": self.T_H,
            "T_C": self.T_C,
            "I": self.I,
            "A_cu": self.A_cu_mm2 * 1e-6,
            "L_cu": self.L_cu,
            "L_ybco": self.L_ybco,
            "n_ybco": int(self.n_ybco),
            "A_shunt": self.A_shunt_mm2 * 1e-6,
            "R_joint": self.R_joint_uohm * 1e-6,
            "L_joint": self.L_joint_cm / 100.0,
            "N": int(self.N),
        }

    def to_display_dict(self) -> Dict[str, Any]:
        # 显示 key 统一用英文变量名，避免双语切换导致表头变化
        return {
            "T_H (K)": self.T_H,
            "T_C (K)": self.T_C,
            "I (A)": self.I,
            "A_cu (mm²)": self.A_cu_mm2,
            "L_cu (m)": self.L_cu,
            "L_ybco (m)": self.L_ybco,
            "n_ybco": int(self.n_ybco),
            "A_shunt (mm²)": self.A_shunt_mm2,
            "R_joint (μΩ)": self.R_joint_uohm,
            "L_joint (cm)": self.L_joint_cm,
            "N": int(self.N),
        }


# =========================================================
# 4) 物性函数（原逻辑）
# =========================================================
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
    num = a + c * T_clamp ** 0.5 + e * T_clamp + g * T_clamp ** 1.5 + i * T_clamp ** 2
    den = 1.0 + b * T_clamp ** 0.5 + d * T_clamp + f * T_clamp ** 1.5 + h * T_clamp ** 2
    log10k = num / den
    return 10.0 ** log10k


def k_hastelloy(T):
    T = np.asarray(T)
    T_clamp = np.clip(T, T_MIN, T_MAX)
    a = 0.58856
    b = 0.23494
    c = -0.00292
    d = 1.679e-5
    e = -3.432e-8
    return (a + b * T_clamp + c * T_clamp ** 2 + d * T_clamp ** 3 + e * T_clamp ** 4) * 0.85


def k_ss304(T):
    ss_coeffs = {
        "a": -1.4087, "b": 1.3982, "c": 0.2543, "d": 0.02406, "e": 0.0,
        "f": 0.4256, "g": -0.4858, "h": 0.1650, "i": -0.01159,
    }
    T = np.asarray(T)
    T_clamp = np.clip(T, T_MIN, T_MAX)
    x_log = np.log10(T_clamp)
    a, b, c, d, e, f, g, h, i = ss_coeffs.values()
    log10k = (
            a + b * x_log + c * x_log ** 2 + d * x_log ** 3 + e * x_log ** 4
            + f * x_log ** 5 + g * x_log ** 6 + h * x_log ** 7 + i * x_log ** 8
    )
    return 10.0 ** log10k


def rho_cu_rrr50(T):
    T = np.asarray(T)
    T_clamp = np.clip(T, T_MIN, T_MAX)
    rho = (
                  0.06948
                  - 0.00434 * T_clamp
                  + 1.17e-4 * T_clamp ** 2
                  - 5.135e-7 * T_clamp ** 3
                  + 7.55179e-10 * T_clamp ** 4
          ) * 1e-8
    return rho


# =========================================================
# 5) 数值：三对角 Thomas
# =========================================================
def solve_tridiag(a, b, c, d):
    a = a.astype(float).copy()
    b = b.astype(float).copy()
    c = c.astype(float).copy()
    d = d.astype(float).copy()
    n = len(b)

    for i in range(1, n):
        w = a[i] / b[i - 1]
        b[i] -= w * c[i - 1]
        d[i] -= w * d[i - 1]

    xsol = np.zeros_like(d)
    xsol[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        xsol[i] = (d[i] - c[i] * xsol[i + 1]) / b[i]
    return xsol


# =========================================================
# 6) 计算 + 出图
# =========================================================
def fig_to_png_bytes(fig, w_px=1500, h_px=820, dpi=160) -> bytes:
    buf = io.BytesIO()
    fig.set_dpi(dpi)
    fig.set_size_inches(w_px / dpi, h_px / dpi)
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches=None, pad_inches=0.0)
    buf.seek(0)
    return buf.getvalue()


def run_model(
        T_H: float, T_C: float, I: float,
        A_cu: float, L_cu: float,
        L_ybco: float, A_shunt: float, n_ybco: int,
        R_joint: float, L_joint: float,
        N: int = 501,
) -> Dict[str, Any]:
    tape_width = 4.0e-3
    t_hast_single = 45e-6
    t_cu_single = 16e-6

    t_hast = t_hast_single * n_ybco
    t_cu_tape = t_cu_single * n_ybco

    A_hast = tape_width * t_hast
    A_cu_tape = tape_width * t_cu_tape
    A_ybco_total = A_hast + A_cu_tape + A_shunt

    L_total = L_cu + L_ybco
    x = np.linspace(0.0, L_total, N)
    dx = x[1] - x[0]

    Q_joint = I ** 2 * R_joint
    q_joint_per_m = Q_joint / L_joint
    x1 = L_cu - L_joint
    x2 = L_cu

    def A_func(x_pos):
        x_pos = np.asarray(x_pos)
        return np.where(x_pos <= L_cu, A_cu, A_ybco_total)

    def k_eq_ybco(T):
        T = np.asarray(T)
        numerator = k_hastelloy(T) * A_hast + k_cu_rrr50(T) * A_cu_tape + k_ss304(T) * A_shunt
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
        return (J ** 2) * rho_func(T, x_pos)

    def solve_temperature(max_iter=350, tol=1e-6):
        T = np.linspace(T_H, T_C, N)
        for _ in range(max_iter):
            T_old = T.copy()

            k_nodes = k_func(T, x)
            A_nodes = A_func(x)
            kA_nodes = k_nodes * A_nodes
            kA_face = 0.5 * (kA_nodes[:-1] + kA_nodes[1:])

            qprime = qv_func(T, x) * A_nodes

            mask_joint = (x >= x1) & (x <= x2) & (x > 0) & (x < L_total)
            qprime = qprime.copy()
            qprime[mask_joint] += q_joint_per_m

            a = np.zeros(N)
            b = np.zeros(N)
            c = np.zeros(N)
            rhs = np.zeros(N)

            b[0] = 1.0
            rhs[0] = T_H
            b[-1] = 1.0
            rhs[-1] = T_C

            for i in range(1, N - 1):
                kA_w = kA_face[i - 1]
                kA_e = kA_face[i]
                a[i] = kA_w / dx ** 2
                c[i] = kA_e / dx ** 2
                b[i] = -(kA_w + kA_e) / dx ** 2
                rhs[i] = -qprime[i]

            T = solve_tridiag(a, b, c, rhs)
            if np.max(np.abs(T - T_old)) < tol:
                break
        return T

    T_sol = solve_temperature()

    A_vals = A_func(x)
    k_nodes = k_func(T_sol, x)
    kA_nodes = k_nodes * A_vals
    kA_face = 0.5 * (kA_nodes[:-1] + kA_nodes[1:])
    Q_face = -kA_face * (T_sol[1:] - T_sol[:-1]) / dx
    Q_hot = Q_face[0]
    Q_cold = Q_face[-1]

    qv_vals = qv_func(T_sol, x)
    qvA_vals = qv_vals * A_vals
    Q_joule_cu = np.trapezoid(qvA_vals, x)
    Q_joule_total = Q_joule_cu + Q_joint

    x_half = 0.5 * (x[1:] + x[:-1])
    T_half = 0.5 * (T_sol[1:] + T_sol[:-1])
    mask_y = x_half > L_cu
    xh = x_half[mask_y]
    Th = T_half[mask_y]

    G_h = k_hastelloy(Th) * A_hast
    G_c = k_cu_rrr50(Th) * A_cu_tape
    G_s = k_ss304(Th) * A_shunt
    G_sum = G_h + G_c + G_s
    f_h = G_h / G_sum
    f_c = G_c / G_sum
    f_s = G_s / G_sum

    fig1, ax1 = plt.subplots(figsize=(7.8, 4.4), dpi=160, constrained_layout=False)
    ax1.plot(x, T_sol, linewidth=2.2)
    ax1.axvline(L_cu, linestyle="--", color="gray", linewidth=1.2, label="Cu–YBCO joint")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("T (K)")
    ax1.set_title("Temperature profile T(x)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper right", frameon=True)
    fig1.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.18)

    fig2, ax2 = plt.subplots(figsize=(7.8, 4.4), dpi=160, constrained_layout=False)
    ax2.plot(xh, f_h, label="Hastelloy", linewidth=2.0)
    ax2.plot(xh, f_c, label="Cu in tape", linewidth=2.0)
    ax2.plot(xh, f_s, label="SS304 shunt", linewidth=2.0)
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("Heat conduction fraction")
    ax2.set_ylim(0, 1)
    ax2.set_title("Conduction fraction in YBCO segment")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper left", frameon=True)
    fig2.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.18)

    img_T = fig_to_png_bytes(fig1)
    img_frac = fig_to_png_bytes(fig2)
    plt.close(fig1)
    plt.close(fig2)

    # Joint temperature at Cu–YBCO interface (x = L_cu)
    idx_joint = int(np.argmin(np.abs(x - L_cu)))
    T_joint = float(T_sol[idx_joint])

    return {
        "x": x,
        "T_sol": T_sol,
        "Q_hot": float(Q_hot),
        "Q_cold": float(Q_cold),
        "Q_joint": float(Q_joint),
        "Q_joule_total": float(Q_joule_total),
        "img_T": img_T,
        "img_frac": img_frac,
        "T_joint": float(T_joint),
    }


# =========================================================
# 7) PDF 报告（双语标题/小节跟随语言）
# =========================================================
def build_pdf_report(
        title: str,
        meta: dict,
        inputs: dict,
        results: dict,
        img_T: bytes,
        img_frac: bytes,
) -> bytes:
    buf = io.BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=A4)
    W, H = A4

    margin_l = 16 * mm
    margin_r = 16 * mm
    y = H - 18 * mm
    line_h = 6.0 * mm

    def hr(y_pos):
        c.setStrokeColor(colors.lightgrey)
        c.setLineWidth(0.6)
        c.line(margin_l, y_pos, W - margin_r, y_pos)

    def txt(x, y_pos, s, size=10, color=colors.black):
        c.setFont(PDF_FONT, size)
        c.setFillColor(color)
        c.drawString(x, y_pos, str(s))

    txt(margin_l, y, title, size=16, color=colors.black)
    y -= 8 * mm
    txt(margin_l, y, f"{meta.get('generated_at', '')}", size=10, color=colors.grey)
    y -= 6 * mm
    hr(y)
    y -= 8 * mm

    txt(margin_l, y, "1. Inputs / 输入参数", size=12)
    y -= 8 * mm

    col_w = (W - margin_l - margin_r) / 2
    left_x = margin_l
    right_x = margin_l + col_w

    items = list(inputs.items())
    for i in range(0, len(items), 2):
        k1, v1 = items[i]
        k2, v2 = items[i + 1] if i + 1 < len(items) else ("", "")
        txt(left_x, y, f"{k1}: {v1}", size=10)
        if k2:
            txt(right_x, y, f"{k2}: {v2}", size=10)
        y -= line_h
        if y < 25 * mm:
            c.showPage()
            y = H - 18 * mm

    y -= 2 * mm
    hr(y)
    y -= 8 * mm

    txt(margin_l, y, "2. Results / 计算结果", size=12)
    y -= 8 * mm
    for k, v in results.items():
        txt(margin_l, y, f"{k}: {v}", size=11)
        y -= 1.1 * line_h
        if y < 25 * mm:
            c.showPage()
            y = H - 18 * mm

    y -= 2 * mm
    hr(y)
    y -= 8 * mm

    def draw_image_block(title_txt: str, img_bytes: bytes):
        nonlocal y
        txt(margin_l, y, title_txt, size=12)
        y -= 6 * mm

        img = ImageReader(io.BytesIO(img_bytes))
        iw, ih = img.getSize()

        max_w = W - margin_l - margin_r
        max_h = 92 * mm
        scale = min(max_w / iw, max_h / ih)
        draw_w = iw * scale
        draw_h = ih * scale

        if y - draw_h < 20 * mm:
            c.showPage()
            y = H - 18 * mm

        c.drawImage(
            img, margin_l, y - draw_h,
            width=draw_w, height=draw_h,
            preserveAspectRatio=True, mask='auto'
        )
        y -= (draw_h + 10 * mm)

    draw_image_block("3. Figure 1 / 图 1: Temperature profile", img_T)
    draw_image_block("4. Figure 2 / 图 2: Conduction fraction", img_frac)

    c.setFont(PDF_FONT, 9)
    c.setFillColor(colors.grey)
    c.drawString(margin_l, 10 * mm, "Generated by Cu–YBCO Thermal Analysis Tool")
    c.save()

    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes


# =========================================================
# 8) Session State：结果 + 历史记录
# =========================================================
if "out" not in st.session_state:
    st.session_state.out = None
if "last_inputs" not in st.session_state:
    st.session_state.last_inputs = None
if "history" not in st.session_state:
    st.session_state.history = []
if "run_count" not in st.session_state:
    st.session_state.run_count = 0


def append_history(inputs: ModelInputs, out: Dict[str, Any]):
    st.session_state.run_count += 1
    rec = {
        "run_id": st.session_state.run_count,
        "time": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **inputs.to_display_dict(),
        "Q_cold (W)": out["Q_cold"],
        "Q_hot (W)": out["Q_hot"],
        "Q_joint (W)": out["Q_joint"],
        "Q_joule_total (W)": out["Q_joule_total"],
        "T_joint (K)": out.get("T_joint", np.nan),
    }
    st.session_state.history.insert(0, rec)


def history_df() -> pd.DataFrame:
    if not st.session_state.history:
        return pd.DataFrame()
    return pd.DataFrame(st.session_state.history)


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def validate_inputs(inp: ModelInputs) -> Tuple[bool, List[str], List[str]]:
    errors, warns = [], []
    if inp.T_H <= inp.T_C:
        errors.append("T_H must be greater than T_C / 热端温度 T_H 必须大于冷端温度 T_C。")
    if inp.A_cu_mm2 <= 0 or inp.A_shunt_mm2 <= 0:
        errors.append("Areas must be positive / 截面积必须为正数。")
    if inp.L_joint_cm <= 0:
        errors.append("L_joint must be positive / 接头长度必须为正数。")
    if inp.L_cu <= 0 or inp.L_ybco <= 0:
        errors.append("Lengths must be positive / 长度必须为正数。")
    if inp.L_joint_cm / 100.0 >= inp.L_cu:
        errors.append("L_joint must be smaller than L_cu / 接头长度不能大于或等于铜长度 L_cu。")
    if inp.n_ybco < 1:
        errors.append("n_ybco >= 1 / 带材并联数量至少为 1。")
    if inp.N < 201:
        warns.append("N may be too low / 网格点数偏低可能导致误差。")
    if inp.I >= 500:
        warns.append("Large current: Joule heating may be significant / 电流较大，焦耳热可能显著增大。")
    return (len(errors) == 0), errors, warns


# =========================================================
# 9) Sidebar：仅保留“语言/导航”属性（输入放到「总览」Tab）
# =========================================================
with st.sidebar:
    st.session_state.lang = st.selectbox(t("language"), LANG_OPTIONS, index=0 if is_cn() else 1)
    st.caption(t("units_note_sidebar"))
    st.divider()
    st.caption(" 版本信息V1")

# =========================================================
# 10) 顶部 Header（跟随语言）
# =========================================================
status_badge = f'<span class="badge badge-idle">{t("not_run")}</span>'
if st.session_state.out is not None:
    status_badge = f'<span class="badge">{t("ran")}</span>'

st.markdown(
    f"""
<div class="hero">
  <div class="hero-title">{t("hero_title")}</div>
  <div class="hero-sub">{t("hero_sub")} {status_badge}</div>
</div>
""",
    unsafe_allow_html=True,
)

# =========================================================
# 12) Tabs
# =========================================================
tab_overview, tab_plots, tab_report, tab_history, tab_about = st.tabs(
    [t("tab_overview"), t("tab_plots"), t("tab_report"), t("tab_history"), t("tab_about")]
)

out = st.session_state.out


def nice_number(x: float) -> str:
    x = float(x)
    ax = abs(x)
    if ax == 0:
        return "0"
    if ax < 1e-3 or ax >= 1e4:
        return f"{x:.6e}"
    return f"{x:.6f}"


def joint_temp_assessment(T_joint: float) -> Tuple[str, str]:
    """Return (level, message) where level in {'error','warning','success'}."""
    tj = float(T_joint)
    if is_cn():
        if tj > 75.0:
            return "error", f"接头温度为 {tj:.2f} K，高于 75 K，请重新进行设计。"
        if tj >= 65.0:
            return "warning", f"接头温度为 {tj:.2f} K，YBCO 存在失超风险，请谨慎使用该设计参数。"
        return "success", f"接头温度为 {tj:.2f} K，YBCO 运行状态良好。"
    else:
        if tj > 75.0:
            return "error", f"Joint temperature is {tj:.2f} K, higher than 75 K. Please redesign."
        if tj >= 65.0:
            return "warning", f"Joint temperature is {tj:.2f} K (65–75 K). Quench risk exists; use with caution."
        return "success", f"Joint temperature is {tj:.2f} K. YBCO operating condition looks good."


# ===== Overview =====
with tab_overview:
    # 页面布局：左侧输入（分子Tab，避免过长），右侧固定示意图+关键结果（输入时无需上翻对照）
    left, right = st.columns([3.5, 6.5], gap="large")

    # -------------------------
    # Left: Inputs (sub-tabs) + Run
    # -------------------------
    with left:
        with st.container(border=True):
            st.subheader(t("params"))
            st.caption(
                "注：A_HTS本身面积由带材决定，不做输入变量。"
                if is_cn() else
                "Inputs are split into sub-tabs to reduce scrolling; the schematic stays visible on the right for quick reference."
            )

            # 用 form：避免每改一个数就全页面重跑
            with st.form("inputs_form", clear_on_submit=False):
                t_bc, t_cu, t_hts, t_joint, t_num = st.tabs([
                    "边界条件" if is_cn() else "Boundary",
                    "Cu段" if is_cn() else "Cu",
                    "HTS段" if is_cn() else "HTS",
                    "接头区" if is_cn() else "Joint",
                    "一维计算节点数" if is_cn() else "Numerics",
                ])

                with t_bc:
                    T_H = st.number_input("T_H (K)", value=float(st.session_state.get("T_H", 50.0)), step=1.0,
                                          help="高温端边界温度" if is_cn() else "Hot-end boundary temperature")
                    T_C = st.number_input("T_L ", value=float(st.session_state.get("T_C", 7.0)), step=1.0,
                                          help="低温端边界温度" if is_cn() else "Cold-end boundary temperature")
                    I = st.number_input("I (A)", value=float(st.session_state.get("I", 240.0)), step=10.0,
                                        help="电流" if is_cn() else "Current")

                with t_cu:
                    A_cu_mm2 = st.number_input("A_Cu (mm²)", value=float(st.session_state.get("A_cu_mm2", 50.0)),
                                               step=1.0,
                                               help=(
                                                   "Cu段横截面积（见示意图 A_Cu）" if is_cn() else "Cu cross-sectional area (see A_Cu in schematic)"))
                    L_cu = st.number_input("L_Cu (m)", value=float(st.session_state.get("L_cu", 0.14)), step=0.01,
                                           format="%.3f",
                                           help=(
                                               "Cu段长度（见示意图 L_Cu）" if is_cn() else "Cu length (see L_Cu in schematic)"))

                with t_hts:
                    L_ybco = st.number_input("L_HTS / L_ybco (m)", value=float(st.session_state.get("L_ybco", 0.10)),
                                             step=0.01, format="%.3f",
                                             help=(
                                                 "HTS段长度（见示意图 L_HTS）" if is_cn() else "HTS length (see L_HTS in schematic)"))
                    n_ybco = st.number_input("n_ybco", value=int(st.session_state.get("n_ybco", 3)), step=1,
                                             min_value=1,
                                             help=("HTS带材并联条数" if is_cn() else "Number of HTS tapes in parallel"))
                    A_shunt_mm2 = st.number_input("A_Shunt (mm²)",
                                                  value=float(st.session_state.get("A_shunt_mm2", 20.8)), step=0.1,
                                                  format="%.2f",
                                                  help=(
                                                      "HTS段并联分流器截面积（见示意图 A_Shunt）" if is_cn() else "Shunt cross-sectional area in HTS segment (see A_Shunt)"))
                    st.caption(
                        "注：A_HTS 为固定常数，不作为输入变量。" if is_cn() else "Note: A_HTS is treated as a constant (not an input).")

                with t_joint:
                    R_joint_uohm = st.number_input("R_joint (μΩ)",
                                                   value=float(st.session_state.get("R_joint_uohm", 39.11)), step=0.1,
                                                   format="%.3f",
                                                   help=(
                                                       "搭接焊接区等效电阻（见示意图 R_joint）" if is_cn() else "Equivalent resistance of lap joint (see R_joint)"))
                    L_joint_cm = st.number_input("L_joint (cm)", value=float(st.session_state.get("L_joint_cm", 2.0)),
                                                 step=0.1, format="%.2f",
                                                 help=(
                                                     "搭接焊接长度（见示意图 L_joint）" if is_cn() else "Lap-joint overlap length (see L_joint)"))

                with t_num:
                    N = st.slider("N", min_value=201, max_value=2001, value=int(st.session_state.get("N", 501)),
                                  step=100,
                                  help=("有限差分网格点数" if is_cn() else "Finite-difference grid points"))
                    st.caption(
                        "建议：401–1201 较稳；N 越大越慢但结果更平滑。" if is_cn() else
                        "Tip: 401–1201 is usually stable; larger N is slower but smoother."
                    )

                inputs_obj = ModelInputs(
                    T_H=T_H, T_C=T_C, I=I,
                    A_cu_mm2=A_cu_mm2, L_cu=L_cu,
                    L_ybco=L_ybco, n_ybco=int(n_ybco),
                    A_shunt_mm2=A_shunt_mm2,
                    R_joint_uohm=R_joint_uohm, L_joint_cm=L_joint_cm,
                    N=int(N),
                )

                ok, errs, warns = validate_inputs(inputs_obj)
                if errs:
                    st.error(" | ".join(errs))
                if warns:
                    st.warning(" | ".join(warns))

                run_btn = st.form_submit_button(("🚀 开始计算" if is_cn() else "🚀 Run"), use_container_width=True)

    # -------------------------
    # Right: Schematic + Key results
    # -------------------------
    with right:
        with st.container(border=True):
            st.subheader("模型结构示意" if is_cn() else "Schematic")
            if os.path.exists(SCHEMATIC_PATH):
                # 你已设为 750，适合放在右侧列；可按屏幕微调
                st.image(SCHEMATIC_PATH, use_container_width=True)
                st.caption(
                    "Cu段 + 搭接焊接接头区 + HTS段（含分流器），热流方向：T_H → T_L"
                    if is_cn() else
                    "Cu segment + lap-joint + HTS segment (with shunt), heat flow: T_H → T_L"
                )
            else:
                st.warning(
                    ("未找到示意图：assets/schematic.png" if is_cn() else "Schematic not found: assets/schematic.png"))

        with st.container(border=True):
            st.subheader("关键结果" if is_cn() else "Key results")
            out = st.session_state.out
            if out is None:
                st.info(
                    "尚未计算。请在左侧输入参数并点击「开始计算」。" if is_cn() else "Not computed yet. Fill inputs on the left and click Run.")
            else:
                c1, c2 = st.columns(2)
                c1.metric("Q_cold (W)", nice_number(out["Q_cold"]))
                c2.metric("Q_joint (W)", nice_number(out["Q_joint"]))
                c3, c4 = st.columns(2)
                c3.metric("Q_joule_total (W)", nice_number(out["Q_joule_total"]))
                c4.metric("Q_hot (W)", nice_number(out["Q_hot"]))

                tj = out.get("T_joint", None)
                if tj is not None and np.isfinite(tj):
                    level, msg = joint_temp_assessment(float(tj))
                    {"error": st.error, "warning": st.warning, "success": st.success}[level](msg)

    # -------------------------
    # Run compute (沿用原始计算逻辑)
    # -------------------------
    if run_btn:
        ok2, errs2, warns2 = validate_inputs(inputs_obj)
        if warns2:
            st.warning(" | ".join(warns2))
        if errs2:
            st.error(" | ".join(errs2))
            st.stop()
        # 记住上次输入（用于下次打开时自动带出）
        st.session_state["T_H"] = float(T_H)
        st.session_state["T_C"] = float(T_C)
        st.session_state["I"] = float(I)
        st.session_state["A_cu_mm2"] = float(A_cu_mm2)
        st.session_state["L_cu"] = float(L_cu)
        st.session_state["L_ybco"] = float(L_ybco)
        st.session_state["n_ybco"] = int(n_ybco)
        st.session_state["A_shunt_mm2"] = float(A_shunt_mm2)
        st.session_state["R_joint_uohm"] = float(R_joint_uohm)
        st.session_state["L_joint_cm"] = float(L_joint_cm)
        st.session_state["N"] = int(N)

        with st.spinner("计算中..." if is_cn() else "Running..."):
            si = inputs_obj.to_si()
            out = run_model(
                T_H=si["T_H"], T_C=si["T_C"], I=si["I"],
                A_cu=si["A_cu"], L_cu=si["L_cu"],
                L_ybco=si["L_ybco"], A_shunt=si["A_shunt"], n_ybco=si["n_ybco"],
                R_joint=si["R_joint"], L_joint=si["L_joint"],
                N=si["N"],
            )
        st.session_state.out = out
        st.session_state.last_inputs = inputs_obj.to_display_dict()
        append_history(inputs_obj, out)
        st.toast(("计算完成 ✅" if is_cn() else "Done ✅"), icon="✅")
        st.rerun()
with tab_plots:
    with st.container(border=True):
        st.subheader(t("plots_title"))

        if out is None:
            st.info(t("plots_no"))
        else:
            # --- Put interpretation hints at the top (user asked) ---
            with st.expander(t("explain_hint"), expanded=False):
                if is_cn():
                    st.write(
                        "- **Q_cold**：冷端漏热（低温端负担指标）"

                        "- **Q_joint**：接头电阻导致的集中焦耳热（∝ I²）"

                        "- **导热占比**：YBCO 段中 Hastelloy/Cu/shunt 的相对导热贡献"
                    )
                else:
                    st.write(
                        "- **Q_cold**: heat leak into the cold end"

                        "- **Q_joint**: Joule heat generated in the joint (∝ I²)"

                        "- **Conduction fraction**: relative contribution of Hastelloy/Cu/shunt in YBCO segment"
                    )

            ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")

            # ---------- Figure 1 ----------
            c_title, c_dl = st.columns([4, 1])
            with c_title:
                st.markdown(f"#### {t('fig1')}")
            with c_dl:
                st.download_button(
                    label=("⬇️ 下载图1" if is_cn() else "⬇️ Download Fig1"),
                    data=out["img_T"],
                    file_name=f"Fig1-TemperatureProfile-{ts}.png",
                    mime="image/png",
                    use_container_width=True,
                    key="dl_fig1_png",
                )
            st.image(out["img_T"], use_container_width=True)

            st.write("")

            # ---------- Figure 2 ----------
            c_title2, c_dl2 = st.columns([4, 1])
            with c_title2:
                st.markdown(f"#### {t('fig2')}")
            with c_dl2:
                st.download_button(
                    label=("⬇️ 下载图2" if is_cn() else "⬇️ Download Fig2"),
                    data=out["img_frac"],
                    file_name=f"Fig2-ConductionFraction-{ts}.png",
                    mime="image/png",
                    use_container_width=True,
                    key="dl_fig2_png",
                )
            st.image(out["img_frac"], use_container_width=True)

# ===== Report =====
with tab_report:
    with st.container(border=True):
        st.subheader(t("report_title"))

        if out is None or st.session_state.last_inputs is None:
            st.info(t("report_no"))
        else:
            inputs_disp = st.session_state.last_inputs

            results_disp = {
                "Q_cold (W)": f"{out['Q_cold']:.6f}",
                "Q_joint (W)": f"{out['Q_joint']:.6f}",
                "Q_joule_total (W)": f"{out['Q_joule_total']:.6f}",
                "Q_hot (W)": f"{out['Q_hot']:.6f}",
                "T_joint (K)": f"{out.get('T_joint', float('nan')):.2f}",
            }

            meta = {"generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

            pdf_title = "Cu–YBCO Thermal Report / 计算报告"
            if is_cn():
                pdf_title = "Cu–YBCO 引线稳态导热 + 焦耳热 计算报告 / Thermal Report"

            pdf_bytes = build_pdf_report(
                title=pdf_title,
                meta=meta,
                inputs=inputs_disp,
                results=results_disp,
                img_T=out["img_T"],
                img_frac=out["img_frac"],
            )

            colA, colB = st.columns([2, 1])
            with colA:
                st.write(t("report_note"))
            with colB:
                st.download_button(
                    label=t("report_btn"),
                    data=pdf_bytes,
                    file_name=f"Cu-YBCO-Report-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

            if not os.path.exists(FONT_PATH):
                st.warning(t("font_warn"))

# ===== History =====
with tab_history:
    with st.container(border=True):
        st.subheader(t("hist_title"))
        df = history_df()

        if df.empty:
            st.info(t("hist_no"))
        else:
            st.caption(t("hist_tip"))
            st.dataframe(df, use_container_width=True, height=340)

            col1, col2, col3 = st.columns([1.2, 1.2, 1.0])
            with col1:
                st.download_button(
                    t("export_all"),
                    data=df_to_csv_bytes(df),
                    file_name=f"Cu-YBCO-History-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with col2:
                st.download_button(
                    t("export_key"),
                    data=df_to_csv_bytes(
                        df[["run_id", "time", "Q_cold (W)", "Q_joint (W)", "Q_joule_total (W)", "Q_hot (W)"]]),
                    file_name=f"Cu-YBCO-KeyResults-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with col3:
                if st.button(t("clear_hist"), use_container_width=True):
                    st.session_state.history = []
                    st.session_state.run_count = 0
                    st.toast(t("hist_cleared"), icon="🧹")
                    st.rerun()

# ===== About =====
with tab_about:
    with st.container(border=True):
        st.subheader(t("about_title"))

        # 中文模式：先中文后英文；英文模式：先英文后中文
        if is_cn():
            st.markdown(
                """
**物性参数 / Materials**
- 无氧铜导热率（RRR=50）：NIST 数据,https://trc.nist.gov/cryogenics/materials/OFHC%20Copper/OFHC_Copper_rev1.htm  
- 不锈钢导热率（UNS S30400）：NIST 数据,https://trc.nist.gov/cryogenics/materials/304Stainless/304Stainless_rev.htm  
- 哈氏合金 C-276 导热率：论文拟合*Physical properties of Hastelloy C-276 at cryogenic temperatures* + 测试修正
- 无氧铜电阻率：*Thermal Conductivity of Aluminum, Copper, Iron, and Tungsten for Temperatures from 1 K to the Melting Point NIST Monograph 177, 1984*

**YBCO 带材 / YBCO Tape**
- 型号：上海超导 ST-4-E；宽度 4 mm
- 基带厚度 45 μm；铜层厚度 8 μm × 2；Ic ~ 140 A

**YBCO 等效热导率**              
- k_eq=(k_hast·A_hast + k_cu_tape·A_cu_tape + k_ss·A_shunt) / (A_hast + A_cu_tape + A_shunt)            
                """
            )
        else:
            st.markdown(
                """
**Materials / 物性参数**
- OFHC Copper thermal conductivity (RRR=50): NIST,https://trc.nist.gov/cryogenics/materials/OFHC%20Copper/OFHC_Copper_rev1.htm  
- SS304 thermal conductivity (UNS S30400): NIST, https://trc.nist.gov/cryogenics/materials/304Stainless/304Stainless_rev.htm  
- Hastelloy C-276 thermal conductivity: fitted from literature *Physical properties of Hastelloy C-276 at cryogenic temperatures* + test-based correction
- OFHC Copper resistivity: *Thermal Conductivity of Aluminum, Copper, Iron, and Tungsten for Temperatures from 1 K to the Melting Point NIST Monograph 177, 1984*

**YBCO Tape / YBCO 带材**
- Model: Shanghai Superconductor ST-4-E; width 4 mm
- Substrate thickness 45 μm; Cu thickness 8 μm × 2; Ic ~ 140 A

**YBCO equivalent thermal conductivity**              
- k_eq=(k_hast·A_hast + k_cu_tape·A_cu_tape + k_ss·A_shunt) / (A_hast + A_cu_tape + A_shunt)                 
                """
            )

# ===== Footer =====
st.markdown("---")
st.caption(t("copyright"))
st.caption(t("center"))