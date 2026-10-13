"""
draw_methodology_flowchart.py

Generates a publication-quality Methodology Flowchart diagram for the AURA-Cache
project report (IEEE Journal / Conference format).
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def generate_methodology_flowchart():
    # Academic publication dimensions (16:9 high-res canvas)
    fig, ax = plt.subplots(figsize=(18, 10.5), dpi=300)
    
    # Clean professional modern theme
    bg_color = "#0A0F1D"       # Deep tech background
    card_bg = "#11192E"        # Card background
    card_border = "#202D4C"    # Card border
    text_white = "#F8FAFC"
    text_muted = "#94A3B8"
    accent_blue = "#0EA5E9"    # Sky / Cyan
    accent_purple = "#8B5CF6"  # Violet
    accent_green = "#10B981"   # Emerald
    accent_amber = "#F59E0B"   # Gold / Orange
    accent_rose = "#F43F5E"    # Rose / Red
    
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10.5)
    ax.axis('off')
    
    # --- Top Banner & Title ---
    banner = FancyBboxPatch((0.8, 9.4), 16.4, 0.9, boxstyle="round,pad=0.15",
                            facecolor="#131C35", edgecolor=accent_blue, linewidth=1.5)
    ax.add_patch(banner)
    ax.text(9.0, 9.9, "AURA-Cache: End-to-End Research Methodology Pipeline", 
            ha='center', va='center', fontsize=16, fontweight='bold', color=text_white, fontfamily='sans-serif')
    ax.text(9.0, 9.55, "Cycle-Accurate Microarchitectural Telemetry  •  Belady Oracle Supervision  •  Unsupervised Phase Discovery  •  Deep Q-Learning  •  C++ Hardware Deployment",
            ha='center', va='center', fontsize=9.5, color=accent_blue, fontfamily='sans-serif')
    
    # Helper to draw module box
    def draw_node(x, y, w, h, step_tag, title, desc_lines, border_color, glow=False, is_sub=False):
        # Card body
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                               facecolor=card_bg, edgecolor=border_color, linewidth=2.0 if glow else 1.2)
        ax.add_patch(patch)
        
        # Step Tag Header
        tag_bg = FancyBboxPatch((x + 0.15, y + h - 0.35), w - 0.3, 0.25, boxstyle="round,pad=0.04",
                                facecolor=border_color, edgecolor='none', alpha=0.25)
        ax.add_patch(tag_bg)
        ax.text(x + w/2, y + h - 0.22, step_tag.upper(),
                ha='center', va='center', fontsize=7.5, fontweight='bold', color=border_color, fontfamily='monospace')
        
        # Title
        ax.text(x + w/2, y + h - 0.55, title,
                ha='center', va='center', fontsize=10.5, fontweight='bold', color=text_white, fontfamily='sans-serif')
        
        # Description Lines
        line_y = y + h - 0.85
        for line in desc_lines:
            ax.text(x + 0.25, line_y, line,
                    ha='left', va='center', fontsize=8.0, color=text_muted, fontfamily='sans-serif')
            line_y -= 0.24

    # Helper to draw arrows
    def draw_flow_arrow(x1, y1, x2, y2, color=accent_blue, label=None, label_side="top"):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                arrowstyle="-|>,head_length=7,head_width=4.5",
                                color=color, linewidth=2.0, mutation_scale=12)
        ax.add_patch(arrow)
        if label:
            mx, my = (x1 + x2)/2, (y1 + y2)/2
            offset = 0.18 if label_side == "top" else -0.18
            ax.text(mx, my + offset, label, ha='center', va='center', fontsize=7.5, 
                    fontweight='bold', color=color, fontfamily='sans-serif',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='#0A0F1D', edgecolor='none', alpha=0.85))

    # =========================================================================
    # ROW 1: INPUT TRACES & HARDWARE TELEMETRY LOGGING (Y: 6.8 to 8.8)
    # =========================================================================
    
    # 1.1 SPEC Traces
    draw_node(0.8, 6.8, 3.2, 2.0, "Input Sources", "SPEC CPU Traces", [
        "• 403.gcc (Compiler Pipeline)",
        "• 429.mcf (Pointer-Chasing)",
        "• 400.perlbench (Interpreter)",
        "• Instruction addresses & IPs"
    ], accent_amber)
    
    # 1.2 ChampSim Engine
    draw_node(4.8, 6.8, 3.6, 2.0, "Simulation Engine", "ChampSim Simulator", [
        "• Out-of-Order CPU Core (4 GHz)",
        "• L1 (32KB) & L2 (512KB) Hierarchy",
        "• 2 MB 16-Way Last-Level Cache",
        "• Intercepts all LLC memory misses"
    ], accent_blue)
    
    # 1.3 Hardware Telemetry Logger
    draw_node(9.2, 6.8, 3.8, 2.0, "Instrumentation", "Hardware Telemetry", [
        "• Way Recency Ranks: r_{w} (0..15)",
        "• Request Types: LOAD, RFO, PREF",
        "• Sliding window: W = 50 accesses",
        "• Rolling Miss Rate & Read Ratio"
    ], accent_purple)

    # 1.4 True Belady MIN Oracle
    draw_node(13.8, 6.8, 3.4, 2.0, "Supervision Target", "Belady MIN Oracle", [
        "• Future access lookahead distance",
        "• Evaluates reuse horizon tau(a_{w})",
        "• Selects optimal victim: w*",
        "• Establishes 100% optimal ceiling"
    ], accent_rose)
    
    # Row 1 Connections
    draw_flow_arrow(4.0, 7.8, 4.8, 7.8, accent_amber, "Trace Stream")
    draw_flow_arrow(8.4, 7.8, 9.2, 7.8, accent_blue, "LLC Accesses")
    draw_flow_arrow(13.0, 7.8, 13.8, 7.8, accent_purple, "Full Log (CSV)")

    # =========================================================================
    # ROW 2: UNSUPERVISED CLUSTERING & STATE TENSOR ENCODING (Y: 3.9 to 5.9)
    # =========================================================================

    # 2.1 KMeans Clustering
    draw_node(0.8, 3.8, 3.4, 2.1, "Phase Discovery", "Unsupervised KMeans", [
        "• Standardized 4D telemetry vector",
        "• K=4 Macro-Phase Archetypes:",
        "  - P0: Burst Scan / High Miss",
        "  - P1: Steady Compute Loop",
        "  - P2: High Temporal Reuse",
        "  - P3: Irregular Pointer Walk"
    ], accent_green)

    # 2.2 Phase Model Serialization
    draw_node(5.0, 3.8, 3.4, 2.1, "Model Artifacts", "Phase Model Export", [
        "• Fitted Scaler Mean & Scale",
        "• 4 Centroids (C_0, C_1, C_2, C_3)",
        "• Serialized to JSON format",
        "• Scaler parameters for online use"
    ], accent_green)

    # 2.3 28-dim State Construction
    draw_node(9.2, 3.8, 3.8, 2.1, "Feature Engineering", "28-Dim State Vector", [
        "• [0..15] Normalized Way Recency",
        "• [16..23] Access Type (One-Hot)",
        "• [24..27] KMeans Phase (One-Hot)",
        "• Concatenated state vector S_t"
    ], accent_blue)

    # 2.4 Offline Supervised MDP Dataset
    draw_node(13.8, 3.8, 3.4, 2.1, "Dataset Assembly", "Labeled RL Dataset", [
        "• Pairs: (State S_t, Victim w*)",
        "• Miss-only instances (h_t = 0)",
        "• Combined mcf + gcc datasets",
        "• 80/20 Chronological Split"
    ], accent_amber)

    # Connecting Row 1 to Row 2
    draw_flow_arrow(11.1, 6.8, 2.5, 5.9, accent_green, "Telemetry Features (M_t, R_t, I_t)")
    draw_flow_arrow(4.2, 4.85, 5.0, 4.85, accent_green, "Centroids")
    draw_flow_arrow(8.4, 4.85, 9.2, 4.85, accent_blue, "Phase IDs")
    draw_flow_arrow(15.5, 6.8, 15.5, 5.9, accent_rose, "Optimal Label w*")
    draw_flow_arrow(13.0, 4.85, 13.8, 4.85, accent_blue, "States S_t")

    # =========================================================================
    # ROW 3: DEEP REINFORCEMENT LEARNING & HARDWARE DEPLOYMENT (Y: 0.8 to 2.8)
    # =========================================================================

    # 3.1 Deep Q-Network
    draw_node(0.8, 0.8, 3.6, 2.1, "Policy Optimization", "Deep Q-Network (DQN)", [
        "• MLP: 28 -> 64 (ReLU) -> 32 -> 16",
        "• Masked Cross-Entropy Loss",
        "• Adam Optimizer (lr = 1e-3, 20 Ep)",
        "• Achieves 63.51% Belady Match"
    ], accent_purple)

    # 3.2 Two-Level HDQN Architecture
    draw_node(5.2, 0.8, 3.4, 2.1, "Hierarchical RL", "Two-Level HDQN", [
        "• Meta-Agent: Workload context goal",
        "• Action-Agent: Eviction ranker",
        "• Resolves cross-phase ambiguity",
        "• Eliminates temporal data leakage"
    ], accent_purple)

    # 3.3 Hardware Quantization
    draw_node(9.4, 0.8, 3.6, 2.1, "Hardware Budget", "INT8 Quantization", [
        "• Total parameters: 4,352 values",
        "• FP32 Footprint: 17.41 KB",
        "• INT8 Quantized: 4.35 KB SRAM",
        "• < 3.4% of 2 MB LLC Tag Store"
    ], accent_amber)

    # 3.4 C++ ChampSim & Interactive UI
    draw_node(13.8, 0.8, 3.4, 2.1, "Deployment & Eval", "C++ Engine & UI", [
        "• phase_aware_rl.cc drop-in policy",
        "• Pipelined inference: ~8 cycles",
        "• Masked beneath L2 miss penalty",
        "• Interactive 6-Stage Web Simulator"
    ], accent_blue)

    # Connecting Row 2 to Row 3
    draw_flow_arrow(14.5, 3.8, 2.6, 2.9, accent_purple, "Supervised (S_t, w*)")
    draw_flow_arrow(4.4, 1.85, 5.2, 1.85, accent_purple, "Comparative HDQN")
    draw_flow_arrow(8.6, 1.85, 9.4, 1.85, accent_amber, "Model Weights")
    draw_flow_arrow(13.0, 1.85, 13.8, 1.85, accent_blue, "Inference Engine")

    # Save outputs to both report/figures and review_figures
    out_path1 = "report/figures/methodology_flowchart.png"
    out_path2 = "review_figures/methodology_flowchart.png"
    
    fig.savefig(out_path1, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    if os.path.exists("review_figures"):
        fig.savefig(out_path2, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
        
    plt.close(fig)
    print(f"[SUCCESS] Methodology Flowchart saved to {out_path1} and {out_path2}")

if __name__ == "__main__":
    generate_methodology_flowchart()
