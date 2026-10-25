"""Colors and chart styling shared by the dashboard pages (palette and rules: dashboard/DASHBOARD.md).

Each arm keeps one categorical color on every chart, so a filter that drops an arm never repaints the
others. Magnitudes use one blue ramp; pass/fail uses the status colors, always next to a ✓/✗ label.
Light and dark steps are separate, validated sets; the page's Streamlit theme picks one.
"""

from typing import Dict, List

# categorical slots in fixed order (validated: adjacent CVD ΔE >= 8.4, normal-vision ΔE >= 19.3, both modes)
CATEGORICAL = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
ARM_ORDER = ["unprotected", "nemo_rails", "no_action_gate", "action_gate_only", "memguard", "memguard_ttl"]
# one blue hue, light -> dark (near zero recedes toward the surface)
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
STATUS = {"good": "#0ca30c", "critical": "#d03b3b"}


def theme(st) -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def arm_colors(st, arms: List[str]) -> Dict[str, str]:
    """A fixed color per arm; arms outside ARM_ORDER take the slots after it, in first-seen order."""
    slots = CATEGORICAL[theme(st)]
    order = ARM_ORDER + [a for a in arms if a not in ARM_ORDER]
    return {a: slots[order.index(a) % len(slots)] for a in arms}


def arm_order(arms) -> List[str]:
    return sorted(set(arms), key=lambda a: (ARM_ORDER.index(a) if a in ARM_ORDER else len(ARM_ORDER), a))


def tint(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}, {alpha})"


def top_down(fig):
    """Horizontal grouped bars draw the first series at the bottom of each group; flip so the bars read
    top-down in legend order (unprotected first)."""
    fig.data = tuple(reversed(fig.data))
    fig.update_layout(legend_traceorder="reversed")
    return fig


def facets(fig, n_rows: int):
    """Small multiples: room above for their titles (which otherwise cover the top row), legend below them."""
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(height=110 + 36 * n_rows, margin=dict(l=8, r=8, t=40, b=48),
                      legend=dict(orientation="h", yanchor="top", y=-0.04, x=0, title_text=""))
    return fig


def tidy(fig, height: int = 360):
    """Recessive grid and axes, legend above the plot, no chart-junk margins."""
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=36, b=8), bargap=0.25, bargroupgap=0.08,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridwidth=1, zeroline=False)
    fig.update_traces(marker_line_width=0, selector=dict(type="bar"))
    return fig
