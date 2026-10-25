
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from models.resource_model import User
from algorithms.baseline import equal_share, demand_proportional
from algorithms.nash_bargaining import solve_nash_bargaining
from evaluation.metrics import evaluate_allocation


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ResourceBargain",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DEFAULTS = [
    ("AI Research", 8.0, 1.5, 0.0),
    ("Data Analytics", 6.0, 1.0, 0.0),
    ("Student Project", 5.0, 1.2, 0.0),
    ("Cloud Workload", 7.0, 1.0, 0.0),
    ("Simulation Task", 4.0, 0.8, 0.0),
    ("Research Lab", 9.0, 1.4, 0.0),
    ("Batch Processing", 3.0, 1.0, 0.0),
    ("Edge Computing", 6.0, 1.1, 0.0),
]


# ============================================================
# VISUAL DESIGN
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --ink: #111313;
        --panel: #181c1c;
        --panel2: #202525;
        --line: #343a38;
        --paper: #eeeee8;
        --muted: #929b96;
        --cyan: #8cf0df;
        --white: #fafbf7;
    }

    .stApp {
        background: var(--ink);
        color: var(--paper);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1500px;
        padding: 1.4rem 2rem 2.2rem;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.04em;
        color: var(--white);
    }

    h1 {
        font-size: clamp(2.3rem, 4vw, 4rem) !important;
        line-height: 1 !important;
        font-weight: 600 !important;
    }

    h2 {
        font-size: 1.45rem !important;
        font-weight: 500 !important;
    }

    h3 {
        font-size: 1.1rem !important;
        font-weight: 500 !important;
    }

    p, label, li {
        color: var(--paper);
    }

    hr {
        border-color: var(--line);
        margin: 1rem 0;
    }

    .rb-kicker {
        color: var(--cyan);
        font-family: 'DM Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .rb-copy {
        font-size: 0.86rem;
        color: var(--muted);
        line-height: 1.6;
    }

    .rb-rule {
        height: 1px;
        background: var(--line);
        margin: 1rem 0;
    }

    .rb-tag {
        display: inline-block;
        padding: 0.3rem 0.5rem;
        border: 1px solid var(--line);
        color: var(--cyan);
        font-family: 'DM Mono', monospace;
        font-size: 0.62rem;
        letter-spacing: 0.06em;
        overflow-wrap: anywhere;
    }

    .rb-panel {
        background: var(--panel);
        border: 1px solid var(--line);
        padding: 1.15rem;
    }

    [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 3px;
        padding: 0.8rem 0.9rem;
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted) !important;
        font-family: 'DM Mono', monospace;
        font-size: 0.64rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    [data-testid="stMetricValue"] {
        color: var(--cyan) !important;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.65rem !important;
    }

    .stButton > button {
        background: var(--cyan);
        color: var(--ink);
        border: 1px solid var(--cyan);
        border-radius: 2px;
        min-height: 2.8rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        transition: 0.15s ease;
    }

    .stButton > button:hover {
        background: var(--white);
        color: var(--ink);
        border-color: var(--white);
    }

    .stDownloadButton > button {
        background: transparent;
        color: var(--cyan);
        border: 1px solid var(--line);
        border-radius: 2px;
        min-height: 2.6rem;
    }

    [data-testid="stDataFrame"],
    [data-testid="stDataEditor"] {
        border: 1px solid var(--line);
    }

    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input {
        background: var(--panel);
        color: var(--white);
        border-color: var(--line);
        border-radius: 2px;
    }

    [data-testid="stExpander"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 2px;
        margin-bottom: 0.3rem;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    @media (max-width: 900px) {
        .block-container {
            padding: 1rem 1rem 2rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def section_label(number, label):
    st.markdown(
        f'<div class="rb-kicker">{number} / {label}</div>',
        unsafe_allow_html=True,
    )


def create_allocation_chart(users, allocations):
    """Create a comparison chart for all allocation strategies."""
    fig, ax = plt.subplots(figsize=(7.5, 3.4))

    fig.patch.set_facecolor("#181c1c")
    ax.set_facecolor("#181c1c")

    names = [user.name for user in users]
    strategies = list(allocations.keys())
    palette = ["#69736e", "#b5bcb4", "#8cf0df"]

    positions = list(range(len(names)))
    width = 0.23

    for index, strategy in enumerate(strategies):
        offset = (index - 1) * width

        ax.bar(
            [p + offset for p in positions],
            allocations[strategy],
            width,
            label=strategy,
            color=palette[index],
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        names,
        color="#c5cbc5",
        fontsize=8,
        rotation=15,
        ha="right",
    )

    ax.tick_params(axis="y", colors="#929b96", labelsize=8)

    ax.set_ylabel(
        "Resource units",
        color="#929b96",
        fontsize=8,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#343a38")

    ax.grid(
        axis="y",
        color="#343a38",
        linewidth=0.6,
    )

    ax.set_axisbelow(True)

    ax.legend(
        frameon=False,
        labelcolor="#eeeee8",
        fontsize=7,
        loc="upper right",
    )

    fig.tight_layout()
    return fig


def validate_users(users):
    """Validate participant names and model inputs."""
    if not users:
        raise ValueError("At least one user is required.")

    if any(not user.name.strip() for user in users):
        raise ValueError("User names cannot be empty.")

    names = [user.name.strip() for user in users]

    if len(names) != len(set(names)):
        raise ValueError("User names must be unique.")


def calculate_simulation(users, capacity):
    """Run the three allocation strategies and evaluate their results."""
    nash_result = solve_nash_bargaining(users, capacity)

    allocations = {
        "Equal Sharing": equal_share(users, capacity),
        "Demand Proportional": demand_proportional(
            users, capacity
        ),
        "Nash Bargaining": nash_result["allocation"],
    }

    metrics = {
        strategy: evaluate_allocation(
            users,
            allocation,
            capacity,
        )
        for strategy, allocation in allocations.items()
    }

    return {
        "users": users,
        "capacity": capacity,
        "allocations": allocations,
        "metrics": metrics,
        "solver_message": nash_result["message"],
    }


def make_input_signature(users, capacity):
    """Create a comparable representation of the current inputs."""
    return (
        float(capacity),
        tuple(
            (
                user.name,
                float(user.demand),
                float(user.weight),
                float(user.disagreement_utility),
            )
            for user in users
        ),
    )


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns([0.72, 0.28])

with header_left:
    st.markdown(
        '<div class="rb-kicker">GAME THEORY / RESOURCE SYSTEMS</div>',
        unsafe_allow_html=True,
    )

    st.markdown("# ResourceBargain")

    st.markdown(
        '<p class="rb-copy">A simulation of how limited computational '
        'resources can be allocated between competing users.</p>',
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown(
        """
        <div style="text-align:right;padding-top:0.6rem">
            <div class="rb-kicker">RESEARCH PROTOTYPE</div>
            <div style="font-family:'Space Grotesk';font-size:1.25rem;
                        margin-top:0.5rem;color:#fafbf7">
                NASH / 01
            </div>
            <div class="rb-copy">Centralized allocation model</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="rb-rule"></div>', unsafe_allow_html=True)


# ============================================================
# INPUT PANEL AND RESULTS PANEL
# ============================================================

controls, workspace = st.columns([0.43, 0.57], gap="large")


# ============================================================
# LEFT PANEL — SIMULATION INPUTS
# ============================================================

with controls:
    section_label("01", "Simulation parameters")
    st.markdown("## Define the system")

    capacity = st.number_input(
        "Available resource capacity",
        min_value=0.0,
        max_value=10000.0,
        value=10.0,
        step=1.0,
        key="resource_capacity",
    )

    number_of_users = st.slider(
        "Number of competing users",
        min_value=2,
        max_value=8,
        value=3,
        key="number_of_users",
    )

    st.markdown('<div class="rb-rule"></div>', unsafe_allow_html=True)

    section_label("02", "User profiles")

    st.markdown(
        '<p class="rb-copy">Edit participant demands and utility '
        'parameters directly in the table.</p>',
        unsafe_allow_html=True,
    )

    # IMPORTANT:
    # Keep the initial table under a separate session-state key.
    # Streamlit reserves editor_key for the widget itself.
    editor_key = f"profile_editor_{number_of_users}"
    initial_key = f"{editor_key}_initial"

    # Create fresh defaults when the number of users changes.
    if st.session_state.get("_profile_count") != number_of_users:
        st.session_state["_profile_count"] = number_of_users

        st.session_state[initial_key] = pd.DataFrame([
            {
                "User": DEFAULTS[i][0],
                "Demand": DEFAULTS[i][1],
                "Weight": DEFAULTS[i][2],
                "Fallback": DEFAULTS[i][3],
            }
            for i in range(number_of_users)
        ])

    # Recover the initial data without assigning to the widget's key.
    if initial_key not in st.session_state:
        st.session_state[initial_key] = pd.DataFrame([
            {
                "User": DEFAULTS[i][0],
                "Demand": DEFAULTS[i][1],
                "Weight": DEFAULTS[i][2],
                "Fallback": DEFAULTS[i][3],
            }
            for i in range(number_of_users)
        ])

    edited_users = st.data_editor(
        st.session_state[initial_key],
        key=editor_key,
        num_rows="fixed",
        use_container_width=True,
        hide_index=True,
        column_config={
            "User": st.column_config.TextColumn(
                "Participant",
                required=True,
                width="medium",
            ),
            "Demand": st.column_config.NumberColumn(
                "Demand",
                min_value=0.0,
                max_value=10000.0,
                step=1.0,
                required=True,
                width="small",
            ),
            "Weight": st.column_config.NumberColumn(
                "Weight",
                min_value=0.01,
                max_value=100.0,
                step=0.1,
                required=True,
                width="small",
            ),
            "Fallback": st.column_config.NumberColumn(
                "Fallback",
                min_value=0.0,
                max_value=100.0,
                step=0.1,
                required=True,
                width="small",
            ),
        },
    )

    st.markdown(
        '<p class="rb-copy">Weight controls the curvature of the utility '
        'function. Fallback represents the disagreement utility used '
        'in bargaining.</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="rb-rule"></div>', unsafe_allow_html=True)

    run_simulation = st.button(
        "RUN SIMULATION  ↗",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# BUILD CURRENT INPUTS
# ============================================================

current_users = None
current_signature = None

try:
    current_users = [
        User(
            name=str(row["User"]).strip(),
            demand=float(row["Demand"]),
            weight=float(row["Weight"]),
            disagreement_utility=float(row["Fallback"]),
        )
        for row in edited_users.to_dict(orient="records")
    ]

    validate_users(current_users)

    current_signature = make_input_signature(
        current_users,
        capacity,
    )

except (ValueError, TypeError, KeyError) as exc:
    st.error(f"Please check the input values: {exc}")


# ============================================================
# RUN THE SIMULATION
# ============================================================

if run_simulation and current_users is not None:
    try:
        result = calculate_simulation(current_users, capacity)

        st.session_state["rb_results"] = result
        st.session_state["rb_input_signature"] = current_signature

    except (ValueError, RuntimeError, TypeError, KeyError) as exc:
        st.error(f"Simulation could not be completed: {exc}")


# ============================================================
# RIGHT PANEL — RESULTS
# ============================================================

with workspace:
    section_label("03", "Allocation analysis")
    st.markdown("## The allocation")

    data = st.session_state.get("rb_results")
    previous_signature = st.session_state.get("rb_input_signature")

    if data is not None and current_signature != previous_signature:
        st.markdown(
            '<span class="rb-tag">INPUTS CHANGED / RERUN TO REFRESH</span>',
            unsafe_allow_html=True,
        )

    if data is None:
        st.markdown(
            """
            <div class="rb-panel">
                <div class="rb-kicker">AWAITING INPUT</div>
                <div style="font-family:'Space Grotesk';font-size:1.8rem;
                            margin-top:0.8rem;color:#fafbf7">
                    Set the constraints.
                </div>
                <p class="rb-copy">
                    Configure the users on the left and run the simulation
                    to calculate allocations and compare the strategies.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="rb-rule"></div>', unsafe_allow_html=True)

        section_label("04", "Allocation methods")

        st.markdown("**Nash bargaining**")
        st.markdown(
            '<p class="rb-copy">Maximizes the product of utility gains '
            'over disagreement outcomes.</p>',
            unsafe_allow_html=True,
        )

        st.markdown("**Equal sharing**")
        st.markdown(
            '<p class="rb-copy">Distributes available resources evenly '
            'while respecting individual demand limits.</p>',
            unsafe_allow_html=True,
        )

        st.markdown("**Demand proportional**")
        st.markdown(
            '<p class="rb-copy">Allocates resources in proportion '
            'to requested demand.</p>',
            unsafe_allow_html=True,
        )

    else:
        users = data["users"]
        allocations = data["allocations"]
        metrics = data["metrics"]
        nash_metrics = metrics["Nash Bargaining"]

        st.markdown(
            f'<span class="rb-tag">SOLVER / '
            f'{data["solver_message"].upper()}</span>',
            unsafe_allow_html=True,
        )

        st.markdown("")

        metric_a, metric_b = st.columns(2)

        metric_a.metric(
            "Allocated capacity",
            f'{nash_metrics["allocated_capacity"]:.2f}',
            f'of {data["capacity"]:.2f} available',
        )

        metric_b.metric(
            "Resource utilization",
            f'{nash_metrics["resource_utilization"]:.1%}',
        )

        metric_c, metric_d = st.columns(2)

        metric_c.metric(
            "Satisfaction fairness",
            f'{nash_metrics["satisfaction_fairness"]:.3f}',
        )

        metric_d.metric(
            "Aggregate utility",
            f'{nash_metrics["total_utility"]:.3f}',
        )

        st.markdown('<div class="rb-rule"></div>', unsafe_allow_html=True)

        section_label("04", "Resource distribution")
        st.markdown("### Allocation by participant")

        allocation_df = pd.DataFrame({
            "Participant": [u.name for u in users],
            "Demand": [u.demand for u in users],
            "Equal": allocations["Equal Sharing"],
            "Proportional": allocations["Demand Proportional"],
            "Nash": allocations["Nash Bargaining"],
        })

        st.dataframe(
            allocation_df.style.format({
                "Demand": "{:.2f}",
                "Equal": "{:.2f}",
                "Proportional": "{:.2f}",
                "Nash": "{:.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown('<div class="rb-rule"></div>', unsafe_allow_html=True)

        section_label("05", "Strategy comparison")
        st.markdown("### Allocation distribution")

        fig = create_allocation_chart(users, allocations)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        comparison_df = pd.DataFrame({
            strategy: {
                "Total utility": result["total_utility"],
                "Satisfaction fairness": result["satisfaction_fairness"],
                "Allocation fairness": result["allocation_fairness"],
                "Resource utilization": result["resource_utilization"],
            }
            for strategy, result in metrics.items()
        }).T

        comparison_df.index.name = "Strategy"

        st.markdown("### Performance metrics")

        st.dataframe(
            comparison_df.style.format("{:.4f}"),
            use_container_width=True,
        )

        with st.expander("Inspect individual user utilities"):
            utility_df = pd.DataFrame({
                strategy: result["utilities"]
                for strategy, result in metrics.items()
            }, index=[u.name for u in users])

            st.bar_chart(utility_df)

        with st.expander("How is fairness measured?"):
            st.markdown(
                """
                **Demand satisfaction fairness** measures how evenly
                users' allocation-to-demand ratios are distributed.

                **Allocation fairness** measures equality in raw resource
                allocations.

                **Total utility** sums the utility obtained by all users.

                These metrics measure different properties. A high
                fairness score does not necessarily imply maximum total
                utility. Nash bargaining optimizes its defined bargaining
                objective rather than directly maximizing every metric.
                """
            )

        st.markdown('<div class="rb-rule"></div>', unsafe_allow_html=True)

        section_label("06", "Export results")

        download_a, download_b = st.columns(2)

        with download_a:
            st.download_button(
                "ALLOCATION CSV ↓",
                data=allocation_df.to_csv(index=False),
                file_name="allocation_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with download_b:
            st.download_button(
                "METRICS CSV ↓",
                data=comparison_df.to_csv(),
                file_name="strategy_metrics.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown('<div class="rb-rule"></div>', unsafe_allow_html=True)

footer_left, footer_right = st.columns([0.7, 0.3])

with footer_left:
    st.markdown(
        '<div class="rb-kicker">RESOURCEBARGAIN / COMPUTATIONAL GAME THEORY</div>',
        unsafe_allow_html=True,
    )

with footer_right:
    st.markdown(
        '<div style="text-align:right" class="rb-copy">'
        'ACADEMIC PROJECT / 2026</div>',
        unsafe_allow_html=True,
    )
