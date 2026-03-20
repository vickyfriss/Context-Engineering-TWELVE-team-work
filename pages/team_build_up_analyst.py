"""
Entrypoint for the Team Build-Up Analyst Streamlit page.
Runs top to bottom every time the user interacts with the app.
"""

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from classes.data_source import TeamStats
from classes.visual import DistributionPlot
from classes.description import TeamDescription
from classes.chat import TeamChat

from utils.page_components import add_common_page_elements
from utils.utils import create_chat


# Add common page elements and create sidebar containers
sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

st.divider()

st.title("Team Build-Up Analyst")
st.write("Team build-up profiling using tournament-level metrics.")

# Define the metrics used for the team build-up analysis
metrics = [
    "buildup_to_create_pct",
    "buildup_to_direct_pct",
    "buildup_that_ends_with_finish_pct",
    "turnover_pct_buildup",
    "opp_box_entries_within_7s_after_turnover",
    "opp_shot_probability_within_7s_after_turnover",
    "first_line_break_pct_buildup",
    # "second_last_line_break_pct_buildup",
]

# Define metrics where lower values are better
negative_metrics = [
    "turnover_pct_buildup",
    "opp_box_entries_within_7s_after_turnover",
    "opp_shot_probability_within_7s_after_turnover",
]

# Load team dataset
teams = TeamStats(data_folder="data/team_build_up_analyst")

# Calculate z-scores and ranks for selected metrics
teams.calculate_statistics(
    metrics=metrics,
    negative_metrics=negative_metrics,
)

# Select focal team from sidebar
with sidebar_container:
    st.subheader("Team selection")
    selected_team = st.selectbox(
        "Select team",
        options=teams.df["team"].unique(),
        index=0,
    )

# Convert selected team into a Team data point
team = teams.to_data_point_by_team(selected_team)

# Optional: show dataset in expander for debugging / transparency
st.expander("Dataframe used", expanded=False).write(teams.df)

# Chat state hash determines whether a new chat should be created
# or an existing one should continue.
to_hash = (team.id, "team_build_up_analyst")

# Create the chat as type TeamChat
chat = create_chat(to_hash, TeamChat, team, teams)

# Add initial content if the chat is empty
if chat.state == "empty":

    # Create the distribution plot
    visual = DistributionPlot(
        columns=metrics[::-1],
        labels=["Worse", "Average", "Better"],
        plot_type="default",
    )
    visual.add_title(
        title=f"Team build-up profile: {team.name}",
        subtitle="Compared to other teams in the dataset (z-scores)",
    )
    visual.add_players(teams, metrics=metrics)
    visual.add_player(team, len(teams.df), metrics=metrics)

    # Create the initial team summary
    description = TeamDescription(team)

    # Use stream_gpt if TeamDescription supports it.
    # If not, replace this line with:
    # summary = description.synthesize_text()
    summary = description.stream_gpt(stream=True)

    # Add starter content to the chat
    chat.add_message(
        f"Please can you summarise {team.name}'s build-up style for me?",
        role="user",
        user_only=False,
        visible=False,
    )
    chat.add_message(visual)
    chat.add_message(summary)

    chat.state = "default"

# Show correlation matrix below the chat starter content
st.subheader("Correlation Matrix")

corr_df = teams.df[metrics].corr(method="pearson")

cmap = LinearSegmentedColormap.from_list(
    "red_white_green",
    ["#d73027", "#ffffff", "#1a9850"],
)

# Minimal style for no background
sns.set_style("white")
sns.set_context("notebook", font_scale=0.8)  # smaller global font

# Use a narrower figure to occupy ~60% of page width
fig_width = 6  # inches
fig_height = 4.5  # proportionally smaller height
fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor="none")

# Prepare readable labels
def format_label(label):
    has_pct = False
    if "pct" in label:
        has_pct = True
        label = label.replace("pct", "").strip("_")
    label = label.replace("_", " ").title()
    if has_pct:
        label += " (%)"
    # Wrap long labels
    if len(label) > 20:
        parts = label.split(" ")
        mid = len(parts) // 2
        label = " ".join(parts[:mid]) + "\n" + " ".join(parts[mid:])
    return label

labels = [format_label(m) for m in metrics]

# Draw heatmap
sns.heatmap(
    corr_df,
    annot=True,
    fmt=".2f",
    cmap=cmap,
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    linecolor="lightgray",
    square=True,
    annot_kws={"size": 8, "weight": "bold"},  # smaller numbers
    cbar_kws={"shrink": 0.7},
)

# Apply formatted labels
ax.set_xticklabels(labels, rotation=45, ha="right", wrap=True)
ax.set_yticklabels(labels, rotation=0, va="center")

sns.despine(left=True, bottom=True)

ax.set_title(
    "Correlation Matrix – Team Build-Up Metrics",
    fontsize=12,  # smaller title
    weight="bold"
)

# Display figure in Streamlit, transparent background
st.pyplot(fig, transparent=True)

# Run chat interaction
chat.get_input()
chat.display_messages()
chat.save_state()