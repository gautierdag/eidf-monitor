import streamlit as st
import pandas as pd
import plotly.express as px

from utils import filter_while_true_pods, get_pending_pods

FILE_PATH = "/nfs/user/s2234411-infk8s/cluster_gpu_usage.json"


st.button("Refresh")

st.markdown("""
# Informatics EIDF Monitoring Dashboard
This dashboard shows the GPU usage in the cluster. 
            
The data is collected from the Kubernetes cluster every 15 minutes.
Active GPUs are those with memory usage greater than 1% in the last hour.
""")


def get_data() -> pd.DataFrame:
    df = pd.read_json(FILE_PATH)

    def add_gpu_id(gpu_usage):
        new_list = []
        for i, g in enumerate(gpu_usage):
            g["gpu_id"] = i
            new_list.append(g)
        return new_list

    interactive_pods = filter_while_true_pods()
    is_interactive = set()
    for pod in interactive_pods:
        is_interactive.add(pod["name"])

    df["is_interactive"] = df["pod_name"].isin(is_interactive)

    df["gpu_usage"] = df["gpu_usage"].apply(add_gpu_id)
    df = df.explode("gpu_usage")
    df = pd.concat([df, df.gpu_usage.apply(pd.Series)], axis=1).drop(
        "gpu_usage", axis=1
    )
    df["gpu_mem_used"] = df["memory_used"] / df["memory_total"] * 100
    df["inactive"] = df["gpu_mem_used"] < 1
    return df


def get_colors(df: pd.DataFrame) -> dict:
    gpu_names = set(df["gpu_name"].unique())
    colors = px.colors.qualitative.Plotly
    color_map = {gpu_name: colors[i] for i, gpu_name in enumerate(gpu_names)}
    return color_map


df = get_data()
color_map = get_colors(df)

# plot current usage, stacked bar chart per user
latest_timestamp = df["timestamp"].max()
current_df = df[df["timestamp"] == latest_timestamp].copy()
last_hour_df = df[
    (df["timestamp"] > latest_timestamp - pd.Timedelta(hours=1))
    & (df["pod_name"].isin(current_df["pod_name"].unique()))
]
last_hour_df = (
    last_hour_df.groupby(["pod_name", "gpu_id"])
    .agg(
        {
            "memory_free": "mean",
            "gpu_mem_used": "mean",
            "gpu_name": "first",
            "inactive": "all",
            "username": "first",
        }
    )
    .reset_index()
)

last_day_df = df[
    (df["timestamp"] > latest_timestamp - pd.Timedelta(days=1))
    & (df["pod_name"].isin(current_df["pod_name"].unique()))
]
last_day_df = (
    last_day_df.groupby(["pod_name", "gpu_id"])
    .agg(
        {
            "username": "first",
            "is_interactive": "any",
            "gpu_mem_used": list,
            "inactive": "all",
            "gpu_name": "first",
            "timestamp": "min",
            "cpu_requested": "last",
            "memory_requested": "last",
        }
    )
    .reset_index()
    .rename(columns={"timestamp": "first_seen"})
)
last_day_df["cpu_requested"] = last_day_df["cpu_requested"].astype(int)
last_day_df["memory_requested"] = last_day_df["memory_requested"].astype(int)

# show current global counts
gpu_counts = current_df.gpu_name.value_counts()

pending_pods = get_pending_pods()
gpu_counts["Pending"] = len(pending_pods)

cols = st.columns(len(gpu_counts) + 1)
for col, (gpu_name, count) in zip(cols[:-1], gpu_counts.items()):
    with col:
        st.metric(
            f"{gpu_name}",
            count,
            delta=None,
            delta_color="normal",
            help=f"{gpu_name}",
            label_visibility="visible",
        )

with cols[-1]:
    st.metric(
        "Total",
        sum(gpu_counts),
        delta=None,
        delta_color="normal",
        help=None,
        label_visibility="visible",
    )


count_inactive_gpus_last_hour = last_hour_df["inactive"].sum()
count_inactive_gpus_last_day = last_day_df["inactive"].sum()

for col, count_inactive, time_period in zip(
    st.columns(2),
    [
        count_inactive_gpus_last_hour,
        count_inactive_gpus_last_day,
    ],
    ["hour", "day"],
):
    with col:
        st.metric(
            f"Inactive GPUs in the last {time_period}",
            count_inactive,
            delta=None,
            delta_color="normal",
            help=f"Number of GPUs with less than 1% memory usage in the last {time_period}",
            label_visibility="visible",
        )


count_inactive_pods_last_hour = (
    last_hour_df.groupby("pod_name")
    .agg({"inactive": "all"})
    .reset_index()["inactive"]
    .sum()
)
count_inactive_pods_last_day = (
    last_day_df.groupby("pod_name")
    .agg({"inactive": "all"})
    .reset_index()["inactive"]
    .sum()
)

for col, count_inactive, time_period, help_message in zip(
    st.columns(2),
    [
        count_inactive_pods_last_hour,
        count_inactive_pods_last_day,
    ],
    ["hour", "day"],
    [
        "|".join(set(last_hour_df[last_hour_df["inactive"]].pod_name.unique())),
        "|".join(set(last_day_df[last_day_df["inactive"]].pod_name.unique())),
    ],
):
    with col:
        st.metric(
            f"Inactive Pods in the last {time_period}",
            count_inactive,
            delta=None,
            delta_color="normal",
            help=help_message,
            label_visibility="visible",
        )


st.data_editor(
    last_day_df,
    column_config={
        "gpu_mem_used": st.column_config.LineChartColumn(
            "GPU Memory Utilization (%)",
            # width="medium",
            help="Memory utilization of the GPUs in the pod over 24hours",
            y_min=0,
            y_max=100,
        ),
    },
    hide_index=True,
    use_container_width=True,
)

# average user usage in last hour
last_hour_usage_df = (
    last_hour_df.groupby(["username", "gpu_name"])
    .agg(
        {
            "gpu_name": "count",
            "inactive": "sum",
            "memory_free": "sum",
            "pod_name": list,
            "gpu_mem_used": "mean",
        }
    )
    .rename(columns={"gpu_name": "count"})
    .reset_index()
)
# count total as well
last_hour_usage_df["count_total"] = last_hour_usage_df.groupby("username")[
    "count"
].transform("sum")
last_hour_usage_df["count_total_inactive"] = last_hour_usage_df.groupby("username")[
    "inactive"
].transform("sum")

last_hour_usage_df["memory_free"] = last_hour_usage_df["memory_free"] / 1024
last_hour_usage_df["memory_free_total"] = last_hour_usage_df.groupby("username")[
    "memory_free"
].transform("sum")

last_hour_usage_df["pod_name"] = last_hour_usage_df["pod_name"].apply(
    lambda x: ", ".join(set(x))
)

# plot current usage per user
sorted_df = last_hour_usage_df.sort_values(by="count_total", ascending=False)
fig = px.bar(
    last_hour_usage_df,
    x="username",
    y="count",
    color="gpu_name",
    title="GPU usage per user (last hour)",
    color_discrete_map=color_map,
    category_orders={"username": sorted_df["username"].tolist()},
    hover_data={"pod_name": True},
    labels={"count": "Number of GPUs"},
)

st.plotly_chart(fig, use_container_width=True)

# plot current inactive GPUs per user
sorted_df = last_hour_usage_df.sort_values(by="count_total_inactive", ascending=False)
fig = px.bar(
    last_hour_usage_df[last_hour_usage_df["inactive"] > 0],
    x="username",
    y="inactive",
    title="Inactive GPUs per user (last hour)",
    color="gpu_name",
    color_discrete_map=color_map,
    category_orders={"username": sorted_df["username"].tolist()},
    hover_data={"pod_name": True},
    labels={"inactive": "Number of Inactive GPUs"},
)
st.plotly_chart(fig, use_container_width=True)


# chart of memory free per user
sorted_df = last_hour_usage_df.sort_values(by="memory_free_total", ascending=False)

fig = px.bar(
    last_hour_usage_df,
    x="username",
    y="memory_free",
    title="Total GPU Memory free per user (last hour)",
    color="gpu_name",
    color_discrete_map=color_map,
    category_orders={"username": sorted_df["username"].tolist()},
    hover_data={"pod_name": True},
    labels={"memory_free": "Memory free (GB)"},
)
st.plotly_chart(fig, use_container_width=True)


# plot average utilization rates
avg_usage_df = (
    df.groupby(["username", "gpu_name"]).agg({"gpu_mem_used": "mean"}).reset_index()
)
sorted_df = last_hour_usage_df.sort_values(by="gpu_mem_used", ascending=False)
fig = px.bar(
    last_hour_usage_df,
    x="username",
    y="gpu_mem_used",
    color="gpu_name",
    title="Average GPU memory usage per user",
    color_discrete_map=color_map,
    barmode="group",
    category_orders={"username": sorted_df["username"].tolist()},
    labels={"gpu_mem_used": "Average GPU Memory Utilization (%)"},
)
st.plotly_chart(fig, use_container_width=True)


# plot GPU usage over time per user
gpu_usage_df = (
    df.groupby(["username", "timestamp"])
    .agg({"gpu_name": "count", "inactive": "sum"})
    .reset_index()
)
fig = px.line(
    gpu_usage_df,
    x="timestamp",
    y="gpu_name",
    color="username",
    title="GPU usage over time per user",
)
st.plotly_chart(fig, use_container_width=True)

fig = px.line(
    gpu_usage_df[gpu_usage_df["inactive"] > 0],
    x="timestamp",
    y="inactive",
    color="username",
    title="Inactive GPUs over time per user",
)
st.plotly_chart(fig, use_container_width=True)
