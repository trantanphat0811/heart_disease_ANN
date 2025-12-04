"""Streamlit dashboard using Plotly for interactive visualizations."""
import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go


@st.cache_data
def load_data(path: str = "data/synthetic_heart_disease_dataset.csv"):
    return pd.read_csv(path)


def main():
    st.set_page_config(layout="wide", page_title="Heart Disease Plotly Dashboard")
    st.title("Heart Disease — Interactive Dashboard (Plotly)")

    data_path = Path("data/synthetic_heart_disease_dataset.csv")
    df = load_data(data_path)

    # quick derived features
    if "Age" in df.columns:
        bins = [0, 30, 45, 60, 75, 200]
        labels = ["<30", "30-44", "45-59", "60-74", "75+"]
        df["Age_Group"] = pd.cut(df["Age"], bins=bins, labels=labels)

    st.sidebar.header("Filters")
    genders = df["Gender"].unique().tolist() if "Gender" in df.columns else []
    gender_sel = st.sidebar.multiselect("Gender", options=genders, default=genders)
    age_groups = df["Age_Group"].unique().tolist() if "Age_Group" in df.columns else []
    age_sel = st.sidebar.multiselect("Age Group", options=age_groups, default=age_groups)

    # filter
    df_filt = df.copy()
    if genders:
        df_filt = df_filt[df_filt["Gender"].isin(gender_sel)]
    if age_groups:
        df_filt = df_filt[df_filt["Age_Group"].isin(age_sel)]

    st.markdown("## Summary stats")
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if "Heart_Disease" in df_filt.columns:
            counts = df_filt["Heart_Disease"].value_counts().rename_axis("label").reset_index(name="count")
            fig = px.pie(counts, names="label", values="count", title="Heart Disease distribution")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.write("Rows (filtered):", len(df_filt))
        if "Age" in df_filt.columns:
            st.write("Age mean:", f"{df_filt['Age'].mean():.1f}")
        if "BMI" in df_filt.columns:
            st.write("BMI mean:", f"{df_filt['BMI'].mean():.1f}")

    with col3:
        numeric_cols = df_filt.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            corr = df_filt[numeric_cols].corr()
            fig = px.imshow(corr, title="Numeric Feature Correlation", color_continuous_scale="RdBu_r")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("## Exploratory plots")
    # Histogram
    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("Histogram: Select numeric column", options=numeric_cols, index=0)
        fig = px.histogram(df_filt, x=x_col, nbins=40, color="Heart_Disease" if "Heart_Disease" in df_filt.columns else None, barmode="overlay", title=f"Histogram of {x_col}")
        st.plotly_chart(fig, use_container_width=True)

    # Scatter
    with col2:
        scatter_x = st.selectbox("Scatter X", options=numeric_cols, index=0, key="sx")
        scatter_y = st.selectbox("Scatter Y", options=numeric_cols, index=1, key="sy")
        fig = px.scatter(df_filt, x=scatter_x, y=scatter_y, color="Heart_Disease" if "Heart_Disease" in df_filt.columns else None, title=f"{scatter_y} vs {scatter_x}", opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("## Box plots by group")
    group_col = st.selectbox("Group by (categorical)", options=[c for c in df_filt.columns if df_filt[c].dtype == object or str(df_filt[c].dtype).startswith('category')], index=0)
    box_num = st.selectbox("Numeric for boxplot", options=numeric_cols, index=0, key="boxnum")
    fig = px.box(df_filt, x=group_col, y=box_num, color=group_col, title=f"{box_num} by {group_col}")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("## Cohort table (top groups)")
    if "Heart_Disease" in df_filt.columns and group_col:
        cohort = df_filt.groupby(group_col)["Heart_Disease"].agg(["count", "mean"]).rename(columns={"mean": "rate"}).sort_values("count", ascending=False).reset_index()
        st.dataframe(cohort)


if __name__ == "__main__":
    main()
