import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="Data Analytics Dashboard",
    layout="wide"
)

# ---------------------------------------------------
# TITLE AND DESCRIPTION
# ---------------------------------------------------

st.title("Interactive Data Analytics Dashboard")

st.markdown("""
This dashboard allows users to:

- Upload CSV files
- Analyze datasets
- Explore missing values
- Generate visualizations
- Detect correlations
- Filter and search data
- Download processed datasets
""")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("Dashboard Menu")

st.sidebar.info(
    "Upload a CSV file and explore the dataset interactively."
)

uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV File",
    type=["csv"]
)

# ---------------------------------------------------
# MAIN DASHBOARD
# ---------------------------------------------------

if uploaded_file is not None:

    # READ CSV FILE
    df = pd.read_csv(uploaded_file)

    # ---------------------------------------------------
    # DATASET PREVIEW
    # ---------------------------------------------------

    st.header("Dataset Preview")

    st.dataframe(df.head())

    # ---------------------------------------------------
    # LAST 5 ROWS
    # ---------------------------------------------------

    st.header("Last 5 Rows")

    st.dataframe(df.tail())

    # ---------------------------------------------------
    # RANDOM SAMPLE
    # ---------------------------------------------------

    st.header("Random Sample Data")

    st.dataframe(df.sample(5))

    # ---------------------------------------------------
    # DATASET SHAPE
    # ---------------------------------------------------

    st.header("Dataset Shape")

    rows, columns = df.shape

    col1, col2 = st.columns(2)

    col1.metric("Rows", rows)

    col2.metric("Columns", columns)

    # ---------------------------------------------------
    # COLUMN NAMES
    # ---------------------------------------------------

    st.header("Column Names")

    st.write(df.columns.tolist())

    # ---------------------------------------------------
    # DATA TYPES
    # ---------------------------------------------------

    st.header("Data Types")

    st.write(df.dtypes)

    # ---------------------------------------------------
    # MISSING VALUES
    # ---------------------------------------------------

    st.header("Missing Values")

    st.write(df.isnull().sum())

    # ---------------------------------------------------
    # MISSING VALUE PERCENTAGE
    # ---------------------------------------------------

    st.header("Missing Value Percentage")

    missing_percentage = (
        df.isnull().sum() / len(df)
    ) * 100

    st.write(missing_percentage)

    # ---------------------------------------------------
    # SUMMARY STATISTICS
    # ---------------------------------------------------

    st.header("Summary Statistics")

    st.write(df.describe())

    # ---------------------------------------------------
    # NUMERIC COLUMNS
    # ---------------------------------------------------

    numeric_columns = df.select_dtypes(
        include=['number']
    ).columns

    # ---------------------------------------------------
    # HISTOGRAM
    # ---------------------------------------------------

    st.header("Histogram")

    hist_column = st.selectbox(
        "Select Column for Histogram",
        numeric_columns
    )

    fig1, ax1 = plt.subplots()

    ax1.hist(df[hist_column].dropna())

    ax1.set_title(f"Histogram of {hist_column}")

    st.pyplot(fig1)

    # ---------------------------------------------------
    # SCATTER PLOT
    # ---------------------------------------------------

    st.header("Scatter Plot")

    x_axis = st.selectbox(
        "Select X-axis",
        numeric_columns,
        key="xaxis"
    )

    y_axis = st.selectbox(
        "Select Y-axis",
        numeric_columns,
        key="yaxis"
    )

    fig2, ax2 = plt.subplots()

    ax2.scatter(df[x_axis], df[y_axis])

    ax2.set_xlabel(x_axis)

    ax2.set_ylabel(y_axis)

    ax2.set_title(f"{x_axis} vs {y_axis}")

    st.pyplot(fig2)

    # ---------------------------------------------------
    # LINE CHART
    # ---------------------------------------------------

    st.header("Line Chart")

    line_column = st.selectbox(
        "Select Column for Line Chart",
        numeric_columns,
        key="linechart"
    )

    fig3, ax3 = plt.subplots()

    ax3.plot(df[line_column])

    ax3.set_title(f"Line Chart of {line_column}")

    st.pyplot(fig3)

    # ---------------------------------------------------
    # BOX PLOT
    # ---------------------------------------------------

    st.header("Box Plot")

    box_column = st.selectbox(
        "Select Column for Box Plot",
        numeric_columns,
        key="boxplot"
    )

    fig4, ax4 = plt.subplots()

    sns.boxplot(y=df[box_column], ax=ax4)

    st.pyplot(fig4)

    # ---------------------------------------------------
    # CORRELATION HEATMAP
    # ---------------------------------------------------

    st.header("Correlation Heatmap")

    correlation = df[numeric_columns].corr()

    fig5, ax5 = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        ax=ax5
    )

    st.pyplot(fig5)

    # ---------------------------------------------------
    # CORRELATION TABLE
    # ---------------------------------------------------

    st.header("Correlation Table")

    st.dataframe(correlation)

    # ---------------------------------------------------
    # CATEGORICAL COLUMNS
    # ---------------------------------------------------

    categorical_columns = df.select_dtypes(
        include=['object']
    ).columns

    # ---------------------------------------------------
    # VALUE COUNTS
    # ---------------------------------------------------

    st.header("Categorical Value Counts")

    if len(categorical_columns) > 0:

        cat_column = st.selectbox(
            "Select Categorical Column",
            categorical_columns
        )

        st.write(df[cat_column].value_counts())

    # ---------------------------------------------------
    # PIE CHART
    # ---------------------------------------------------

    st.header("Pie Chart")

    if len(categorical_columns) > 0:

        pie_column = st.selectbox(
            "Select Column for Pie Chart",
            categorical_columns,
            key="piechart"
        )

        pie_data = df[pie_column].value_counts().head(5)

        fig6, ax6 = plt.subplots()

        ax6.pie(
            pie_data,
            labels=pie_data.index,
            autopct='%1.1f%%'
        )

        st.pyplot(fig6)

    # ---------------------------------------------------
    # FILTER DATASET
    # ---------------------------------------------------

    st.header("Filter Dataset")

    filter_column = st.selectbox(
        "Select Column",
        df.columns
    )

    filter_value = st.text_input(
        "Enter Value to Filter"
    )

    if filter_value:

        filtered_df = df[
            df[filter_column]
            .astype(str)
            .str.contains(filter_value)
        ]

        st.dataframe(filtered_df)

    # ---------------------------------------------------
    # SEARCH DATASET
    # ---------------------------------------------------

    st.header("Search Dataset")

    search_term = st.text_input(
        "Search Any Value"
    )

    if search_term:

        search_df = df[
            df.astype(str)
            .apply(
                lambda row:
                row.str.contains(search_term).any(),
                axis=1
            )
        ]

        st.dataframe(search_df)

    # ---------------------------------------------------
    # DOWNLOAD DATASET
    # ---------------------------------------------------

    st.header("Download Dataset")

    csv = df.to_csv(index=False)

    st.download_button(
        label="Download Dataset",
        data=csv,
        file_name="processed_dataset.csv",
        mime="text/csv"
    )

    # ---------------------------------------------------
    # FOOTER
    # ---------------------------------------------------

    st.markdown("---")

    st.markdown(
        "Developed by Kantipudi Sahithi | Synent Technologies Internship"
    )

# ---------------------------------------------------
# IF NO FILE UPLOADED
# ---------------------------------------------------

else:

    st.info(
        "Please upload a CSV file to begin analysis."
    )