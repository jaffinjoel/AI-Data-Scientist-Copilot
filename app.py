import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Data Scientist Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------
st.sidebar.title("🤖 AI Data Scientist Copilot")
st.sidebar.markdown("Enterprise Analytics Platform")

menu = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Dataset Upload",
        "Data Quality",
        "Cleaning",
        "EDA",
        "Machine Learning",
        "AI Insights",
        "AI Chat",
        "Reports",
        "Settings"
    ]
)
# ---------------------------------------------------
# Automatic Column Detection
# ---------------------------------------------------



# ---------------------------------------------------
# Home Page
# ---------------------------------------------------
# ---------------------------------------------------
# Home Page
# ---------------------------------------------------
if menu == "Home":

    st.title("🤖 AI Data Scientist Copilot")
    st.markdown("### Enterprise Analytics Dashboard")

    # Check if a dataset is uploaded
    if "df" not in st.session_state:
        st.info("Upload a sales dataset to begin analysis.")
        st.stop()

    df = st.session_state["df"]

    # -----------------------------
    # Automatic Revenue Column Detection
    # -----------------------------
    price_col = None
    for col in df.columns:
        if col.lower() in ["price", "total", "sales", "amount", "revenue"]:
            price_col = col
            break

    # -----------------------------
    # KPI Metrics
    # -----------------------------
    total_records = len(df)
    total_columns = len(df.columns)
    missing_values = int(df.isnull().sum().sum())

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", f"{total_records:,}")

    with col2:
        st.metric("Columns", total_columns)

    with col3:
        st.metric("Missing Values", missing_values)

    if price_col is not None:
        total_revenue = df[price_col].sum()
        avg_order_value = df[price_col].mean()

        col4, col5 = st.columns(2)

        with col4:
            st.metric("Total Revenue", f"₹{total_revenue:,.0f}")

        with col5:
            st.metric("Average Order Value", f"₹{avg_order_value:,.2f}")

    # -----------------------------
    # Dataset Preview
    # -----------------------------
    st.divider()
    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # -----------------------------
    # Executive Dashboard
    # -----------------------------
    st.divider()
    st.subheader("Executive Dashboard")

    # Detect Product / Category column
    category_col = None
    for col in df.columns:
        if col.lower() in ["product", "product line", "category", "item"]:
            category_col = col
            break

    # Revenue by Product
    if price_col is not None and category_col is not None:
        product_df = (
            df.groupby(category_col)[price_col]
            .sum()
            .reset_index()
            .sort_values(price_col, ascending=False)
        )

        fig = px.bar(
            product_df,
            x=category_col,
            y=price_col,
            title="Revenue by Product Category"
        )

        st.plotly_chart(fig, use_container_width=True)

    # Detect City / Region column
    city_col = None
    for col in df.columns:
        if col.lower() in ["city", "region", "state", "location"]:
            city_col = col
            break

    # Revenue by City
    if price_col is not None and city_col is not None:
        city_df = (
            df.groupby(city_col)[price_col]
            .sum()
            .reset_index()
            .sort_values(price_col, ascending=False)
        )

        fig = px.bar(
            city_df,
            x=city_col,
            y=price_col,
            title="Revenue by Location"
        )

        st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Quick Business Summary
    # -----------------------------
    st.divider()
    st.subheader("Quick Business Summary")

    if price_col is not None:
        st.success(
            f"Loaded **{total_records} records** with **{total_columns} columns**. "
            f"Detected **{price_col}** as the revenue column. "
            f"Total revenue is **₹{total_revenue:,.0f}** with an average order value of **₹{avg_order_value:,.2f}**."
        )
    else:
        st.warning(
            "No revenue-related column (Price, Total, Sales, Amount, or Revenue) was detected."
        )
# ---------------------------------------------------
# Dataset Upload Page
# ---------------------------------------------------
elif menu == "Dataset Upload":

    st.title("📂 Dataset Upload")
    st.write("Upload a CSV, Excel, or JSON dataset.")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "xls", "json"]
    )

    if uploaded_file is not None:

        file_name = uploaded_file.name
        file_type = file_name.split(".")[-1].lower()

        st.success(f"File uploaded successfully: {file_name}")

        # Read dataset
        if file_type == "csv":
            df = pd.read_csv(uploaded_file)

        elif file_type in ["xlsx", "xls"]:
            df = pd.read_excel(uploaded_file)

        elif file_type == "json":
            df = pd.read_json(uploaded_file)

        else:
            st.error("Unsupported file type.")
            st.stop()

        # Save dataset for other pages
        st.session_state["df"] = df

        st.subheader("Dataset Preview")
        st.dataframe(df, use_container_width=True)

        st.subheader("Dataset Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Rows", df.shape[0])

        with col2:
            st.metric("Columns", df.shape[1])

        with col3:
            memory = round(df.memory_usage(deep=True).sum() / 1024, 2)
            st.metric("Memory (KB)", memory)

        st.subheader("Column Names")
        st.write(list(df.columns))

# ---------------------------------------------------
# Data Quality Page
# ---------------------------------------------------
elif menu == "Data Quality":

    st.title("📊 Data Quality Report")

    if "df" not in st.session_state:
        st.warning("Please upload a dataset first from the Dataset Upload page.")
        st.stop()

    df = st.session_state["df"]

    total_rows = df.shape[0]
    total_columns = df.shape[1]
    missing_values = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    memory_usage = round(df.memory_usage(deep=True).sum() / 1024, 2)

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

    # Simple Data Health Score
    health_score = 100

    if missing_values > 0:
        health_score -= min(30, missing_values)

    if duplicate_rows > 0:
        health_score -= min(20, duplicate_rows)

    health_score = max(0, health_score)

    st.subheader("Overall Data Health")
    st.metric("Health Score", f"{health_score}/100")

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", total_rows)

    with col2:
        st.metric("Columns", total_columns)

    with col3:
        st.metric("Memory (KB)", memory_usage)

    st.divider()

    col4, col5 = st.columns(2)

    with col4:
        st.metric("Missing Values", missing_values)

    with col5:
        st.metric("Duplicate Rows", duplicate_rows)

    st.divider()

    st.subheader("Detected Column Types")

    col6, col7 = st.columns(2)

    with col6:
        st.write("**Numeric Columns**")
        st.write(numeric_columns)

    with col7:
        st.write("**Categorical Columns**")
        st.write(categorical_columns)

    st.divider()

    st.subheader("Detailed Missing Values")

    missing_table = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum().values,
        "Missing %": (df.isnull().sum().values / len(df) * 100).round(2)
    })

    st.dataframe(missing_table, use_container_width=True)

# ---------------------------------------------------
# ---------------------------------------------------
# Data Cleaning Page
# ---------------------------------------------------
elif menu == "Cleaning":

    st.title("🧹 Data Cleaning")

    if "df" not in st.session_state:
        st.warning("Please upload a dataset first from the Dataset Upload page.")
        st.stop()

    df = st.session_state["df"].copy()

    st.subheader("Current Dataset")

    st.dataframe(df, use_container_width=True)

    st.divider()

    st.subheader("Cleaning Options")

    remove_duplicates = st.checkbox("Remove Duplicate Rows")

    fill_missing = st.checkbox("Fill Missing Values Automatically")

    drop_missing = st.checkbox("Drop Rows Containing Missing Values")

    if st.button("Run Data Cleaning"):

        original_rows = df.shape[0]

        summary = []

        if remove_duplicates:

            before = df.shape[0]

            df = df.drop_duplicates()

            removed = before - df.shape[0]

            summary.append(f"Removed {removed} duplicate rows")

        if fill_missing:

            for column in df.columns:

                if df[column].dtype in ["int64", "float64"]:

                    df[column] = df[column].fillna(df[column].median())

                else:

                    df[column] = df[column].fillna(df[column].mode()[0])

            summary.append("Filled missing values automatically")

        if drop_missing:

            before = df.shape[0]

            df = df.dropna()

            removed = before - df.shape[0]

            summary.append(f"Dropped {removed} rows with missing values")

        st.session_state["cleaned_df"] = df

        st.success("Data cleaning completed successfully!")

        st.subheader("Cleaning Summary")

        if summary:

            for item in summary:

                st.write(f"• {item}")

        else:

            st.write("No cleaning operation was selected.")

        st.divider()

        st.subheader("Cleaned Dataset")

        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(

            label="📥 Download Cleaned Dataset",

            data=csv,

            file_name="cleaned_dataset.csv",

            mime="text/csv"

        )

# ---------------------------------------------------
# Other Pages
# ---------------------------------------------------
# ---------------------------------------------------
# EDA Page
# ---------------------------------------------------
elif menu == "EDA":

    st.title("📈 Exploratory Data Analysis")

    if "df" not in st.session_state:
        st.warning("Please upload a dataset first from the Dataset Upload page.")
        st.stop()

    df = st.session_state["df"]

    st.subheader("Dataset Preview")
    st.dataframe(df, use_container_width=True)

    st.divider()

    st.subheader("Statistical Summary")
    st.dataframe(df.describe(include="all"), use_container_width=True)

    st.divider()

    numeric_columns = [
    col for col in df.select_dtypes(include=["number"]).columns
    if "id" not in col.lower()
]
    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

    # ---------------- Histograms ----------------
    st.subheader("Numeric Feature Distributions")

    for column in numeric_columns:
        fig = px.histogram(
            df,
            x=column,
            nbins=10,
            title=f"Distribution of {column}"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---------------- Boxplots ----------------
    st.subheader("Outlier Detection (Boxplots)")

    for column in numeric_columns:
        fig = px.box(
            df,
            y=column,
            title=f"Boxplot of {column}"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---------------- Correlation Heatmap ----------------
    if len(numeric_columns) > 1:

        st.subheader("Correlation Heatmap")

        corr = df[numeric_columns].corr()

        heatmap = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Feature Correlation Matrix"
        )

        st.plotly_chart(heatmap, use_container_width=True)

    st.divider()

    # ---------------- Category Distributions ----------------
    st.subheader("Categorical Feature Distributions")

    for column in categorical_columns:

        counts = df[column].value_counts().reset_index()
        counts.columns = [column, "Count"]

        fig = px.bar(
            counts,
            x=column,
            y="Count",
            title=f"Distribution of {column}"
        )

        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# ---------------------------------------------------
# # ---------------------------------------------------
# ---------------------------------------------------
# ---------------------------------------------------
# ---------------------------------------------------

# ---------------------------------------------------
# Machine Learning Page
# ---------------------------------------------------
elif menu == "Machine Learning":

    st.title("🤖 Machine Learning")

    if "df" not in st.session_state:
        st.warning("Please upload a dataset first.")
        st.stop()

    df = st.session_state["df"]

    target = "Price"

    if target not in df.columns:
        st.error("Target column 'Price' not found in the dataset.")
        st.stop()

    st.subheader("Regression Prediction")
    st.write(f"Target Variable: **{target}**")

    if st.button("Train Model"):

        X = df.drop(columns=[target])
        y = df[target]

        numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
        categorical_features = X.select_dtypes(include=["object"]).columns

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median"))
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features)
            ]
        )

        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", RandomForestRegressor(
                    n_estimators=100,
                    random_state=42
                ))
            ]
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)

        st.success("Model trained successfully!")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("R² Score", f"{r2:.2f}")

        with col2:
            st.metric("MAE", f"₹{mae:,.0f}")

        result_df = pd.DataFrame({
            "Actual Price": y_test.values,
            "Predicted Price": predictions.round(0).astype(int)
        })

        st.subheader("Prediction Results")
        st.dataframe(result_df, use_container_width=True)

        st.subheader("Model Summary")
        st.info(
            "The Random Forest model has been trained using both numerical and categorical features with automatic preprocessing."
        )


# ---------------------------------------------------
# AI Insights Page
# ---------------------------------------------------
elif menu == "AI Insights":

    st.title("🧠 AI Business Insights")

    if "df" not in st.session_state:
        st.warning("Please upload a dataset first.")
        st.stop()

    df = st.session_state["df"]

    if "Price" not in df.columns:
        st.error("This page requires a 'Price' column in the dataset.")
        st.stop()

    total_orders = len(df)
    total_revenue = df["Price"].sum()
    avg_order_value = df["Price"].mean()

    st.success(f"""
### AI Generated Executive Summary

- **Total Orders:** {total_orders}
- **Total Revenue:** ₹{total_revenue:,.0f}
- **Average Order Value:** ₹{avg_order_value:,.2f}
""")

    if "Product" in df.columns:
        st.subheader("Revenue by Product")

        product_revenue = (
            df.groupby("Product")["Price"]
            .sum()
            .reset_index()
        )

        fig = px.bar(
            product_revenue,
            x="Product",
            y="Price",
            title="Revenue by Product"
        )

        st.plotly_chart(fig, use_container_width=True)

    if "City" in df.columns:
        st.subheader("Revenue by City")

        city_revenue = (
            df.groupby("City")["Price"]
            .sum()
            .reset_index()
        )

        fig = px.bar(
            city_revenue,
            x="City",
            y="Price",
            title="Revenue by City"
        )

        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------
# AI Chat Page
# ---------------------------------------------------
elif menu == "AI Chat":

    st.title("💬 AI Dataset Assistant")

    if "df" not in st.session_state:
        st.warning("Please upload a dataset first.")
        st.stop()

    df = st.session_state["df"]

    # ---------- Automatic Column Detection ----------
    price_col = None
    for c in df.columns:
        if c.lower() in ["price", "sales", "revenue", "amount", "total"]:
            price_col = c
            break

    product_col = None
    for c in df.columns:
        if "product" in c.lower() or "category" in c.lower() or "line" in c.lower():
            product_col = c
            break

    city_col = None
    for c in df.columns:
        if "city" in c.lower() or "branch" in c.lower() or "location" in c.lower():
            city_col = c
            break

    customer_col = None
    for c in df.columns:
        if "customer" in c.lower() or "client" in c.lower() or "member" in c.lower():
            customer_col = c
            break

    st.write("Ask a question about your uploaded dataset.")

    question = st.text_input(
        "Ask a question",
        placeholder="Example: Which city has the highest revenue?"
    )

    if st.button("Get Answer", key="ai_chat_btn"):

        q = question.lower()

        if price_col is None:
            st.error("No sales/price column detected in the uploaded dataset.")
            st.stop()

        # Total Revenue
        if any(word in q for word in ["total revenue", "overall revenue", "total sales", "sales amount"]):
            total = df[price_col].sum()
            st.success(f"Total revenue is ₹{total:,.0f}")

        # Highest Revenue City
        elif city_col and any(word in q for word in ["highest revenue city", "top city", "best city", "which city"]):
            city = df.groupby(city_col)[price_col].sum().idxmax()
            value = df.groupby(city_col)[price_col].sum().max()
            st.success(f"{city} has the highest revenue: ₹{value:,.0f}")

        # Highest Revenue Product
        elif product_col and any(word in q for word in ["highest revenue product", "top product", "best product", "which product"]):
            product = df.groupby(product_col)[price_col].sum().idxmax()
            value = df.groupby(product_col)[price_col].sum().max()
            st.success(f"{product} generates the highest revenue: ₹{value:,.0f}")

        # Highest Spending Customer
        elif customer_col and any(word in q for word in ["top customer", "highest spending customer", "best customer", "who has highest revenue", "who spent the most"]):
            customer = df.groupby(customer_col)[price_col].sum().idxmax()
            value = df.groupby(customer_col)[price_col].sum().max()
            st.success(f"{customer} is the highest spending customer: ₹{value:,.0f}")

        # Average Order Value
        elif any(word in q for word in ["average order value", "average price", "average revenue", "average sales"]):
            avg = df[price_col].mean()
            st.success(f"Average order value is ₹{avg:,.2f}")

        # Revenue by City
        elif city_col and "revenue by city" in q:
            table = df.groupby(city_col)[price_col].sum().reset_index()
            st.dataframe(table, use_container_width=True)

        # Revenue by Product
        elif product_col and "revenue by product" in q:
            table = df.groupby(product_col)[price_col].sum().reset_index()
            st.dataframe(table, use_container_width=True)

        # Dataset Summary
        elif "summary" in q or "dataset summary" in q:
            st.write(df.describe(include="all"))

        else:
            st.info(
                "I can answer questions about revenue, products, cities, customers, average order value, and dataset summaries."
            )

# ---------------------------------------------------
# ---------------------------------------------------
# Reports Page
# ---------------------------------------------------
elif menu == "Reports":

    st.title("📄 Reports")

    if "df" not in st.session_state:
        st.warning("Upload a dataset first.")
        st.stop()

    df = st.session_state["df"]

    report = pd.DataFrame({
        "Metric": [
            "Rows",
            "Columns",
            "Missing Values",
            "Numeric Columns"
        ],
        "Value": [
            len(df),
            len(df.columns),
            int(df.isnull().sum().sum()),
            len(df.select_dtypes(include="number").columns)
        ]
    })

    st.dataframe(report, use_container_width=True)

    csv = report.to_csv(index=False)

    st.download_button(
        "⬇ Download Report",
        data=csv,
        file_name="analytics_report.csv",
        mime="text/csv"
    )

# ---------------------------------------------------
# Settings Page
# ---------------------------------------------------
elif menu == "Settings":

    st.title("⚙ Settings")
    st.write("Configuration options will be added in future versions.")

# ---------------------------------------------------
# Other Pages
# ---------------------------------------------------
else:

    st.title(menu)
    st.write("This module will be implemented in upcoming phases.")