"""
RootIQ: Evidence-based root-cause analysis platform.
Turns messy business data into: Data → Evidence → Problem → Root-Cause → Recommendation.
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime

# ============================================================================
# PAGE CONFIG & SESSION STATE INITIALIZATION
# ============================================================================

st.set_page_config(
    page_title="RootIQ",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 RootIQ")
st.markdown("**AI that finds the 'why' behind your business data.**")

# Initialize session state
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "results" not in st.session_state:
    st.session_state.results = None
if "df" not in st.session_state:
    st.session_state.df = None
if "profile" not in st.session_state:
    st.session_state.profile = None

# ============================================================================
# ERROR HANDLING & CUSTOM EXCEPTIONS
# ============================================================================

class DataLoadError(Exception):
    """Custom exception for data loading errors."""
    pass

# ============================================================================
# DATA LOADING FUNCTION (Section 3)
# ============================================================================

def load_data(uploaded_file):
    """Load CSV, XLSX, or JSON file. Raise DataLoadError on failure."""
    try:
        if uploaded_file is None:
            raise DataLoadError("No file uploaded.")
        
        filename = uploaded_file.name.lower()
        
        # CSV
        if filename.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            return df, 'csv'
        
        # XLSX
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            return df, 'xlsx'
        
        # JSON
        elif filename.endswith('.json'):
            df = pd.read_json(uploaded_file)
            return df, 'json'
        
        else:
            raise DataLoadError(f"Unsupported file type: {filename}. Use CSV, XLSX, or JSON.")
    
    except Exception as e:
        raise DataLoadError(f"Failed to load file: {str(e)}")

# ============================================================================
# DATA PROFILING FUNCTION (Section 3)
# ============================================================================

def profile_dataset(df):
    """Compute dataset profile: rows, columns, dtypes, missing values, etc."""
    profile = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": df.duplicated().sum(),
        "unique_counts": {col: df[col].nunique() for col in df.columns},
        "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
        "datetime_columns": df.select_dtypes(include=['datetime64']).columns.tolist(),
    }
    return profile

# ============================================================================
# COLUMN DETECTION FUNCTION (Section 3)
# ============================================================================

def normalize_column_name(name):
    """Normalize column name for matching: lowercase, strip punctuation, collapse whitespace."""
    return name.lower().strip().replace('_', '').replace(' ', '').replace('-', '')

def detect_columns(df):
    """Detect business-relevant columns using synonym matching."""
    
    # Synonym table from spec
    synonym_map = {
        "date": ["date", "orderdate", "transactiondate", "purchasedate", "month", "week", "day", "period", "timestamp", "createdat"],
        "revenue": ["revenue", "sales", "totalsales", "grossrevenue", "netsales", "income", "amount", "totalamount"],
        "cost": ["cost", "expense", "expenses", "cogs", "expenditure", "spend", "totalcost"],
        "profit": ["profit", "netprofit", "margin", "netincome", "grossprofit"],
        "orders": ["orders", "ordercount", "quantity", "qty", "units", "unitssold", "transactions", "numorders"],
        "customer": ["customer", "customerid", "client", "clientid", "buyer", "userid"],
        "product": ["product", "productname", "sku", "item", "itemname"],
        "category": ["category", "productcategory", "segment", "region", "location", "channel", "country", "state", "city", "market"],
        "marketing_spend": ["adspend", "marketingspend", "advertising", "adcost", "campaignspend", "marketingcost"],
        "conversion": ["conversionrate", "cvr", "conversion"],
        "traffic": ["visits", "websitevisits", "sessions", "traffic", "pageviews", "impressions"],
    }
    
    column_map = {}
    normalized_cols = {normalize_column_name(col): col for col in df.columns}
    
    for role, synonyms in synonym_map.items():
        best_match = None
        best_confidence = "Low"
        
        for synonym in synonyms:
            norm_synonym = normalize_column_name(synonym)
            for norm_col, orig_col in normalized_cols.items():
                if norm_synonym in norm_col or norm_col in norm_synonym:
                    # Check dtype consistency
                    col_dtype = df[orig_col].dtype
                    if role in ["revenue", "cost", "profit", "orders", "marketing_spend", "traffic", "conversion"]:
                        if pd.api.types.is_numeric_dtype(col_dtype):
                            best_match = orig_col
                            best_confidence = "High" if norm_synonym == norm_col else "Medium"
                            break
                    else:
                        best_match = orig_col
                        best_confidence = "High" if norm_synonym == norm_col else "Medium"
                        break
            if best_match:
                break
        
        if best_match:
            column_map[role] = {"column": best_match, "confidence": best_confidence}
    
    return column_map

# ============================================================================
# DESCRIPTIVE STATS (Section 4.1)
# ============================================================================

def compute_descriptive_stats(df, column_map):
    """Compute mean, median, min, max, std for numeric columns."""
    stats = {}
    
    for role, col_info in column_map.items():
        if role in ["revenue", "cost", "profit", "orders", "marketing_spend", "traffic", "conversion"]:
            col = col_info["column"]
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                numeric_data = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(numeric_data) > 0:
                    stats[col] = {
                        "mean": float(numeric_data.mean()),
                        "median": float(numeric_data.median()),
                        "min": float(numeric_data.min()),
                        "max": float(numeric_data.max()),
                        "std": float(numeric_data.std()),
                    }
    
    return stats

# ============================================================================
# TIME-SERIES ANALYSIS (Section 4.2)
# ============================================================================

def compute_time_series_analysis(df, column_map):
    """Time-series analysis if date column detected."""
    if "date" not in column_map:
        return None
    
    date_col = column_map["date"]["column"]
    if date_col not in df.columns:
        return None
    
    # Try to parse as datetime
    try:
        df_ts = df.copy()
        df_ts[date_col] = pd.to_datetime(df_ts[date_col], errors='coerce')
        df_ts = df_ts.dropna(subset=[date_col])
        df_ts = df_ts.sort_values(by=date_col)
        
        if len(df_ts) < 2:
            return None
        
        # Resample to monthly (or best fit)
        df_ts.set_index(date_col, inplace=True)
        
        ts_analysis = {
            "date_range": f"{df_ts.index.min().date()} to {df_ts.index.max().date()}",
            "total_periods": len(df_ts),
            "metrics": {}
        }
        
        # Compute growth rates for numeric columns
        for col in df_ts.select_dtypes(include=['number']).columns:
            numeric_series = pd.to_numeric(df_ts[col], errors='coerce').dropna()
            if len(numeric_series) > 1:
                pct_change = numeric_series.pct_change()
                ts_analysis["metrics"][col] = {
                    "periods": len(numeric_series),
                    "avg_growth": float(pct_change.mean()) if len(pct_change) > 0 else 0,
                    "trend": "up" if pct_change.mean() > 0 else "down"
                }
        
        return ts_analysis
    
    except Exception as e:
        return None

# ============================================================================
# ANOMALY DETECTION (Section 4.3)
# ============================================================================

def detect_anomalies(df, column_map):
    """Detect anomalies using IQR or rolling z-score."""
    anomalies = []
    
    for role, col_info in column_map.items():
        if role in ["revenue", "cost", "orders", "traffic"]:
            col = col_info["column"]
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                numeric_data = pd.to_numeric(df[col], errors='coerce').dropna()
                
                if len(numeric_data) >= 5:
                    # IQR method
                    Q1 = numeric_data.quantile(0.25)
                    Q3 = numeric_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = (numeric_data < lower_bound) | (numeric_data > upper_bound)
                    
                    if outliers.any():
                        for idx, is_outlier in outliers.items():
                            if is_outlier:
                                anomalies.append({
                                    "column": col,
                                    "index": int(idx) if isinstance(idx, (int, pd.Int64Dtype)) else str(idx),
                                    "value": float(numeric_data[idx]),
                                    "expected_range": [float(lower_bound), float(upper_bound)],
                                    "method": "IQR"
                                })
    
    return anomalies[:10]  # Top 10 anomalies

# ============================================================================
# CORRELATION ANALYSIS (Section 4.4)
# ============================================================================

def compute_correlations(df, column_map):
    """Correlation matrix for numeric business columns."""
    numeric_cols = [col_info["column"] for col_info in column_map.values() 
                    if col_info["column"] in df.select_dtypes(include=['number']).columns]
    
    if len(numeric_cols) < 2:
        return []
    
    corr_matrix = df[numeric_cols].corr()
    correlations = []
    
    # Extract top correlations
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.3:
                correlations.append({
                    "pair": [corr_matrix.columns[i], corr_matrix.columns[j]],
                    "r": float(r),
                    "type": "correlation"
                })
    
    # Sort by abs correlation
    correlations.sort(key=lambda x: abs(x["r"]), reverse=True)
    return correlations[:5]  # Top 5

# ============================================================================
# SEGMENTATION (Section 4.5)
# ============================================================================

def compute_segmentation(df, column_map):
    """Segment by category/region/product and find top/bottom performers."""
    if "category" not in column_map:
        return None
    
    category_col = column_map["category"]["column"]
    revenue_col = column_map.get("revenue", {}).get("column")
    
    if category_col not in df.columns or not revenue_col or revenue_col not in df.columns:
        return None
    
    try:
        segmentation = df.groupby(category_col)[revenue_col].agg(['sum', 'mean', 'count'])
        segmentation = segmentation.sort_values('sum', ascending=False)
        
        return {
            "top_performers": segmentation.head(3).index.tolist(),
            "bottom_performers": segmentation.tail(3).index.tolist(),
        }
    except Exception as e:
        return None

# ============================================================================
# SIDEBAR: USER INPUTS
# ============================================================================

st.sidebar.header("📤 Upload & Configure")

uploaded_file = st.sidebar.file_uploader(
    "Upload your data (CSV, XLSX, JSON)",
    type=["csv", "xlsx", "json"]
)

st.sidebar.divider()

business_type = st.sidebar.text_input(
    "Business Type",
    placeholder="e.g., Online clothing store",
)

dataset_description = st.sidebar.text_area(
    "Dataset Description",
    placeholder="e.g., Monthly sales, orders, ad spend",
)

st.sidebar.divider()

with st.sidebar.expander("⚙️ Advanced Options", expanded=False):
    date_column_override = st.text_input("Date Column (optional)", placeholder="e.g., order_date")
    target_metric_override = st.text_input("Target Metric (optional)", placeholder="e.g., revenue")

st.sidebar.divider()

ai_provider = st.sidebar.selectbox(
    "AI Provider",
    options=["Groq (Recommended)", "Gemini (Fallback)"],
    index=0,
)

st.sidebar.divider()

analyze_button = st.sidebar.button("🔬 Analyze", use_container_width=True, type="primary")

# ============================================================================
# MAIN PAGE LOGIC
# ============================================================================

if not uploaded_file:
    st.info("👆 Upload a CSV, XLSX, or JSON file to begin analysis.")
    st.stop()

if not business_type or not dataset_description:
    st.warning("⚠️ Please fill in Business Type and Dataset Description.")
    st.stop()

# Load and profile data
try:
    df, file_type = load_data(uploaded_file)
    st.session_state.df = df
    profile = profile_dataset(df)
    st.session_state.profile = profile
except DataLoadError as e:
    st.error(f"❌ {str(e)}")
    st.stop()

# Detect columns
column_map = detect_columns(df)

if not column_map:
    st.error("❌ No business-relevant columns detected. Check your data.")
    st.stop()

# ============================================================================
# DISPLAY DATA OVERVIEW
# ============================================================================

st.header("📊 Data Overview")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Rows", profile["row_count"])
with col2:
    st.metric("Columns", profile["column_count"])
with col3:
    st.metric("Duplicates", profile["duplicate_rows"])
with col4:
    st.metric("Missing Values", sum(profile["missing_values"].values()))

st.divider()

# Display detected columns
st.subheader("🎯 Detected Business Columns")
for role, col_info in column_map.items():
    confidence_color = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
    st.write(f"{confidence_color[col_info['confidence']]} **{role.title()}**: {col_info['column']}")

st.divider()

# ============================================================================
# ANALYSIS BUTTON
# ============================================================================

if analyze_button:
    with st.spinner("🔄 Analyzing data..."):
        
        # Compute analytics
        stats = compute_descriptive_stats(df, column_map)
        ts_analysis = compute_time_series_analysis(df, column_map)
        anomalies = detect_anomalies(df, column_map)
        correlations = compute_correlations(df, column_map)
        segmentation = compute_segmentation(df, column_map)
        
        st.session_state.analysis_complete = True
        st.session_state.results = {
            "stats": stats,
            "ts_analysis": ts_analysis,
            "anomalies": anomalies,
            "correlations": correlations,
            "segmentation": segmentation,
        }
    
    st.success("✅ Analysis complete!")

# ============================================================================
# DISPLAY RESULTS
# ============================================================================

if st.session_state.analysis_complete:
    
    results = st.session_state.results
    
    # Descriptive Stats
    st.header("📈 Descriptive Statistics")
    if results["stats"]:
        stats_df = pd.DataFrame(results["stats"]).T
        st.dataframe(stats_df, use_container_width=True)
    else:
        st.info("No numeric data to analyze.")
    
    st.divider()
    
    # Time Series
    if results["ts_analysis"]:
        st.header("📊 Time Series Analysis")
        st.write(f"**Date Range**: {results['ts_analysis']['date_range']}")
        st.write(f"**Total Periods**: {results['ts_analysis']['total_periods']}")
        
        if results["ts_analysis"]["metrics"]:
            metrics_df = pd.DataFrame(results["ts_analysis"]["metrics"]).T
            st.dataframe(metrics_df, use_container_width=True)
    
    st.divider()
    
    # Anomalies
    if results["anomalies"]:
        st.header("⚠️ Detected Anomalies")
        anomalies_df = pd.DataFrame(results["anomalies"])
        st.dataframe(anomalies_df, use_container_width=True)
    else:
        st.info("No anomalies detected.")
    
    st.divider()
    
    # Correlations
    if results["correlations"]:
        st.header("🔗 Correlations")
        corr_df = pd.DataFrame(results["correlations"])
        st.dataframe(corr_df, use_container_width=True)
    else:
        st.info("No significant correlations found.")
    
    st.divider()
    
    # Segmentation
    if results["segmentation"]:
        st.header("📊 Segmentation Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Top Performers**")
            for i, perf in enumerate(results["segmentation"]["top_performers"], 1):
                st.write(f"{i}. {perf}")
        with col2:
            st.write("**Bottom Performers**")
            for i, perf in enumerate(results["segmentation"]["bottom_performers"], 1):
                st.write(f"{i}. {perf}")
    
    st.divider()
    
    # Sample data
    st.header("🔍 Sample Data")
    st.dataframe(df.head(10), use_container_width=True)
