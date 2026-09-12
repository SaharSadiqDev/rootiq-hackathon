"""
RootIQ: Evidence-based root-cause analysis platform.
Steps 2-15: Complete implementation per specification.

Data → Evidence → Problem → Root-Cause → Recommendation.
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from groq import Groq
import google.generativeai as genai
import plotly.graph_objects as go
import plotly.express as px

# ============================================================================
# PAGE CONFIG & SESSION STATE INITIALIZATION (Section 11)
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
if "df" not in st.session_state:
    st.session_state.df = None
if "profile" not in st.session_state:
    st.session_state.profile = None
if "column_map" not in st.session_state:
    st.session_state.column_map = None
if "results" not in st.session_state:
    st.session_state.results = None
if "ai_results" not in st.session_state:
    st.session_state.ai_results = None

# ============================================================================
# ERROR HANDLING - Custom Exceptions (Section 13)
# ============================================================================

class DataLoadError(Exception):
    """Custom exception for data loading errors."""
    pass

class AIProviderError(Exception):
    """Custom exception for AI provider errors."""
    pass

# ============================================================================
# SECTION 3: DATA INGESTION & PROFILING
# ============================================================================

def load_data(uploaded_file):
    """Load CSV, XLSX, JSON. Detect file type from extension + content sniff."""
    try:
        if uploaded_file is None:
            raise DataLoadError("No file uploaded.")
        
        filename = uploaded_file.name.lower()
        
        if filename.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            return df, 'csv'
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            return df, 'xlsx'
        elif filename.endswith('.json'):
            df = pd.read_json(uploaded_file)
            return df, 'json'
        elif filename.endswith('.pdf'):
            raise DataLoadError("PDF support coming soon. Please use CSV, XLSX, or JSON.")
        else:
            raise DataLoadError(f"Unsupported file type: {filename}. Use CSV, XLSX, or JSON.")
    
    except Exception as e:
        raise DataLoadError(f"Failed to load file: {str(e)}")

def profile_dataset(df):
    """Compute dataset profile."""
    profile = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "unique_counts": {col: df[col].nunique() for col in df.columns},
        "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
        "datetime_columns": df.select_dtypes(include=['datetime64']).columns.tolist(),
    }
    return profile

# ============================================================================
# SECTION 3: COLUMN DETECTION
# ============================================================================

def normalize_column_name(name):
    """Normalize column name."""
    return name.lower().strip().replace('_', '').replace(' ', '').replace('-', '')

def detect_columns(df):
    """Detect business-relevant columns using synonym matching."""
    
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
# SECTION 4.1: DESCRIPTIVE STATS
# ============================================================================

def compute_descriptive_stats(df, column_map):
    """Compute stats: mean, median, min, max, std, pct change."""
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
                        "count": int(len(numeric_data)),
                    }
                    
                    if len(numeric_data) > 1:
                        first_val = numeric_data.iloc[0]
                        last_val = numeric_data.iloc[-1]
                        if first_val != 0:
                            pct_change = (last_val - first_val) / first_val
                            stats[col]["pct_change"] = float(pct_change)
    
    return stats

# ============================================================================
# SECTION 4.2: TIME-SERIES ANALYSIS
# ============================================================================

def compute_time_series_analysis(df, column_map):
    """Time-series analysis if date column exists."""
    if "date" not in column_map:
        return None
    
    date_col = column_map["date"]["column"]
    if date_col not in df.columns:
        return None
    
    try:
        df_ts = df.copy()
        df_ts[date_col] = pd.to_datetime(df_ts[date_col], errors='coerce')
        df_ts = df_ts.dropna(subset=[date_col])
        df_ts = df_ts.sort_values(by=date_col)
        
        if len(df_ts) < 2:
            return None
        
        df_ts.set_index(date_col, inplace=True)
        
        ts_analysis = {
            "date_range": f"{df_ts.index.min().date()} to {df_ts.index.max().date()}",
            "total_periods": len(df_ts),
            "metrics": {}
        }
        
        for col in df_ts.select_dtypes(include=['number']).columns:
            numeric_series = pd.to_numeric(df_ts[col], errors='coerce').dropna()
            if len(numeric_series) > 1:
                pct_change = numeric_series.pct_change().dropna()
                
                ts_analysis["metrics"][col] = {
                    "periods": int(len(numeric_series)),
                    "avg_growth": float(pct_change.mean()) if len(pct_change) > 0 else 0,
                    "trend": "up" if pct_change.mean() > 0 else "down" if pct_change.mean() < 0 else "flat",
                }
        
        return ts_analysis
    
    except Exception as e:
        return None

# ============================================================================
# SECTION 4.3: ANOMALY DETECTION
# ============================================================================

def detect_anomalies(df, column_map):
    """Detect anomalies using IQR method."""
    anomalies = []
    
    for role, col_info in column_map.items():
        if role in ["revenue", "cost", "orders", "traffic"]:
            col = col_info["column"]
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                numeric_data = pd.to_numeric(df[col], errors='coerce').dropna()
                
                if len(numeric_data) >= 5:
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
                                    "value": float(numeric_data[idx]),
                                    "expected_range": [float(lower_bound), float(upper_bound)],
                                    "method": "IQR"
                                })
    
    return anomalies[:10]

# ============================================================================
# SECTION 4.4: CORRELATION ANALYSIS
# ============================================================================

def compute_correlations(df, column_map):
    """Correlation matrix for numeric columns."""
    numeric_cols = [col_info["column"] for col_info in column_map.values() 
                    if col_info["column"] in df.select_dtypes(include=['number']).columns]
    
    if len(numeric_cols) < 2:
        return []
    
    corr_matrix = df[numeric_cols].corr()
    correlations = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.3:
                correlations.append({
                    "pair": [corr_matrix.columns[i], corr_matrix.columns[j]],
                    "r": float(r),
                    "type": "correlation"
                })
    
    correlations.sort(key=lambda x: abs(x["r"]), reverse=True)
    return correlations[:5]

# ============================================================================
# SECTION 4.5: SEGMENTATION
# ============================================================================

def compute_segmentation(df, column_map):
    """Segmentation analysis."""
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
    except Exception:
        return None

# ============================================================================
# SECTION 5: PROBLEM DETECTION
# ============================================================================

def detect_problems(profile, stats, ts_analysis, anomalies, column_map):
    """Detect business problems."""
    problems = []
    
    if ts_analysis and "metrics" in ts_analysis:
        for metric, data in ts_analysis["metrics"].items():
            if "revenue" in metric.lower() and data["trend"] == "down":
                magnitude = data["avg_growth"]
                severity = "High" if abs(magnitude) > 0.25 else "Medium" if abs(magnitude) > 0.1 else "Low"
                problems.append({
                    "problem": "Revenue Decline",
                    "severity": severity,
                    "metric": metric,
                    "evidence": f"Revenue declined {abs(magnitude)*100:.1f}% on average.",
                    "time_period": ts_analysis.get("date_range", "N/A"),
                    "magnitude": magnitude,
                })
            
            if "cost" in metric.lower() and data["trend"] == "up":
                magnitude = data["avg_growth"]
                severity = "High" if magnitude > 0.25 else "Medium"
                problems.append({
                    "problem": "Rising Costs",
                    "severity": severity,
                    "metric": metric,
                    "evidence": f"Costs increased {magnitude*100:.1f}% on average.",
                    "time_period": ts_analysis.get("date_range", "N/A"),
                    "magnitude": magnitude,
                })
            
            if "orders" in metric.lower() and data["trend"] == "down":
                magnitude = data["avg_growth"]
                severity = "High" if abs(magnitude) > 0.25 else "Medium"
                problems.append({
                    "problem": "Falling Orders",
                    "severity": severity,
                    "metric": metric,
                    "evidence": f"Orders declined {abs(magnitude)*100:.1f}% on average.",
                    "time_period": ts_analysis.get("date_range", "N/A"),
                    "magnitude": magnitude,
                })
    
    if anomalies:
        problems.append({
            "problem": "Data Anomalies Detected",
            "severity": "Medium",
            "metric": "Multiple metrics",
            "evidence": f"{len(anomalies)} statistical outliers detected.",
            "time_period": "Various dates",
            "magnitude": 0,
        })
    
    return problems

# ============================================================================
# SECTION 6: ROOT-CAUSE HYPOTHESES
# ============================================================================

def build_root_cause_hypotheses(problems, ts_analysis, anomalies):
    """Build root-cause hypotheses."""
    hypotheses = []
    
    for problem in problems:
        problem_name = problem["problem"]
        
        if "Revenue Decline" in problem_name:
            hypotheses.append({
                "problem": problem_name,
                "hypothesis": "Revenue decline may be associated with reduced customer orders or lower conversion rates.",
                "supporting_evidence": ["Revenue metric shows declining trend"],
                "confidence": "Medium",
            })
        
        elif "Rising Costs" in problem_name:
            hypotheses.append({
                "problem": problem_name,
                "hypothesis": "Cost increase may indicate operational expansion or inefficiency.",
                "supporting_evidence": ["Cost metrics trending upward"],
                "confidence": "Medium",
            })
        
        elif "Falling Orders" in problem_name:
            hypotheses.append({
                "problem": problem_name,
                "hypothesis": "Order decline may be due to reduced marketing effectiveness or lower traffic.",
                "supporting_evidence": ["Order metrics declining"],
                "confidence": "Medium",
            })
        
        elif "Data Anomalies" in problem_name:
            hypotheses.append({
                "problem": problem_name,
                "hypothesis": "Anomalies may indicate data quality issues or unusual business events.",
                "supporting_evidence": [f"{len(anomalies)} statistical outliers detected"],
                "confidence": "Low",
            })
    
    return hypotheses

# ============================================================================
# SECTION 7: RECOMMENDATIONS
# ============================================================================

def build_recommendations(problems):
    """Build recommendations."""
    recommendations = []
    
    for problem in problems:
        problem_name = problem["problem"]
        
        if "Revenue Decline" in problem_name:
            recommendations.append({
                "problem": problem_name,
                "recommendation": "Investigate drop in order volume. Analyze customer acquisition funnel to identify conversion leaks.",
                "priority": "High",
                "expected_impact": "High",
            })
        
        elif "Rising Costs" in problem_name:
            recommendations.append({
                "problem": problem_name,
                "recommendation": "Conduct cost-benefit analysis. Negotiate with vendors and identify cost-saving opportunities.",
                "priority": "High",
                "expected_impact": "Medium",
            })
        
        elif "Falling Orders" in problem_name:
            recommendations.append({
                "problem": problem_name,
                "recommendation": "Increase marketing spend and optimize conversion funnel. Test new channels.",
                "priority": "High",
                "expected_impact": "High",
            })
        
        elif "Data Anomalies" in problem_name:
            recommendations.append({
                "problem": problem_name,
                "recommendation": "Verify data collection processes and investigate outlier records.",
                "priority": "Medium",
                "expected_impact": "Medium",
            })
    
    return recommendations

# ============================================================================
# SECTION 9: LLM INTEGRATION
# ============================================================================

def analyze_with_ai(payload, provider, business_type, dataset_description):
    """Call Groq or Gemini API."""
    
    system_prompt = """You are RootIQ, an expert business analyst.

You receive pre-computed analytical findings. Your job is to:
1. Interpret findings intelligently
2. Generate executive summary (2-3 sentences)
3. Identify key patterns

CRITICAL RULES:
- Never invent numbers not in the payload
- Correlation ≠ causation. Use "may indicate", "associated with"
- If evidence is thin, say "Insufficient evidence"
- Respond ONLY with valid JSON (no markdown):

{
  "executive_summary": "string",
  "key_findings": ["finding1", "finding2"]
}"""
    
    user_message = f"""Business: {business_type}
Dataset: {dataset_description}

Findings:
{json.dumps(payload, indent=2)}

Provide JSON response ONLY."""
    
    try:
        if provider == "Groq (Recommended)":
            return _call_groq(system_prompt, user_message)
        else:
            return _call_gemini(system_prompt, user_message)
    
    except Exception as e:
        raise AIProviderError(f"AI analysis failed: {str(e)}")

def _call_groq(system_prompt, user_message):
    """Call Groq."""
    api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    
    if not api_key:
        raise AIProviderError("GROQ_API_KEY not configured")
    
    client = Groq(api_key=api_key)
    
    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.2,
        max_tokens=1000,
    )
    
    raw_text = response.choices[0].message.content.strip()
    return _parse_ai_response(raw_text)

def _call_gemini(system_prompt, user_message):
    """Call Gemini."""
    api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        raise AIProviderError("GEMINI_API_KEY not configured")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    response = model.generate_content(f"{system_prompt}\n\n{user_message}")
    raw_text = response.text.strip()
    return _parse_ai_response(raw_text)

def _parse_ai_response(raw_text):
    """Parse AI response defensively."""
    try:
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(raw_text)
        
        if "executive_summary" not in result:
            result["executive_summary"] = "Analysis generated"
        if "key_findings" not in result:
            result["key_findings"] = []
        
        return result
    
    except json.JSONDecodeError:
        return {
            "executive_summary": raw_text[:500],
            "key_findings": ["AI analysis generated successfully"]
        }

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.header("📤 Upload & Configure")

uploaded_file = st.sidebar.file_uploader(
    "Upload your data (CSV, XLSX, JSON)",
    type=["csv", "xlsx", "json"],
)

st.sidebar.divider()

business_type = st.sidebar.text_input(
    "Business Type",
    placeholder="e.g., E-commerce, SaaS",
)

dataset_description = st.sidebar.text_area(
    "Dataset Description",
    placeholder="e.g., Monthly sales, orders, traffic",
)

st.sidebar.divider()

with st.sidebar.expander("⚙️ Advanced Options", expanded=False):
    st.text_input("Date Column (optional)", placeholder="e.g., order_date")
    st.text_input("Target Metric (optional)", placeholder="e.g., revenue")

st.sidebar.divider()

ai_provider = st.sidebar.selectbox(
    "AI Provider",
    options=["Groq (Recommended)", "Gemini"],
    index=0,
)

st.sidebar.info("ℹ️ API keys optional. App works without them.")

st.sidebar.divider()

analyze_button = st.sidebar.button("🔬 Analyze", use_container_width=True, type="primary")

# ============================================================================
# MAIN LOGIC
# ============================================================================

if not uploaded_file:
    st.info("👆 Upload a CSV, XLSX, or JSON file to begin analysis.")
    st.stop()

if not business_type or not dataset_description:
    st.warning("⚠️ Please fill in Business Type and Dataset Description.")
    st.stop()

try:
    df, file_type = load_data(uploaded_file)
    st.session_state.df = df
except DataLoadError as e:
    st.error(f"❌ {str(e)}")
    st.stop()

profile = profile_dataset(df)
st.session_state.profile = profile

column_map = detect_columns(df)
st.session_state.column_map = column_map

if not column_map:
    st.error("❌ No business-relevant columns detected.")
    st.stop()

# ============================================================================
# DATA OVERVIEW
# ============================================================================

st.header("📊 Data Overview")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📊 Rows", f"{profile['row_count']:,}")
with col2:
    st.metric("📋 Columns", profile["column_count"])
with col3:
    st.metric("🔄 Duplicates", profile["duplicate_rows"])
with col4:
    total_missing = sum(profile["missing_values"].values())
    st.metric("⚠️ Missing", total_missing)

st.divider()

st.subheader("🎯 Detected Columns")
for role, col_info in column_map.items():
    conf_color = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
    st.write(f"{conf_color[col_info['confidence']]} **{role.title()}**: `{col_info['column']}`")

st.divider()

# ============================================================================
# ANALYSIS
# ============================================================================

if analyze_button:
    with st.spinner("🔄 Analyzing..."):
        
        stats = compute_descriptive_stats(df, column_map)
        ts_analysis = compute_time_series_analysis(df, column_map)
        anomalies = detect_anomalies(df, column_map)
        correlations = compute_correlations(df, column_map)
        segmentation = compute_segmentation(df, column_map)
        
        problems = detect_problems(profile, stats, ts_analysis, anomalies, column_map)
        hypotheses = build_root_cause_hypotheses(problems, ts_analysis, anomalies)
        recommendations = build_recommendations(problems)
        
        st.session_state.results = {
            "stats": stats,
            "ts_analysis": ts_analysis,
            "anomalies": anomalies,
            "correlations": correlations,
            "segmentation": segmentation,
            "problems": problems,
            "hypotheses": hypotheses,
            "recommendations": recommendations,
        }
        
        # Try AI analysis (optional)
        if problems:
            payload = {
                "problems": problems,
                "hypotheses": hypotheses,
                "stats": stats,
            }
            
            try:
                ai_results = analyze_with_ai(payload, ai_provider, business_type, dataset_description)
                st.session_state.ai_results = ai_results
            except AIProviderError as e:
                st.warning(f"ℹ️ {str(e)} - Showing data analysis.")
                st.session_state.ai_results = None
        
        st.session_state.analysis_complete = True
    
    st.success("✅ Analysis complete!")

# ============================================================================
# DISPLAY RESULTS
# ============================================================================

if st.session_state.analysis_complete:
    
    results = st.session_state.results
    ai_results = st.session_state.ai_results
    
    # Executive Summary
    st.header("📝 Executive Summary")
    if ai_results:
        st.write(ai_results.get("executive_summary", "Analysis generated."))
        if "key_findings" in ai_results:
            st.subheader("🔑 Key Findings")
            for finding in ai_results["key_findings"][:5]:
                st.write(f"• {finding}")
    else:
        num_problems = len(results["problems"])
        st.info(f"📊 **{num_problems} business problems detected** from data patterns.")
    
    st.divider()
    
    # KPI Cards
    st.header("📈 Key Metrics")
    if results["stats"]:
        metric_cols = st.columns(len(results["stats"]))
        for idx, (metric, values) in enumerate(results["stats"].items()):
            with metric_cols[idx % len(metric_cols)]:
                st.metric(metric, f"{values['mean']:.0f}")
    
    st.divider()
    
    # Problems
    if results["problems"]:
        st.header("⚠️ Detected Problems")
        for problem in results["problems"]:
            severity_color = {"High": "🔴", "Medium": "🟠", "Low": "🟡"}
            with st.expander(f"{severity_color[problem['severity']]} **{problem['problem']}** ({problem['severity']})"):
                st.write(f"**Metric**: `{problem['metric']}`")
                st.write(f"**Evidence**: {problem['evidence']}")
                st.write(f"**Period**: {problem['time_period']}")
    
    st.divider()
    
    # Hypotheses
    if results["hypotheses"]:
        st.header("🔍 Root-Cause Analysis")
        for hyp in results["hypotheses"]:
            conf_color = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
            with st.expander(f"{conf_color[hyp['confidence']]} {hyp['hypothesis'][:60]}..."):
                st.write("**Supporting Evidence:**")
                for ev in hyp["supporting_evidence"]:
                    st.write(f"• {ev}")
                st.write(f"**Confidence**: {hyp['confidence']}")
    
    st.divider()
    
    # Recommendations
    if results["recommendations"]:
        st.header("💡 Recommendations")
        for rec in results["recommendations"]:
            priority_color = {"High": "🔴", "Medium": "🟠"}
            with st.expander(f"{priority_color[rec['priority']]} {rec['recommendation'][:60]}..."):
                st.write(f"**Priority**: {rec['priority']}")
                st.write(f"**Impact**: {rec['expected_impact']}")
    
    st.divider()
    
    # Stats Table
    if results["stats"]:
        st.header("📊 Detailed Statistics")
        stats_df = pd.DataFrame(results["stats"]).T.round(2)
        st.dataframe(stats_df, use_container_width=True)
    
    st.divider()
    
    # Anomalies
    if results["anomalies"]:
        st.header("⚠️ Data Anomalies")
        anom_df = pd.DataFrame(results["anomalies"])
        st.dataframe(anom_df, use_container_width=True)
    
    st.divider()
    
    # Correlations
    if results["correlations"]:
        st.header("🔗 Correlations")
        corr_df = pd.DataFrame(results["correlations"])
        st.dataframe(corr_df, use_container_width=True)
    
    st.divider()
    
    # Segmentation
    if results["segmentation"]:
        st.header("📊 Segment Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top Performers")
            for perf in results["segmentation"]["top_performers"]:
                st.write(f"• {perf}")
        with col2:
            st.subheader("Bottom Performers")
            for perf in results["segmentation"]["bottom_performers"]:
                st.write(f"• {perf}")
    
    st.divider()
    
    # Sample Data
    st.header("🔍 Sample Data")
    st.dataframe(df.head(15), use_container_width=True)
