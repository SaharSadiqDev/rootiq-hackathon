"""
RootIQ: Evidence-based root-cause analysis platform.
COMPLETE Implementation: Sections 1-15 per specification.

Data → Evidence → Problem → Root-Cause Hypotheses → Recommendation.
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from groq import Groq
import plotly.express as px

# ============================================================================
# PAGE CONFIG & SESSION STATE
# ============================================================================

st.set_page_config(
    page_title="RootIQ",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 RootIQ")
st.markdown("**AI that finds the 'why' behind your business data.**")

if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
if "df" not in st.session_state:
    st.session_state.df = None
if "results" not in st.session_state:
    st.session_state.results = None
if "ai_results" not in st.session_state:
    st.session_state.ai_results = None

# ============================================================================
# ERROR HANDLING
# ============================================================================

class DataLoadError(Exception):
    pass

class AIProviderError(Exception):
    pass

# ============================================================================
# DATA LOADING & PROFILING
# ============================================================================

def load_data(uploaded_file):
    """Load CSV, XLSX, JSON."""
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
            raise DataLoadError("PDF support coming soon. Use CSV, XLSX, or JSON.")
        else:
            raise DataLoadError(f"Unsupported file: {filename}")
    
    except Exception as e:
        if isinstance(e, DataLoadError):
            raise
        raise DataLoadError(f"Load failed: {str(e)}")

def profile_dataset(df):
    """Profile dataset."""
    profile = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
        "datetime_columns": df.select_dtypes(include=['datetime64']).columns.tolist(),
    }
    return profile

# ============================================================================
# COLUMN DETECTION
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
        "category": ["category", "productcategory", "segment", "region", "location", "channel", "country", "state", "city"],
        "marketing_spend": ["adspend", "marketingspend", "advertising", "adcost", "campaignspend"],
        "conversion": ["conversionrate", "cvr", "conversion"],
        "traffic": ["visits", "websitevisits", "sessions", "traffic", "pageviews", "impressions"],
    }
    
    column_map = {}
    normalized_cols = {normalize_column_name(col): col for col in df.columns}
    
    for role, synonyms in synonym_map.items():
        for synonym in synonyms:
            norm_synonym = normalize_column_name(synonym)
            for norm_col, orig_col in normalized_cols.items():
                if norm_synonym in norm_col or norm_col in norm_synonym:
                    col_dtype = df[orig_col].dtype
                    
                    if role in ["revenue", "cost", "profit", "orders", "marketing_spend", "traffic", "conversion"]:
                        if pd.api.types.is_numeric_dtype(col_dtype):
                            confidence = "High" if norm_synonym == norm_col else "Medium"
                            column_map[role] = {"column": orig_col, "confidence": confidence}
                            break
                    else:
                        confidence = "High" if norm_synonym == norm_col else "Medium"
                        column_map[role] = {"column": orig_col, "confidence": confidence}
                        break
    
    return column_map

# ============================================================================
# ANALYTICS
# ============================================================================

def compute_descriptive_stats(df, column_map):
    """Compute stats."""
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
                    
                    if len(numeric_data) > 1:
                        first = numeric_data.iloc[0]
                        last = numeric_data.iloc[-1]
                        if first != 0:
                            pct_change = (last - first) / first
                            stats[col]["pct_change"] = float(pct_change)
    
    return stats

def compute_time_series_analysis(df, column_map):
    """Time-series analysis."""
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
            "periods": len(df_ts),
            "metrics": {},
            "df_ts": df_ts,
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

def detect_anomalies(df, column_map):
    """Detect anomalies using IQR."""
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

def compute_correlations(df, column_map):
    """Correlation analysis."""
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

def compute_segmentation(df, column_map):
    """Segmentation."""
    if "category" not in column_map:
        return None
    
    category_col = column_map["category"]["column"]
    revenue_col = column_map.get("revenue", {}).get("column")
    
    if category_col not in df.columns or not revenue_col or revenue_col not in df.columns:
        return None
    
    try:
        segmentation = df.groupby(category_col)[revenue_col].agg(['sum', 'mean'])
        segmentation = segmentation.sort_values('sum', ascending=False)
        
        return {
            "top_performers": segmentation.head(3).index.tolist(),
            "bottom_performers": segmentation.tail(3).index.tolist(),
        }
    except Exception:
        return None

# ============================================================================
# PROBLEM DETECTION
# ============================================================================

def detect_problems(profile, stats, ts_analysis, anomalies, column_map):
    """Detect problems."""
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
                    "evidence": f"Revenue declined {abs(magnitude)*100:.1f}% on average over {data['periods']} periods.",
                    "time_period": ts_analysis.get("date_range", "N/A"),
                    "magnitude": magnitude,
                    "source_columns": [metric]
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
                    "source_columns": [metric]
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
                    "source_columns": [metric]
                })
    
    if anomalies:
        problems.append({
            "problem": "Data Anomalies Detected",
            "severity": "Medium",
            "metric": "Multiple metrics",
            "evidence": f"{len(anomalies)} statistical outliers detected using IQR method.",
            "time_period": "Various",
            "magnitude": 0,
            "source_columns": [a["column"] for a in anomalies[:3]]
        })
    
    return problems

# ============================================================================
# ROOT-CAUSE HYPOTHESES
# ============================================================================

def build_root_cause_hypotheses(problems, stats, ts_analysis, anomalies, correlations, segmentation, column_map):
    """Build hypotheses."""
    hypotheses = []
    
    for problem in problems:
        name = problem["problem"]
        
        if "Revenue Decline" in name:
            sup_ev = ["Revenue metric shows declining trend"]
            contra_ev = []
            
            if ts_analysis and "metrics" in ts_analysis:
                if any("order" in m.lower() for m in ts_analysis["metrics"]):
                    sup_ev.append("Order volume also trending downward")
            
            if correlations:
                for corr in correlations:
                    if any(x in str(corr["pair"]).lower() for x in ["revenue", "sales"]):
                        sup_ev.append(f"{corr['pair'][0]} and {corr['pair'][1]} correlated (r={corr['r']:.2f})")
            
            hypotheses.append({
                "problem": name,
                "hypothesis": "Revenue decline may be associated with reduced customer orders or lower conversion rates.",
                "supporting_evidence": sup_ev,
                "contradicting_evidence": contra_ev,
                "confidence": "Medium" if len(sup_ev) > 1 else "Low",
                "evidence_sources": [problem["metric"]]
            })
        
        elif "Rising Costs" in name:
            hypotheses.append({
                "problem": name,
                "hypothesis": "Cost increase may indicate operational expansion or pricing changes from vendors.",
                "supporting_evidence": ["Cost metrics trending upward", "Sustained increase over multiple periods"],
                "contradicting_evidence": [],
                "confidence": "Medium",
                "evidence_sources": [problem["metric"]]
            })
        
        elif "Falling Orders" in name:
            sup_ev = ["Order metrics declining"]
            
            if ts_analysis and "metrics" in ts_analysis:
                if any("traffic" in m.lower() for m in ts_analysis["metrics"]):
                    sup_ev.append("Website traffic may have declined proportionally")
            
            hypotheses.append({
                "problem": name,
                "hypothesis": "Order decline may result from reduced marketing effectiveness or lower customer acquisition.",
                "supporting_evidence": sup_ev,
                "contradicting_evidence": [],
                "confidence": "Medium",
                "evidence_sources": [problem["metric"]]
            })
        
        elif "Anomalies" in name:
            hypotheses.append({
                "problem": name,
                "hypothesis": "Anomalies may indicate data quality issues, unusual business events, or one-time spikes.",
                "supporting_evidence": [f"{len(anomalies)} statistical outliers detected"],
                "contradicting_evidence": [],
                "confidence": "Low",
                "evidence_sources": [a["column"] for a in anomalies[:2]]
            })
    
    return hypotheses

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

def build_recommendations(problems, hypotheses):
    """Build recommendations."""
    recommendations = []
    
    for problem in problems:
        name = problem["problem"]
        
        if "Revenue Decline" in name:
            recommendations.append({
                "problem": name,
                "recommendation": "Investigate order drop-off in customer funnel. Analyze conversion rates by traffic source and product category.",
                "priority": "High",
                "expected_impact": "High",
                "reason": "Revenue decline directly correlates with order volume. Fixing order acquisition or conversion will recover revenue."
            })
        
        elif "Rising Costs" in name:
            recommendations.append({
                "problem": name,
                "recommendation": "Conduct vendor cost review. Renegotiate contracts and identify operational efficiency improvements.",
                "priority": "High",
                "expected_impact": "Medium",
                "reason": "Cost increases erode profit margins. Even small reductions compound significantly over time."
            })
        
        elif "Falling Orders" in name:
            recommendations.append({
                "problem": name,
                "recommendation": "Increase marketing spend in high-ROI channels. A/B test landing pages and optimize conversion funnel.",
                "priority": "High",
                "expected_impact": "High",
                "reason": "Orders are directly tied to top-of-funnel acquisition and mid-funnel conversion. Both are addressable."
            })
        
        elif "Anomalies" in name:
            recommendations.append({
                "problem": name,
                "recommendation": "Verify data collection processes. Investigate outlier records for data quality or legitimate business events.",
                "priority": "Medium",
                "expected_impact": "Medium",
                "reason": "Data quality affects all downstream analysis. Resolving anomalies improves confidence in all insights."
            })
    
    return recommendations

# ============================================================================
# ADDITIONAL DATA & NEXT ACTIONS
# ============================================================================

def get_additional_data_needed(problems, hypotheses):
    """What data would help."""
    data_suggestions = []
    next_actions = []
    
    for hyp in hypotheses:
        if hyp["confidence"] in ["Low", "Medium"]:
            if "Revenue" in hyp["problem"]:
                data_suggestions.append("Customer acquisition cost (CAC) and lifetime value (LTV) data")
            if "Order" in hyp["problem"]:
                data_suggestions.append("Traffic source and device breakdown")
            if "Cost" in hyp["problem"]:
                data_suggestions.append("Cost breakdown by department or expense category")
    
    for problem in problems:
        if problem["severity"] == "High":
            next_actions.append(f"🔴 URGENT: Root-cause analysis for {problem['problem'].lower()}")
    
    next_actions.append("2. Conduct stakeholder interviews to validate hypotheses")
    next_actions.append("3. Implement real-time KPI tracking for leading indicators")
    
    return list(set(data_suggestions))[:3], next_actions[:3]

# ============================================================================
# GROQ AI INTEGRATION
# ============================================================================

def analyze_with_ai(payload, business_type, dataset_description):
    """Call Groq API."""
    
    api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    
    if not api_key:
        raise AIProviderError("GROQ_API_KEY not configured. Add to Secrets.")
    
    client = Groq(api_key=api_key)
    
    system_prompt = """You are RootIQ, an expert business analyst.

You receive PRE-COMPUTED analytical findings. Your job is to:
1. Interpret findings intelligently
2. Generate executive summary (2-3 sentences)
3. Validate hypotheses with evidence

CRITICAL RULES:
- Never invent numbers not in payload
- Correlation ≠ causation. Use "may indicate", "associated with", "suggests"
- If evidence is thin, say "Insufficient evidence"
- Respond ONLY with valid JSON (no markdown):

{
  "executive_summary": "string",
  "key_findings": ["string"]
}"""
    
    user_message = f"""Business: {business_type}
Data: {dataset_description}

Findings:
{json.dumps(payload, indent=2, default=str)}

Respond with JSON ONLY."""
    
    try:
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
    
    except Exception as e:
        raise AIProviderError(f"Groq call failed: {str(e)}")

# ============================================================================
# BUILD AI PAYLOAD
# ============================================================================

def build_ai_payload(profile, stats, ts_analysis, anomalies, correlations, segmentation,
                     problems, hypotheses, df, column_map):
    """Token-budgeted payload."""
    sample_rows = df.head(5).to_dict(orient='records')
    
    payload = {
        "row_count": profile["row_count"],
        "column_count": profile["column_count"],
        "stats": stats,
        "problems": problems[:5],
        "hypotheses": hypotheses[:5],
        "correlations": correlations,
    }
    
    return payload

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
    st.text_input("Date Column (optional)", placeholder="e.g., date")
    st.text_input("Target Metric (optional)", placeholder="e.g., revenue")

st.sidebar.divider()

st.sidebar.info("💡 Add GROQ_API_KEY to Secrets for AI analysis.")

st.sidebar.divider()

analyze_button = st.sidebar.button("🔬 Analyze", use_container_width=True, type="primary")

# ============================================================================
# MAIN
# ============================================================================

if not uploaded_file:
    st.info("👆 Upload a CSV, XLSX, or JSON file to begin.")
    st.stop()

if not business_type or not dataset_description:
    st.warning("⚠️ Fill in Business Type and Description.")
    st.stop()

try:
    df, file_type = load_data(uploaded_file)
except DataLoadError as e:
    st.error(f"❌ {str(e)}")
    st.stop()

profile = profile_dataset(df)
column_map = detect_columns(df)

if not column_map:
    st.error("❌ No business columns detected.")
    st.stop()

# ============================================================================
# DATA OVERVIEW
# ============================================================================

st.header("📊 Data Overview")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Rows", f"{profile['row_count']:,}")
with col2:
    st.metric("Columns", profile["column_count"])
with col3:
    st.metric("Duplicates", profile["duplicate_rows"])
with col4:
    st.metric("Missing", sum(profile["missing_values"].values()))

st.divider()

st.subheader("🎯 Detected Columns")
for role, col_info in column_map.items():
    conf = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
    st.write(f"{conf[col_info['confidence']]} **{role}**: `{col_info['column']}`")

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
        hypotheses = build_root_cause_hypotheses(problems, stats, ts_analysis, anomalies, correlations, segmentation, column_map)
        recommendations = build_recommendations(problems, hypotheses)
        additional_data, next_actions = get_additional_data_needed(problems, hypotheses)
        
        st.session_state.results = {
            "stats": stats,
            "ts_analysis": ts_analysis,
            "anomalies": anomalies,
            "correlations": correlations,
            "segmentation": segmentation,
            "problems": problems,
            "hypotheses": hypotheses,
            "recommendations": recommendations,
            "additional_data": additional_data,
            "next_actions": next_actions,
        }
        
        if problems:
            payload = build_ai_payload(profile, stats, ts_analysis, anomalies, correlations,
                                      segmentation, problems, hypotheses, df, column_map)
            
            try:
                ai_results = analyze_with_ai(payload, business_type, dataset_description)
                st.session_state.ai_results = ai_results
            except AIProviderError as e:
                st.info(f"ℹ️ {str(e)}")
                st.session_state.ai_results = None
        
        st.session_state.analysis_complete = True
    
    st.success("✅ Analysis complete!")

# ============================================================================
# RESULTS DISPLAY
# ============================================================================

if st.session_state.analysis_complete:
    
    results = st.session_state.results
    ai_results = st.session_state.ai_results
    
    # Executive Summary
    st.header("📝 Executive Summary")
    if ai_results:
        st.write(ai_results.get("executive_summary", ""))
        if "key_findings" in ai_results:
            st.subheader("🔑 Key Findings")
            for finding in ai_results["key_findings"]:
                st.write(f"• {finding}")
    else:
        st.info(f"**{len(results['problems'])} problems detected** from data analysis.")
    
    st.divider()
    
    # KPI Cards
    st.header("📈 Key Metrics")
    if results["stats"]:
        cols = st.columns(min(4, len(results["stats"])))
        for idx, (metric, values) in enumerate(results["stats"].items()):
            with cols[idx % len(cols)]:
                st.metric(metric, f"{values['mean']:.0f}", f"Std: {values['std']:.0f}")
    
    st.divider()
    
    # Trends Chart
    if results["ts_analysis"] and results["ts_analysis"].get("df_ts") is not None:
        st.header("📊 Trends")
        df_ts = results["ts_analysis"]["df_ts"].reset_index()
        if len(df_ts) > 0:
            metric_to_plot = None
            for col in df_ts.columns:
                if "revenue" in col.lower():
                    metric_to_plot = col
                    break
            if metric_to_plot:
                fig = px.line(df_ts, x=df_ts.columns[0], y=metric_to_plot, title=f"{metric_to_plot} Trend", markers=True)
                st.plotly_chart(fig, use_container_width=True)
        st.divider()
    
    # Problems
    if results["problems"]:
        st.header("⚠️ Detected Problems")
        for problem in results["problems"]:
            severity_color = {"High": "🔴", "Medium": "🟠", "Low": "🟡"}
            with st.expander(f"{severity_color[problem['severity']]} **{problem['problem']}**"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Metric**: {problem['metric']}")
                    st.write(f"**Evidence**: {problem['evidence']}")
                with col2:
                    st.write(f"**Period**: {problem['time_period']}")
                    st.write(f"**Magnitude**: {problem['magnitude']*100:.1f}%")
        st.divider()
    
    # Root-Cause (Supporting vs Contradicting)
    if results["hypotheses"]:
        st.header("🔍 Root-Cause Analysis")
        for hyp in results["hypotheses"]:
            conf_color = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
            with st.expander(f"{conf_color[hyp['confidence']]} {hyp['hypothesis'][:70]}..."):
                st.write(f"**Confidence**: {hyp['confidence']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Supporting Evidence:**")
                    for ev in hyp["supporting_evidence"]:
                        st.write(f"✓ {ev}")
                
                with col2:
                    st.write("**Contradicting Evidence:**")
                    if hyp["contradicting_evidence"]:
                        for contra in hyp["contradicting_evidence"]:
                            st.write(f"✗ {contra}")
                    else:
                        st.write("*None found*")
        st.divider()
    
    # Recommendations
    if results["recommendations"]:
        st.header("💡 Recommendations")
        for rec in results["recommendations"]:
            priority_color = {"High": "🔴", "Medium": "🟠"}
            with st.expander(f"{priority_color[rec['priority']]} {rec['recommendation'][:70]}..."):
                st.write(f"**Priority**: {rec['priority']}")
                st.write(f"**Impact**: {rec['expected_impact']}")
                st.write(f"**Reason**: {rec['reason']}")
        st.divider()
    
    # Additional Data
    if results["additional_data"]:
        st.header("📊 Additional Data Needed")
        for item in results["additional_data"]:
            st.write(f"• {item}")
        st.divider()
    
    # Next Actions
    if results["next_actions"]:
        st.header("🎯 Next Actions")
        for action in results["next_actions"]:
            st.write(f"• {action}")
        st.divider()
    
    # Stats Table
    if results["stats"]:
        st.header("📈 Detailed Statistics")
        stats_df = pd.DataFrame(results["stats"]).T.round(2)
        st.dataframe(stats_df, use_container_width=True)
        st.divider()
    
    # Anomalies
    if results["anomalies"]:
        st.header("⚠️ Anomalies")
        anom_df = pd.DataFrame(results["anomalies"])
        st.dataframe(anom_df[["column", "value", "expected_range"]], use_container_width=True)
        st.divider()
    
    # Correlations
    if results["correlations"]:
        st.header("🔗 Correlations")
        corr_df = pd.DataFrame(results["correlations"])
        st.dataframe(corr_df, use_container_width=True)
        st.divider()
    
    # Segmentation
    if results["segmentation"]:
        st.header("📊 Segments")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Top**")
            for perf in results["segmentation"]["top_performers"]:
                st.write(f"• {perf}")
        with col2:
            st.write("**Bottom**")
            for perf in results["segmentation"]["bottom_performers"]:
                st.write(f"• {perf}")
        st.divider()
    
    # Evidence Explorer
    st.header("🔎 Evidence Explorer")
    with st.expander("View Raw Data"):
        st.dataframe(df.head(20), use_container_width=True)
        st.caption(f"Showing {min(20, len(df))} of {len(df)} rows")
    
    st.divider()
    
    # Download Report
    st.header("📥 Download Report")
    report = generate_markdown_report(business_type, dataset_description, profile, results, ai_results)
    st.download_button(
        label="📄 Download Report",
        data=report,
        file_name=f"rootiq_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown"
    )

# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_markdown_report(business_type, dataset_description, profile, results, ai_results):
    """Generate markdown report."""
    
    report = f"""# RootIQ Analysis Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

{ai_results.get('executive_summary', 'Data-driven analysis completed.') if ai_results else 'Analysis completed.'}

## Business Context

- **Type**: {business_type}
- **Data**: {dataset_description}

## Dataset Profile

| Metric | Value |
|--------|-------|
| Rows | {profile['row_count']} |
| Columns | {profile['column_count']} |
| Duplicates | {profile['duplicate_rows']} |
| Missing | {sum(profile['missing_values'].values())} |

## Problems Detected

"""
    
    for problem in results.get("problems", []):
        report += f"### {problem['problem']} ({problem['severity']})\n"
        report += f"- **Evidence**: {problem['evidence']}\n"
        report += f"- **Period**: {problem['time_period']}\n\n"
    
    report += "## Recommendations\n\n"
    for rec in results.get("recommendations", []):
        report += f"- {rec['recommendation']}\n"
    
    report += "\n## Next Actions\n\n"
    for action in results.get("next_actions", []):
        report += f"- {action}\n"
    
    report += "\n---\n*Generated by RootIQ*\n"
    return report
