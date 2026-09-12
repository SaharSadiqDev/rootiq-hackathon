"""
RootIQ: Evidence-based root-cause analysis platform.
COMPLETE - With User API Key Input
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from groq import Groq
import plotly.express as px

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

class DataLoadError(Exception):
    pass

class AIProviderError(Exception):
    pass

def load_data(uploaded_file):
    try:
        if uploaded_file is None:
            raise DataLoadError("No file uploaded.")
        filename = uploaded_file.name.lower()
        if filename.endswith('.csv'):
            return pd.read_csv(uploaded_file), 'csv'
        elif filename.endswith('.xlsx'):
            return pd.read_excel(uploaded_file, engine='openpyxl'), 'xlsx'
        elif filename.endswith('.json'):
            return pd.read_json(uploaded_file), 'json'
        else:
            raise DataLoadError(f"Unsupported: {filename}")
    except Exception as e:
        if isinstance(e, DataLoadError):
            raise
        raise DataLoadError(f"Load failed: {str(e)}")

def profile_dataset(df):
    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
    }

def normalize_column_name(name):
    return name.lower().strip().replace('_', '').replace(' ', '').replace('-', '')

def detect_columns(df):
    synonym_map = {
        "date": ["date", "orderdate", "transactiondate", "purchasedate", "month", "week", "day"],
        "revenue": ["revenue", "sales", "totalsales", "income", "amount"],
        "cost": ["cost", "expense", "cogs", "spend"],
        "profit": ["profit", "netprofit", "margin"],
        "orders": ["orders", "ordercount", "quantity", "qty"],
        "customer": ["customer", "customerid", "client"],
        "product": ["product", "productname", "sku"],
        "category": ["category", "segment", "region", "location"],
        "marketing_spend": ["adspend", "marketingspend", "advertising"],
        "conversion": ["conversion", "conversionrate"],
        "traffic": ["traffic", "visits", "sessions"],
    }
    
    column_map = {}
    normalized_cols = {normalize_column_name(col): col for col in df.columns}
    
    for role, synonyms in synonym_map.items():
        for synonym in synonyms:
            norm_synonym = normalize_column_name(synonym)
            for norm_col, orig_col in normalized_cols.items():
                if norm_synonym in norm_col or norm_col in norm_synonym:
                    if role in ["revenue", "cost", "profit", "orders", "marketing_spend", "traffic"]:
                        if not pd.api.types.is_numeric_dtype(df[orig_col]):
                            continue
                    column_map[role] = {"column": orig_col, "confidence": "High"}
                    break
    
    return column_map

def compute_descriptive_stats(df, column_map):
    stats = {}
    for role, col_info in column_map.items():
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

def compute_time_series_analysis(df, column_map):
    if "date" not in column_map:
        return None
    
    date_col = column_map["date"]["column"]
    if date_col not in df.columns:
        return None
    
    try:
        df_ts = df.copy()
        df_ts[date_col] = pd.to_datetime(df_ts[date_col], errors='coerce')
        df_ts = df_ts.dropna(subset=[date_col]).sort_values(by=date_col)
        
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
                    "avg_growth": float(pct_change.mean()) if len(pct_change) > 0 else 0,
                    "trend": "up" if pct_change.mean() > 0 else "down",
                }
        
        return ts_analysis
    except:
        return None

def detect_anomalies(df, column_map):
    anomalies = []
    for role, col_info in column_map.items():
        if role in ["revenue", "cost", "orders"]:
            col = col_info["column"]
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                numeric_data = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(numeric_data) >= 5:
                    Q1 = numeric_data.quantile(0.25)
                    Q3 = numeric_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    outliers = (numeric_data < lower) | (numeric_data > upper)
                    if outliers.any():
                        for idx, is_out in outliers.items():
                            if is_out:
                                anomalies.append({"column": col, "value": float(numeric_data[idx])})
    return anomalies[:10]

def compute_correlations(df, column_map):
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
                correlations.append({"pair": [corr_matrix.columns[i], corr_matrix.columns[j]], "r": float(r)})
    
    correlations.sort(key=lambda x: abs(x["r"]), reverse=True)
    return correlations[:5]

def detect_problems(ts_analysis, anomalies):
    problems = []
    if ts_analysis and "metrics" in ts_analysis:
        for metric, data in ts_analysis["metrics"].items():
            if "revenue" in metric.lower() and data["trend"] == "down":
                problems.append({
                    "problem": "Revenue Decline",
                    "severity": "High",
                    "metric": metric,
                    "evidence": f"Revenue declined {abs(data['avg_growth'])*100:.1f}%"
                })
            if "cost" in metric.lower() and data["trend"] == "up":
                problems.append({
                    "problem": "Rising Costs",
                    "severity": "High",
                    "metric": metric,
                    "evidence": f"Costs increased {data['avg_growth']*100:.1f}%"
                })
            if "orders" in metric.lower() and data["trend"] == "down":
                problems.append({
                    "problem": "Falling Orders",
                    "severity": "High",
                    "metric": metric,
                    "evidence": f"Orders declined {abs(data['avg_growth'])*100:.1f}%"
                })
    
    if anomalies:
        problems.append({
            "problem": "Data Anomalies",
            "severity": "Medium",
            "metric": "Multiple",
            "evidence": f"{len(anomalies)} outliers detected"
        })
    
    return problems

def build_hypotheses(problems):
    hypotheses = []
    for p in problems:
        if "Revenue" in p["problem"]:
            hypotheses.append({
                "problem": p["problem"],
                "hypothesis": "Revenue decline associated with reduced orders.",
                "supporting": ["Revenue trending down"],
                "confidence": "Medium"
            })
        elif "Costs" in p["problem"]:
            hypotheses.append({
                "problem": p["problem"],
                "hypothesis": "Cost increase indicates operational changes.",
                "supporting": ["Costs trending up"],
                "confidence": "Medium"
            })
        elif "Orders" in p["problem"]:
            hypotheses.append({
                "problem": p["problem"],
                "hypothesis": "Orders declining due to market factors.",
                "supporting": ["Orders trending down"],
                "confidence": "Medium"
            })
        elif "Anomalies" in p["problem"]:
            hypotheses.append({
                "problem": p["problem"],
                "hypothesis": "Data anomalies indicate unusual events.",
                "supporting": ["Outliers detected"],
                "confidence": "Low"
            })
    return hypotheses

def build_recommendations(problems):
    recommendations = []
    for p in problems:
        if "Revenue" in p["problem"]:
            recommendations.append({
                "problem": p["problem"],
                "recommendation": "Analyze order drop-off. Optimize conversion.",
                "priority": "High"
            })
        elif "Costs" in p["problem"]:
            recommendations.append({
                "problem": p["problem"],
                "recommendation": "Review vendor costs. Renegotiate contracts.",
                "priority": "High"
            })
        elif "Orders" in p["problem"]:
            recommendations.append({
                "problem": p["problem"],
                "recommendation": "Increase marketing. Optimize funnel.",
                "priority": "High"
            })
        elif "Anomalies" in p["problem"]:
            recommendations.append({
                "problem": p["problem"],
                "recommendation": "Verify data collection. Investigate outliers.",
                "priority": "Medium"
            })
    return recommendations

def generate_report(business_type, profile, results):
    report = f"""# RootIQ Analysis Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Business Context
- Type: {business_type}

## Dataset Profile
- Rows: {profile['row_count']}
- Columns: {profile['column_count']}

## Problems Detected
"""
    for p in results.get("problems", []):
        report += f"- {p['problem']} ({p['severity']}): {p['evidence']}\n"
    
    report += "\n## Recommendations\n"
    for r in results.get("recommendations", []):
        report += f"- {r['recommendation']}\n"
    
    return report

# ============================================================================
# GROQ AI WITH USER API KEY INPUT
# ============================================================================

def analyze_with_groq(payload, business_type, api_key, model_name):
    """Call Groq with user-provided API key."""
    
    if not api_key:
        raise AIProviderError("Groq API Key required. Add in sidebar.")
    
    try:
        client = Groq(api_key=api_key)
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are RootIQ analyst. Respond with JSON: {\"executive_summary\": \"text\", \"key_findings\": []}"},
                {"role": "user", "content": f"Business: {business_type}\n\nFindings:\n{json.dumps(payload, indent=2, default=str)}"}
            ],
            temperature=0.2,
            max_tokens=500,
        )
        
        raw_text = response.choices[0].message.content.strip()
        
        try:
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_text:
                raw_text = raw_text.split("```")[1].split("```")[0].strip()
            
            return json.loads(raw_text)
        except:
            return {"executive_summary": raw_text[:300], "key_findings": ["Analysis generated"]}
    
    except Exception as e:
        raise AIProviderError(f"Groq error: {str(e)}")

# ============================================================================
# SIDEBAR - WITH API KEY INPUT
# ============================================================================

st.sidebar.header("📤 Upload & Configure")
uploaded_file = st.sidebar.file_uploader("Upload data (CSV, XLSX, JSON)", type=["csv", "xlsx", "json"])
st.sidebar.divider()

business_type = st.sidebar.text_input("Business Type", placeholder="E-commerce")
dataset_description = st.sidebar.text_area("Dataset Description", placeholder="Sales data")
st.sidebar.divider()

# ========== API KEY INPUT ==========
st.sidebar.header("🔑 Groq API Configuration")

groq_api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    placeholder="gsk_...",
    help="Get from https://console.groq.com/keys"
)

# If no API key in input, try secrets
if not groq_api_key:
    groq_api_key = st.secrets.get("GROQ_API_KEY", "")

groq_model = st.sidebar.selectbox(
    "Groq Model",
    options=[
        "llama-2-70b-4096",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
    ],
    index=2,
    help="Choose model for analysis"
)

if groq_api_key:
    st.sidebar.success("✅ API Key configured")
else:
    st.sidebar.warning("⚠️ No API Key (AI features disabled)")

st.sidebar.divider()

analyze_button = st.sidebar.button("🔬 Analyze", use_container_width=True, type="primary")

# ============================================================================
# MAIN
# ============================================================================

if not uploaded_file:
    st.info("👆 Upload a file")
    st.stop()

if not business_type or not dataset_description:
    st.warning("⚠️ Fill required fields")
    st.stop()

try:
    df, _ = load_data(uploaded_file)
except DataLoadError as e:
    st.error(f"❌ {str(e)}")
    st.stop()

profile = profile_dataset(df)
column_map = detect_columns(df)

if not column_map:
    st.error("❌ No columns detected")
    st.stop()

st.header("📊 Data Overview")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Rows", profile['row_count'])
with col2:
    st.metric("Columns", profile["column_count"])
with col3:
    st.metric("Duplicates", profile["duplicate_rows"])
with col4:
    st.metric("Missing", sum(profile["missing_values"].values()))

st.divider()
st.subheader("🎯 Detected Columns")
for role, col_info in column_map.items():
    st.write(f"🟢 **{role}**: `{col_info['column']}`")

st.divider()

if analyze_button:
    with st.spinner("🔄 Analyzing..."):
        stats = compute_descriptive_stats(df, column_map)
        ts = compute_time_series_analysis(df, column_map)
        anomalies = detect_anomalies(df, column_map)
        correlations = compute_correlations(df, column_map)
        
        problems = detect_problems(ts, anomalies)
        hypotheses = build_hypotheses(problems)
        recommendations = build_recommendations(problems)
        
        st.session_state.results = {
            "stats": stats,
            "ts": ts,
            "anomalies": anomalies,
            "correlations": correlations,
            "problems": problems,
            "hypotheses": hypotheses,
            "recommendations": recommendations,
        }
        
        # Try AI if API key available
        if groq_api_key and problems:
            try:
                payload = {
                    "row_count": profile["row_count"],
                    "problems": problems[:5],
                    "stats": stats,
                }
                ai_results = analyze_with_groq(payload, business_type, groq_api_key, groq_model)
                st.session_state.ai_results = ai_results
            except AIProviderError as e:
                st.info(f"ℹ️ {str(e)}")
                st.session_state.ai_results = None
        
        st.session_state.analysis_complete = True
    
    st.success("✅ Complete!")

if st.session_state.analysis_complete:
    results = st.session_state.results
    ai_results = st.session_state.ai_results
    
    st.header("📝 Summary")
    if ai_results:
        st.write(ai_results.get("executive_summary", ""))
    else:
        st.info(f"**{len(results['problems'])} problems detected** from data analysis.")
    st.divider()
    
    st.header("📈 Metrics")
    if results["stats"]:
        cols = st.columns(min(4, len(results["stats"])))
        for idx, (metric, values) in enumerate(results["stats"].items()):
            with cols[idx % len(cols)]:
                st.metric(metric, f"{values['mean']:.0f}")
    st.divider()
    
    if results["ts"] and results["ts"].get("df_ts") is not None:
        st.header("📊 Trends")
        df_ts = results["ts"]["df_ts"].reset_index()
        for col in df_ts.columns:
            if "revenue" in col.lower():
                fig = px.line(df_ts, x=df_ts.columns[0], y=col, markers=True)
                st.plotly_chart(fig, use_container_width=True)
                break
        st.divider()
    
    if results["problems"]:
        st.header("⚠️ Problems")
        for p in results["problems"]:
            severity_emoji = "🔴" if p["severity"] == "High" else "🟠"
            with st.expander(f"{severity_emoji} {p['problem']}"):
                st.write(f"**Evidence**: {p['evidence']}")
        st.divider()
    
    if results["hypotheses"]:
        st.header("🔍 Root Causes")
        for h in results["hypotheses"]:
            with st.expander(f"{h['hypothesis']}"):
                st.write(f"**Confidence**: {h['confidence']}")
                for e in h["supporting"]:
                    st.write(f"✓ {e}")
        st.divider()
    
    if results["recommendations"]:
        st.header("💡 Recommendations")
        for r in results["recommendations"]:
            with st.expander(f"{r['recommendation']}"):
                st.write(f"**Priority**: {r['priority']}")
        st.divider()
    
    if results["stats"]:
        st.header("📈 Statistics")
        st.dataframe(pd.DataFrame(results["stats"]).T.round(2), use_container_width=True)
        st.divider()
    
    if results["correlations"]:
        st.header("🔗 Correlations")
        st.dataframe(pd.DataFrame(results["correlations"]), use_container_width=True)
        st.divider()
    
    st.header("🔎 Data")
    with st.expander("View Raw Data"):
        st.dataframe(df.head(20), use_container_width=True)
    
    st.divider()
    
    st.header("📥 Download")
    report = generate_report(business_type, profile, results)
    st.download_button(
        label="📄 Download Report",
        data=report,
        file_name=f"rootiq_{datetime.now().strftime('%Y%m%d')}.md",
        mime="text/markdown"
    )
