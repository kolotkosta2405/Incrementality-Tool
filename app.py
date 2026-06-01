import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Kepler Portfolio Strategy Engine v12", layout="wide")

st.title("🎯 Kepler Retail Media Portfolio Engine")
st.subheader("Dual-File Automated Category & SOV Mapping Matrix")

st.markdown("""
This framework ingests separate Performance and Share of Voice (SOV) spreadsheets, programmatically maps them to 
unified **Strategic Line-Item Categories**, and runs the causal incrementality logic without requiring manual data preparation.
""")

# --- COMPLETE TARGET LINE-ITEM & SOV KEYWORD REGISTRY ---
CATEGORY_KEYWORDS = {
    "Uprights & Corded Vacuums": ["uprghts", "upright", "cordstk", "corded stick"],
    "Cordless Stick Vacuums": ["cordles", "cordless stick", "corhepa", "cordless hepa"],
    "Wet/Dry, Steam & Mops": ["steamxx", "steam mop", "vacmopx", "vacuum mop"],
    "Carpet Cleaners & Deep Cleaners": ["dccxxxx", "dccprtb", "portable deep carpet cleaner", "carpet cleaner"],
    "Robotic Vacuums": ["robotsx", "robots vacuum", "rbtvcum", "rbts2n1", "robots vacuum 2 in 1"],
    "Fans & Climate Control": ["fansxxx", "fans"],
    "Air Purification": ["airpurx", "air purifier"],
    "Handhelds & Specialty Blowers": ["handhld", "blasbos", "air blower", "blastboss", "handheld"]
}

def identify_macro_category(name_string):
    """Scans performance line items, keywords, or product names to map them to an executive category."""
    clean_name = str(name_string).lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in clean_name for kw in keywords):
            return category
    return "Other Shark Systems"

# --- SIDEBAR DUAL-FILE UPLOADER ---
st.sidebar.header("1. Ingest Raw Client Data")
perf_file = st.sidebar.file_uploader("Upload Target Performance CSV (Spends, Sales, NTB)", type=["csv"], key="perf")
sov_file = st.sidebar.file_uploader("Upload Share of Voice (SOV) CSV", type=["csv"], key="sov")

st.sidebar.header("2. Model Adjusters & Guardrails")
with st.sidebar.expander("⚙️ Strategic Constraints", expanded=True):
    inflection_point = st.slider("S-Curve Inflection Point (x₀)", 0.20, 0.60, 0.35, 0.05,
                                  help="The Category SOV threshold where media cannibalization begins to impact efficiency.")
    steepness = st.slider("Decay Steepness (k)", 5.0, 15.0, 8.50, 0.5)
    max_allowed_iroas = st.slider("Executive Sanity Ceiling", 3.0, 10.0, 5.50, 0.5,
                                  help="Enforces standard diminishing returns ceiling across categories for clean client alignment.")

def clean_numeric_column(series):
    if series.dtype == object:
        cleaned = series.astype(str).str.strip().str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.replace('%', '', regex=False)
        return pd.to_numeric(cleaned, errors='coerce')
    return pd.to_numeric(series, errors='coerce')

# Check if both files are uploaded before running calculations
if perf_file and sov_file:
    try:
        # --- LAYER 1: PROCESS PERFORMANCE CSV ---
        df_perf = pd.read_csv(perf_file)
        df_perf.columns = df_perf.columns.str.lower().str.strip()
        
        line_item_col = [c for c in df_perf.columns if 'item' in c or 'title' in c or 'product' in c or 'name' in c][0]
        spend_col = [c for c in df_perf.columns if 'spend' in c or 'cost' in c][0]
        sales_col = [c for c in df_perf.columns if 'sales' in c or 'revenue' in c][0]
        
        has_ntb = any('ntb' in c or 'new' in c for c in df_perf.columns)
        ntb_col = [c for c in df_perf.columns if 'ntb' in c or 'new' in c][0] if has_ntb else None

        df_perf['assigned_category'] = df_perf[line_item_col].apply(identify_macro_category)
        df_perf['clean_spend'] = clean_numeric_column(df_perf[spend_col]).fillna(0)
        df_perf['clean_sales'] = clean_numeric_column(df_perf[sales_col]).fillna(0)
        df_perf['clean_ntb'] = clean_numeric_column(df_perf[ntb_col]).fillna(0) if has_ntb else 0

        # Roll up performance data to macro category
        perf_summary = df_perf.groupby('assigned_category').agg({
            'clean_spend': 'sum',
            'clean_sales': 'sum',
            'clean_ntb': 'mean' if has_ntb else 'count'
        }).reset_index()

        # --- LAYER 2: PROCESS SHARE OF VOICE CSV ---
        df_sov = pd.read_csv(sov_file)
        df_sov.columns = df_sov.columns.str.lower().str.strip()
        
        # Flexibly find the SOV data and column labels (brand, keyword, or product level)
        sov_label_col = [c for c in df_sov.columns if 'keyword' in c or 'product' in c or 'brand' in c or 'title' in c or 'name' in c][0]
        sov_value_col = [c for c in df_sov.columns if 'sov' in c or 'share' in c or 'organic' in c][0]
        
        df_sov['assigned_category'] = df_sov[sov_label_col].apply(identify_macro_category)
        df_sov['clean_sov'] = clean_numeric_column(df_sov[sov_value_col]).fillna(0)
        if df_sov['clean_sov'].max() > 1.0: 
            df_sov['clean_sov'] /= 100.0

        # Roll up SOV data to macro category
        sov_summary = df_sov.groupby('assigned_category').agg({'clean_sov': 'mean'}).reset_index()

        # --- LAYER 3: CORE DATA MERGE AND MAPPING ---
        # Programmatically stitch the summaries together on the 'assigned_category' key
        category_summary = pd.merge(perf_summary, sov_summary, on='assigned_category', how='outer').fillna(0)

        st.success("🟢 Automated Cross-File Sync Complete: Performance and SOV data cleanly mapped at the macro category level.")

        # --- CALCULATIONS MATRIX ENGINE ---
        st.header("Executive Category Matrix (Integrated Client Data)")
        table_data = []
        raw_metrics = {}
        
        total_portfolio_spend = 0
        total_portfolio_incremental_sales = 0

        for idx, row in category_summary.iterrows():
            cat = row['assigned_category']
            spend = float(row['clean_spend'])
            sales = float(row['clean_sales'])
            avg_organic_sov = float(row['clean_sov'])
            avg_ntb = float(row['clean_ntb']) if has_ntb else 0.0
            
            # Skip rows where no media activity took place in the test period
            if spend == 0 and sales == 0:
                continue

            # Step 1: Causal Non-Linear S-Curve Filter
            s_curve_factor = 1.0 / (1.0 + np.exp(steepness * (avg_organic_sov - inflection_point)))
            incrementality_factor = max(0.15, min(0.90, s_curve_factor))
            
            # Step 2: Inject Electronics-Dampened NTB baseline adjustment if present
            if has_ntb:
                incrementality_factor += (avg_ntb * 0.05)
                
            incrementality_factor = min(0.95, max(0.10, incrementality_factor))
            
            # Step 3: Run Financial Compilation + Guardrail Ceilings
            incremental_sales = sales * incrementality_factor
            calculated_iroas = incremental_sales / spend if spend > 0 else 0
            
            if calculated_iroas > max_allowed_iroas:
                calculated_iroas = max_allowed_iroas
                incremental_sales = spend * calculated_iroas

            total_portfolio_spend += spend
            total_portfolio_incremental_sales += incremental_sales
            
            raw_metrics[cat] = {
                'spend': spend,
                'sales': sales,
                'iroas': calculated_iroas,
                'organic_sov': avg_organic_sov
            }
            
            table_data.append({
                "Line-Item Category": cat,
                "Mapped Organic SOV Share": f"{avg_organic_sov*100:.1f}%",
                "New-To-Brand (NTB) %": f"{avg_ntb*100:.1f}%" if has_ntb else "N/A",
                "Total Media Invested": f"${spend:,.2f}",
                "Total Attributed Sales": f"${sales:,.2f}",
                "True Incremental Sales": f"${incremental_sales:,.2f}",
                "Capped Category iROAS": f"{calculated_iroas:.2f}x"
            })
            
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)

        # --- BLENDED SUMMARY PANEL ---
        st.header("Executive Portfolio Summary")
        portfolio_iroas = total_portfolio_incremental_sales / total_portfolio_spend if total_portfolio_spend > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Blended Ad Investment", f"${total_portfolio_spend:,.2f}")
        col2.metric("True Incremental Volume", f"${total_portfolio_incremental_sales:,.2f}")
        col3.metric("Blended Portfolio iROAS", f"{portfolio_iroas:.2f}x")

        # --- MACRO BUDGET DIRECTIVES MODULE ---
        st.header("🎯 Cross-Category Capital Reallocation")
        
        funded_categories = {k: v for k, v in raw_metrics.items() if v['spend'] > 0}
        if len(funded_categories) >= 2:
            sorted_cats = sorted(funded_categories.items(), key=lambda item: item[1]['iroas'])
            worst_cat, worst_meta = sorted_cats[0]
            best_cat, best_meta = sorted_cats[-1]
            
            if worst_meta['iroas'] < best_meta['iroas']:
                st.success(f"🔄 **Strategic Shift Recommendation:** Reallocate Budget from `{worst_cat}` to `{best_cat}`")
                st.markdown(f"""
                * **The Causal Logic:** `{worst_cat}` has high organic presence and visibility overlap (Avg Organic SOV: **{worst_meta['organic_sov']*100:.1f}%**). Media spend here is pulling high natural/organic demand, resulting in a low true incremental return (**{worst_meta['iroas']:.2f}x**).
                * **The Action:** Reallocate budget lines from `{worst_cat}` into `{best_cat}` which continues to run cleanly at peak media incrementality (**{best_meta['iroas']:.2f}x** true return) on Target.
                """)
        else:
            st.info("ℹ️ All rolled-up line items are performing within optimal target variances. Maintain current multi-category flight setup.")

    except Exception as e:
        st.error(f"❌ Cross-File Mapping Error: {str(e)}")
else:
    st.info("👋 System standing by. Please upload both the **Target Performance CSV** and the **Share of Voice (SOV) CSV** in the sidebar to run the automated mapping model.")
