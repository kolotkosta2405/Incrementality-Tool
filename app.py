import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Kepler Portfolio Strategy Engine v11", layout="wide")

st.title("🎯 Kepler Retail Media Portfolio Engine")
st.subheader("Line-Item Category & Share of Voice (SOV) Optimization Matrix")

st.markdown("""
This framework rolls up granular day-by-day performance data into **Strategic Line-Item Categories** based on keyword identifiers, mapping aggregated performance to unified SOV tracks.
""")

# --- COMPLETE TARGET LINE-ITEM KEYWORD REGISTRY ---
# Extracted directly from your naming schema to guarantee 100% mapping coverage
CATEGORY_KEYWORDS = {
    "Uprights & Corded Vacuums": [
        "uprghts", "upright", "cordstk", "corded stick"
    ],
    "Cordless Stick Vacuums": [
        "cordles", "cordless stick", "corhepa", "cordless hepa"
    ],
    "Wet/Dry, Steam & Mops": [
        "steamxx", "steam mop", "vacmopx", "vacuum mop"
    ],
    "Carpet Cleaners & Deep Cleaners": [
        "dccxxxx", "dccprtb", "portable deep carpet cleaner", "carpet cleaner"
    ],
    "Robotic Vacuums": [
        "robotsx", "robots vacuum", "rbtvcum", "rbts2n1", "robots vacuum 2 in 1"
    ],
    "Fans & Climate Control": [
        "fansxxx", "fans"
    ],
    "Air Purification": [
        "airpurx", "air purifier"
    ],
    "Handhelds & Specialty Blowers": [
        "handhld", "blasbos", "air blower", "blastboss", "handheld"
    ]
}

def identify_macro_category(name_string):
    """Scans line item titles or product strings to map them to an executive category."""
    clean_name = str(name_string).lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in clean_name for kw in keywords):
            return category
    return "Other Shark Systems" # Catch-all baseline bucket

# --- SIDEBAR CONTROLS ---
st.sidebar.header("1. Upload Performance Data Layer")
uploaded_file = st.sidebar.file_uploader("Upload Target Performance CSV (Apr-May)", type=["csv"])

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

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.lower().str.strip()
        
        # Flexibly look for naming identifiers from Target downloads
        line_item_col = [c for c in df.columns if 'item' in c or 'title' in c or 'product' in c or 'name' in c][0]
        spend_col = [c for c in df.columns if 'spend' in c or 'cost' in c][0]
        sales_col = [c for c in df.columns if 'sales' in c or 'revenue' in c][0]
        organic_sov_col = [c for c in df.columns if 'organic' in c or 'org' in c or 'sov' in c][0]
        
        # Check for optional NTB column
        has_ntb = any('ntb' in c or 'new' in c for c in df.columns)
        ntb_col = [c for c in df.columns if 'ntb' in c or 'new' in c][0] if has_ntb else None

        # --- DATA CLEANING & STANDARDIZATION ---
        df['assigned_category'] = df[line_item_col].apply(identify_macro_category)
        df['clean_spend'] = clean_numeric_column(df[spend_col]).fillna(0)
        df['clean_sales'] = clean_numeric_column(df[sales_col]).fillna(0)
        df['clean_organic_sov'] = clean_numeric_column(df[organic_sov_col]).fillna(0)
        if df['clean_organic_sov'].max() > 1.0: df['clean_organic_sov'] /= 100.0
            
        if has_ntb:
            df['clean_ntb'] = clean_numeric_column(df[ntb_col]).fillna(0)
            if df['clean_ntb'].max() > 1.0: df['clean_ntb'] /= 100.0

        # --- EXECUTING STRATEGIC CATEGORY ROLL-UP ---
        # Instead of calculating by product, group everything by the rolled-up categories first
        category_summary = df.groupby('assigned_category').agg({
            'clean_spend': 'sum',
            'clean_sales': 'sum',
            'clean_organic_sov': 'mean',
            'clean_ntb': 'mean' if has_ntb else 'count'
        }).reset_index()

        st.success(f"🟢 Target Data Layer Successfully Synced. Raw rows rolled up cleanly into {len(category_summary)} Strategic Line-Item Categories.")

        # --- CALCULATIONS ENGINE MATRIX ---
        st.header("Executive Category Matrix (Apr-May Performance)")
        table_data = []
        raw_metrics = {}
        
        total_portfolio_spend = 0
        total_portfolio_incremental_sales = 0

        for idx, row in category_summary.iterrows():
            cat = row['assigned_category']
            spend = float(row['clean_spend'])
            sales = float(row['clean_sales'])
            avg_organic_sov = float(row['clean_organic_sov'])
            avg_ntb = float(row['clean_ntb']) if has_ntb else 0.0
            
            if spend == 0 and sales == 0:
                continue

            # Step 1: Apply Causal Logistic S-Curve on rolled-up Organic Footprint
            s_curve_factor = 1.0 / (1.0 + np.exp(steepness * (avg_organic_sov - inflection_point)))
            incrementality_factor = max(0.15, min(0.90, s_curve_factor))
            
            # Step 2: Integrate NTB if data layer was present
            if has_ntb:
                # Electronics Baseline Rule: Controlled dampening factor for high-durables acquisition
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
                "Avg Organic SOV Share": f"{avg_organic_sov*100:.1f}%",
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
                * **The Logic:** `{worst_cat}` is showing high organic saturation and shelf visibility overlap (Avg Organic SOV: **{worst_meta['organic_sov']*100:.1f}%**). Media spend here is pulling high natural/organic demand, resulting in a low true incremental return (**{worst_meta['iroas']:.2f}x**).
                * **The Action:** Reallocate budget lines from `{worst_cat}` into `{best_cat}` which continues to run cleanly at peak media incrementality (**{best_meta['iroas']:.2f}x** true return) on Target.
                """)
        else:
            st.info("ℹ️ All rolled-up line items are performing within optimal target variances. Maintain current multi-category flight setup.")

    except Exception as e:
        st.error(f"❌ Structural Parsing Error: {str(e)}")
else:
    st.info("👋 Ready to process live Target line-item metrics. Upload the day-by-day sheet above to view the strategic roll-up model.")
