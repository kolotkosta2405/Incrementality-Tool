import streamlit as st
import pandas as pd
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Kepler Multi-Layer Strategy Engine v18", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎯 Kepler Retail Media Portfolio Engine")
st.subheader("Omni-Channel Multi-Layer Ingestion & Causal Mapping Matrix")

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
    clean_name = str(name_string).lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in clean_name for kw in keywords):
            return category
    return "Other Shark Systems"

# --- SIDEBAR: ZERO-TOUCH INTELLIGENT DROP ZONE ---
st.sidebar.header("1. Ingest Data Layers")
uploaded_files = st.sidebar.file_uploader(
    "Drag & drop all Target CSV files here simultaneously", 
    type=["csv"], 
    accept_multiple_files=True
)

df_perf, df_prod, df_kw, df_brand = None, None, None, None

if uploaded_files:
    for file in uploaded_files:
        try:
            preview = pd.read_csv(file, nrows=2)
            preview.columns = preview.columns.str.lower().str.strip()
            file.seek(0)
            
            if 'spend' in preview.columns or 'cost' in preview.columns:
                df_perf = pd.read_csv(file)
                st.sidebar.success(f"📊 Performance: {file.name}")
            elif 'product' in preview.columns or 'asin' in preview.columns:
                df_prod = pd.read_csv(file)
                st.sidebar.success(f"📦 Product SOV: {file.name}")
            elif 'keyword' in preview.columns or 'term' in preview.columns:
                df_kw = pd.read_csv(file)
                st.sidebar.success(f"🔑 Keyword SOV: {file.name}")
            elif 'brand' in preview.columns and not ('keyword' in preview.columns or 'product' in preview.columns):
                df_brand = pd.read_csv(file)
                st.sidebar.success(f"🏢 Brand SOV: {file.name}")
        except Exception as e:
            st.sidebar.error(f"Error reading {file.name}: {e}")

st.sidebar.header("2. Model Adjusters & Guardrails")
with st.sidebar.expander("⚙️ Strategic Constraints", expanded=True):
    inflection_point = st.sidebar.slider("S-Curve Inflection Point (x₀)", 0.20, 0.60, 0.35, 0.05)
    steepness = st.sidebar.slider("Decay Steepness (k)", 5.0, 15.0, 8.50, 0.5)
    max_allowed_iroas = st.sidebar.slider("Executive Sanity Ceiling", 3.0, 10.0, 5.50, 0.5)

# --- MAIN ENGINE PROCESSING MATRIX ---
if df_perf is not None and (df_prod is not None or df_kw is not None or df_brand is not None):
    try:
        # --- LAYER 1: UNIFY PERFORMANCE COLUMNS ---
        df_perf.columns = df_perf.columns.str.lower().str.strip()
        
        line_item_col = [c for c in df_perf.columns if 'item' in c or 'title' in c or 'product' in c or 'name' in c][0]
        spend_col = [c for c in df_perf.columns if 'spend' in c or 'cost' in c][0]
        sales_col = [c for c in df_perf.columns if 'sales' in c or 'revenue' in c][0]
        ntb_col = [c for c in df_perf.columns if 'ntb' in c or 'new' in c][0]

        df_perf['assigned_category'] = df_perf[line_item_col].apply(identify_macro_category)
        
        # Pure numeric parsing without destructive string methods
        df_perf['clean_spend'] = pd.to_numeric(df_perf[spend_col], errors='coerce').fillna(0.0)
        df_perf['clean_sales'] = pd.to_numeric(df_perf[sales_col], errors='coerce').fillna(0.0)
        df_perf['clean_ntb'] = pd.to_numeric(df_perf[ntb_col], errors='coerce').fillna(0.0)
        
        # Dynamic scaler: handles both formats (0.93 or 93.0) seamlessly
        if df_perf['clean_ntb'].max() > 1.0: 
            df_perf['clean_ntb'] /= 100.0

        # --- LAYER 2: PROCESS THE CASCADING SOV LOOKUPS ---
        prod_sov_dict, kw_sov_dict, brand_baseline = {}, {}, 0.30
        
        if df_brand is not None:
            df_brand.columns = df_brand.columns.str.lower().str.strip()
            b_org_col = [c for c in df_brand.columns if 'organic' in c or 'sov' in c or 'share' in c][0]
            df_brand['clean_sov'] = pd.to_numeric(df_brand[b_org_col], errors='coerce').fillna(0.0)
            if df_brand['clean_sov'].max() > 1.0: df_brand['clean_sov'] /= 100.0
            brand_baseline = df_brand['clean_sov'].mean()

        if df_kw is not None:
            df_kw.columns = df_kw.columns.str.lower().str.strip()
            kw_label_col = [c for c in df_kw.columns if 'keyword' in c or 'term' in c][0]
            kw_org_col = [c for c in df_kw.columns if 'organic' in c or 'sov' in c or 'share' in c][0]
            df_kw['assigned_category'] = df_kw[kw_label_col].apply(identify_macro_category)
            df_kw['clean_sov'] = pd.to_numeric(df_kw[kw_org_col], errors='coerce').fillna(0.0)
            if df_kw['clean_sov'].max() > 1.0: df_kw['clean_sov'] /= 100.0
            kw_sov_dict = df_kw.groupby('assigned_category')['clean_sov'].mean().to_dict()

        if df_prod is not None:
            df_prod.columns = df_prod.columns.str.lower().str.strip()
            p_label_col = [c for c in df_prod.columns if 'product' in c or 'title' in c or 'asin' in c][0]
            p_org_col = [c for c in df_prod.columns if 'organic' in c or 'sov' in c or 'share' in c][0]
            df_prod['assigned_category'] = df_prod[p_label_col].apply(identify_macro_category)
            df_prod['clean_sov'] = pd.to_numeric(df_prod[p_org_col], errors='coerce').fillna(0.0)
            if df_prod['clean_sov'].max() > 1.0: df_prod['clean_sov'] /= 100.0
            prod_sov_dict = df_prod.groupby('assigned_category')['clean_sov'].mean().to_dict()

        # --- LAYER 3: CORE COMPILATION ENGINE ---
        st.header("Executive Category Matrix (Cascaded SOV Integration)")
        
        category_summary = df_perf.groupby('assigned_category').agg({
            'clean_spend': 'sum',
            'clean_sales': 'sum',
            'clean_ntb': 'mean'
        }).reset_index()

        table_data = []
        raw_metrics = {}
        total_portfolio_spend = 0
        total_portfolio_incremental_sales = 0

        for idx, row in category_summary.iterrows():
            cat = row['assigned_category']
            spend = float(row['clean_spend'])
            sales = float(row['clean_sales'])
            avg_ntb = float(row['clean_ntb'])
            
            if spend == 0 and sales == 0:
                continue

            assigned_sov = brand_baseline
            sov_source = "Brand Default Layer"
            
            if cat in kw_sov_dict:
                assigned_sov = kw_sov_dict[cat]
                sov_source = "Keyword Tab Match"
            if cat in prod_sov_dict:
                assigned_sov = prod_sov_dict[cat]
                sov_source = "Product Tab Match (Max Precision)"

            # Step 1: Logistic S-Curve Cannibalization Filter
            s_curve_factor = 1.0 / (1.0 + np.exp(steepness * (assigned_sov - inflection_point)))
            base_incrementality = max(0.15, min(0.90, s_curve_factor))
            
            # Step 2: Customer Acquisition Optimization Layer
            final_incrementality = base_incrementality + (avg_ntb * 0.05)
            final_incrementality = min(0.95, max(0.10, final_incrementality))
            
            # Step 3: Financial Calculations
            incremental_sales = sales * final_incrementality
            calculated_iroas = incremental_sales / spend if spend > 0 else 0
            
            is_capped = False
            if calculated_iroas > max_allowed_iroas:
                calculated_iroas = max_allowed_iroas
                incremental_sales = spend * calculated_iroas
                is_capped = True

            total_portfolio_spend += spend
            total_portfolio_incremental_sales += incremental_sales
            
            raw_metrics[cat] = {
                'spend': spend,
                'sales': sales,
                'iroas': calculated_iroas,
                'organic_sov': assigned_sov,
                'ntb': avg_ntb,
                'inc_factor': final_incrementality,
                'capped': is_capped
            }
            
            table_data.append({
                "Line-Item Category": cat,
                "Resolved Organic SOV": f"{assigned_sov*100:.1f}%",
                "SOV Mapping Source": sov_source,
                "New-To-Brand (NTB) %": f"{avg_ntb*100:.1f}%",
                "Total Media Invested": f"${spend:,.2f}",
                "Total Attributed Sales": f"${sales:,.2f}",
                "True Incremental Sales": f"${incremental_sales:,.2f}",
                "Capped Category iROAS": f"{calculated_iroas:.2f}x"
            })
            
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)

        # --- EXECUTIVE SUMMARY TOTALS ---
        st.header("Executive Portfolio Summary")
        portfolio_iroas = total_portfolio_incremental_sales / total_portfolio_spend if total_portfolio_spend > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Blended Ad Investment", f"${total_portfolio_spend:,.2f}")
        col2.metric("True Incremental Volume", f"${total_portfolio_incremental_sales:,.2f}")
        col3.metric("Blended Portfolio iROAS", f"{portfolio_iroas:.2f}x")

        # --- DETAILED CATEGORY ACTION MODULES ---
        st.header("🎯 Deep Category Action Plans")
        rec_cols = st.columns(2)
        for idx, (cat, metrics) in enumerate(raw_metrics.items()):
            col_to_use = rec_cols[idx % 2]
            with col_to_use:
                with st.expander(f"📋 Strategic Plan: {cat}", expanded=True):
                    st.write(f"**Current iROAS:** {metrics['iroas']:.2f}x | **Organic SOV:** {metrics['organic_sov']*100:.1f}%")
                    
                    if metrics['organic_sov'] > inflection_point:
                        st.error("⚠️ **Status: Cannibalization Risk (High Organic Overlap)**")
                        st.markdown(f"""
                        * **Analysis:** Organic footprint is at **{metrics['organic_sov']*100:.1f}%**, which sits past our model's inflection threshold of **{inflection_point*100:.1f}%**. Paid ads are pulling conversions that naturally would have occurred.
                        * **Tactical Directive:** Scale back baseline brand search bidding by **10-15%**. Shift those funds toward generic non-brand category terms or conquesting targets where your organic visibility is weaker.
                        """)
                    elif metrics['iroas'] >= (max_allowed_iroas - 0.5) or metrics['capped']:
                        st.success("🔥 **Status: High-Efficiency Market Expansion**")
                        st.markdown(f"""
                        * **Analysis:** Running at maximum incremental efficiency. High NTB rates (**{metrics['ntb']*100:.1f}%**) combined with clean organic headroom mean your spend is driving genuine top-line revenue growth.
                        * **Tactical Directive:** Uncap budgets for this category immediately. Increase investment by **20%** or maintain uncapped budget flights to capture all available incremental conversion volumes.
                        """)
                    else:
                        st.warning("⚖️ **Status: Stable Mid-Tier Performance**")
                        st.markdown(f"""
                        * **Analysis:** The media footprint is balanced. Incrementality factor is calculated at **{metrics['inc_factor']*100:.1f}%**, keeping efficiency stable without immediate signs of severe keyword fatigue.
                        * **Tactical Directive:** Maintain current run-rates. Optimize performance internally by adjusting line-item bid modifiers and refreshing creative concepts rather than making macro budget changes.
                        """)

        # --- THE MATHEMATICAL CALCULATIONS APPENDIX ---
        st.markdown("---")
        st.header("🧮 Model Methodology & Causal Logic Appendix")
        st.markdown(r"""
        This dashboard replaces standard linear attribution models with a multi-layered causal framework designed to calculate **True Incremental Return on Ad Spend ($iROAS$)**. The model operates through a three-stage mathematical sequence applied to every rolled-up product category:

        ### 1. The Non-Linear Organic Cannibalization Filter (S-Curve)
        Paid media efficiency is fundamentally dependent on baseline organic shelf presence. To model how high organic Share of Voice ($SOV_{org}$) cannibalizes paid ad credit, we route the resolved organic share through a **Logistic S-Curve Decay Function**:
        
        $$f(SOV_{org}) = \frac{1}{1 + e^{k \cdot (SOV_{org} - x_0)}}$$
        
        *Where:*
        * $x_0$ represent the **Inflection Point** (set via sidebar; defaults to $0.35$). This is the threshold where cannibalization begins accelerating.
        * $k$ represents the **Decay Steepness** (set via sidebar). This controls how aggressively ad credit drops off as organic dominance nears $100\%$.
        
        This base factor is bounded between a baseline floor of $0.15$ and a ceiling of $0.90$ to account for standard baseline market velocity.

        ### 2. New-To-Brand (NTB) Acquisition Scaling
        To ensure the model rewards customer acquisition, we add an optimization layer linked to the category's average New-To-Brand sales percentage ($NTB\%$):
        
        $$\text{Final Incrementality Factor } (\alpha) = f(SOV_{org}) + (NTB\% \cdot 0.05)$$
        
        This final scalar $\alpha$ represents the true percentage of sales directly caused by your retail media spend, capped strictly at a maximum ceiling of $95\%$ and a absolute floor of $10\%$.

        ### 3. Financial Synthesis & Guardrail Enforcement
        Once the true incremental volume is isolated, the final metric values are synthesized via:
        
        $$\text{True Incremental Sales} = \text{Total Attributed Sales} \times \alpha$$
        
        $$iROAS = \frac{\text{True Incremental Sales}}{\text{Total Media Invested}}$$
        
        If low spend levels coupled with large organic volumes produce a volatile mathematical anomaly, the system enforces the **Executive Sanity Ceiling** ($Ceiling_{exec}$):
        
        $$\text{If } iROAS > Ceiling_{exec}, \text{ then } iROAS = Ceiling_{exec} \text{ and } \text{True Incremental Sales} = \text{Spend} \times Ceiling_{exec}$$
        """)

    except Exception as e:
        st.error(f"❌ Processing Error Across Ingested Layers: {str(e)}")
else:
    st.info("👋 System standing by. Drop all exported Target data files (Performance + SOV layers) together into the file box to initialize.")
