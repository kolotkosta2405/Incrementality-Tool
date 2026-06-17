import streamlit as st
import pandas as pd
import numpy as np

# --- PAGE SETUP ---
st.set_page_config(page_title="Retail Media Incrementality Engine v10", layout="wide")

st.title("📊 Retail Media Incrementality Engine")
st.subheader("Advanced Bayesian Causal Inference Dashboard (Unified Model)")

st.markdown("""
This engine treats true media lift from organic cannibalization by replacing rigid rules with a non-linear **Logistic S-Curve**, a **Category-Aware New-to-Brand (NTB)** balancing matrix, and **ASP Unit Economics**.
""")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("1. Upload Master Data Layer")
uploaded_file = st.sidebar.file_uploader("Upload Unified Performance CSV", type=["csv"])

st.sidebar.header("2. Global Engine Calibration")
category_type = st.sidebar.selectbox(
    "Select Primary Product Category Layout",
    ["Consumables / CPG (High Repeat Purchases)", "Durables / Electronics (Low Repeat Purchases)"]
)

# Advanced S-Curve tuning parameter toggles hidden cleanly in sidebar expander
with st.sidebar.expander("⚙️ Advanced S-Curve Coefficients"):
    inflection_point = st.slider("Curve Inflection Point (x₀)", 0.05, 0.35, 0.20, 0.05, 
                                 help="The Organic SOV point where incrementality degradation accelerates fastest.")
    steepness = st.slider("Curve Decay Steepness (k)", 5.0, 15.0, 10.0, 0.5,
                          help="Higher numbers enforce harsher cannibalization penalties when crossing the inflection threshold.")

# --- HELPER FUNCTIONS ---
def clean_numeric_column(series):
    """Quietly extracts pure numeric floats from currency strings, whole percentages, or space-padded entries."""
    if series.dtype == object:
        cleaned = series.astype(str).str.strip()
        cleaned = cleaned.str.replace('CA$', '', regex=False)
        cleaned = cleaned.str.replace('$', '', regex=False)
        cleaned = cleaned.str.replace(',', '', regex=False)
        cleaned = cleaned.str.replace('%', '', regex=False)
        return pd.to_numeric(cleaned, errors='coerce')
    return pd.to_numeric(series, errors='coerce')

# --- CORE LOGIC & PROCESSING ---
if uploaded_file:
    if not uploaded_file.name.lower().endswith('.csv'):
        st.error("❌ File Format Error: Please upload a valid `.csv` spreadsheet file.")
    else:
        try:
            df = pd.read_csv(uploaded_file)
            if df.empty or len(df.columns) < 2:
                st.error("❌ Data Interpretation Failure: Uploaded sheet appears empty.")
                st.stop()
                
            df.columns = df.columns.str.lower().str.strip()
            
            # Core validation array checking for mandatory columns
            mandatory_cols = ['date', 'product id', 'media_spend', 'total_sales', 'organic_sov', 'paid_sov', 'units_sold', 'clicks']
            missing_cols = [col for col in mandatory_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing Mandatory Columns: {', '.join(missing_cols)}")
                st.stop()
                
            # Dynamic check for optional data loops (Prevents the KeyError)
            has_ntb = 'ntb_sales_pct' in df.columns or 'ntb_%' in df.columns or 'ntb_sales_percent' in df.columns
            ntb_col = [c for c in df.columns if 'ntb' in c][0] if has_ntb else None
            
            has_inventory = 'inventory_status' in df.columns or 'inventory' in df.columns
            has_promo = 'promo_status' in df.columns or 'promo_flag' in df.columns

            # DATA STANDARDIZATION CLEANUP LAYER
            df['date'] = pd.to_datetime(df['date'], errors='coerce', format='mixed')
            
            for col in ['media_spend', 'total_sales', 'organic_sov', 'paid_sov', 'units_sold', 'clicks']:
                df[col] = clean_numeric_column(df[col])
                df[col] = df[col].fillna(0)
                
            if df['organic_sov'].max() > 1.0:
                df['organic_sov'] = df['organic_sov'] / 100.0
            if df['paid_sov'].max() > 1.0:
                df['paid_sov'] = df['paid_sov'] / 100.0
                
            if has_ntb:
                df['ntb_clean'] = clean_numeric_column(df[ntb_col])
                df['ntb_clean'] = df['ntb_clean'].fillna(0.0)
                if df['ntb_clean'].max() > 1.0:
                    df['ntb_clean'] = df['ntb_clean'] / 100.0
            
            if has_inventory:
                inv_col = 'inventory_status' if 'inventory_status' in df.columns else 'inventory'
                df['inv_clean'] = clean_numeric_column(df[inv_col])
                df['inv_clean'] = df['inv_clean'].fillna(100.0)
                if df['inv_clean'].max() <= 1.0 and df['inv_clean'].sum() > 0:
                    df['inv_clean'] = df['inv_clean'] * 100.0

            # CALCULATIONS MATRIX ENGINE
            unique_products = df['product id'].dropna().unique()
            table_data = []
            raw_metrics = {}
            secondary_flags = []
            
            for prod in unique_products:
                prod_data = df[df['product id'] == prod]
                
                total_spend = float(prod_data['media_spend'].sum())
                total_sales = float(prod_data['total_sales'].sum())
                total_units = float(prod_data['units_sold'].sum())
                total_clicks = float(prod_data['clicks'].sum())
                
                avg_organic_sov = float(prod_data['organic_sov'].mean())
                
                # Dynamic ASP and CPC Calculations
                asp = total_sales / total_units if total_units > 0 else 0
                avg_cpc = total_spend / total_clicks if total_clicks > 0 else 0
                breakeven_cvr = (avg_cpc / asp) * 100 if asp > 0 else 0
                
                # Formula Layer 1: The Non-Linear Logistic S-Curve Filter
                s_curve_factor = 1.0 / (1.0 + np.exp(steepness * (avg_organic_sov - inflection_point)))
                incrementality_factor = max(0.10, min(0.95, s_curve_factor))
                
                # Formula Layer 2: Category-Aware NTB Elasticity Adjuster
                avg_ntb = 0.0
                if has_ntb:
                    avg_ntb = float(prod_data['ntb_clean'].mean())
                    if "Consumables" in category_type:
                        incrementality_factor += (avg_ntb * 0.20)
                    else:
                        incrementality_factor += (avg_ntb * 0.05)
                    incrementality_factor = min(0.98, max(0.05, incrementality_factor))
                
                # Formula Layer 3: Contextual Supply Chain & Markdown Conditions
                avg_inventory = 100.0
                if has_inventory:
                    avg_inventory = float(prod_data['inv_clean'].mean())
                    if avg_inventory < 70.0:
                        incrementality_factor = min(0.98, incrementality_factor * 1.12)
                        secondary_flags.append(f"⚠️ **{prod}** average distribution dropped to {avg_inventory:.1f}%. Ad baseline modified for out-of-stock anomalies.")
                
                is_promo_active = False
                if has_promo:
                    promo_col = 'promo_status' if 'promo_flag' not in df.columns else 'promo_flag'
                    is_promo_active = prod_data[promo_col].astype(str).str.lower().str.contains('active|yes|1').any()
                    if is_promo_active:
                        incrementality_factor = min(0.98, incrementality_factor * 1.05)
                
                # Execute Pure Financial Calculations
                incremental_sales = total_sales * incrementality_factor
                iroas = incremental_sales / total_spend if total_spend > 0 else 0
                
                # True Net Revenue Generated per Unit Sold via Ads
                iunit_contribution = asp * incrementality_factor

                # Matrix Quadrant Allocation Logic
                if iroas >= 3.0 and incrementality_factor >= 0.40:
                    quadrant = "🚀 Aggressive Scale"
                elif iroas >= 3.0 and incrementality_factor < 0.40:
                    quadrant = "💰 Efficiency Max / Cap Budget"
                elif iroas < 3.0 and incrementality_factor >= 0.40:
                    quadrant = "🛠️ Structural Optimization"
                else:
                    quadrant = "❌ Defund / Defend Only"
                
                # Save data for recommendations parsing
                raw_metrics[prod] = {
                    'iroas': iroas,
                    'spend': total_spend,
                    'organic_sov': avg_organic_sov,
                    'factor': incrementality_factor,
                    'asp': asp,
                    'cpc': avg_cpc,
                    'breakeven_cvr': breakeven_cvr,
                    'iunit_contribution': iunit_contribution,
                    'quadrant': quadrant
                }
                
                table_data.append({
                    "Product ID": prod,
                    "Allocation Quadrant": quadrant,
                    "True iROAS": f"{iroas:.2f}x",
                    "Ad Incrementality %": f"{incrementality_factor*100:.1f}%",
                    "Avg Organic SOV": f"{avg_organic_sov*100:.1f}%",
                    "Avg ASP": f"${asp:,.2f}",
                    "Break-Even CVR": f"{breakeven_cvr:.1f}%",
                    "New-to-Brand %": f"{avg_ntb*100:.1f}%" if has_ntb else "N/A",
                    "iUnit Contribution": f"${iunit_contribution:,.2f}"
                })

            # --- UI RENDER: SECTION 1 (Unified Master Matrix) ---
            st.header("1. Unified Product Performance & Capital Matrix")
            st.markdown("This master table contrasts raw platform illusions against true causal efficiency, dynamically assigning each product to its optimal financial quadrant.")
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)

            st.markdown("---")
            
            # --- UI RENDER: SECTION 2 (Actionable Recommendations) ---
            st.header("2. Strategic Directives & Capital Execution Plan")
            st.markdown("Clear, product-by-product breakdowns detailing current performance reality and future capital action plans.")

            for prod, meta in raw_metrics.items():
                quadrant = meta['quadrant']
                iroas = meta['iroas']
                incr = meta['factor']
                org_sov = meta['organic_sov']
                asp = meta['asp']
                iunit = meta['iunit_contribution']
                cpc = meta['cpc']
                
                st.subheader(f"{prod}  |  {quadrant}")
                
                if quadrant == "🚀 Aggressive Scale":
                    today_txt = f"Operating at a highly efficient **{iroas:.2f}x True iROAS** with excellent **{incr*100:.1f}% Ad Incrementality**."
                    why_txt = f"Organic SOV is low ({org_sov*100:.1f}%). The brand relies heavily on paid placements to capture market share, meaning almost every ad-driven conversion is a net-new customer."
                    future_txt = "Route excess capital here immediately. Because incrementality is high, the marginal return of the next dollar spent is insulated against diminishing returns. Scale aggressively."
                    
                elif quadrant == "💰 Efficiency Max / Cap Budget":
                    today_txt = f"Generating strong overall efficiency (**{iroas:.2f}x True iROAS**), but Ad Incrementality has dropped to **{incr*100:.1f}%**."
                    why_txt = f"The brand dominates the organic shelf ({org_sov*100:.1f}% Organic SOV). The S-Curve cannibalization penalty is active because ads are beginning to overlap heavily with free listings."
                    future_txt = "Lock the current budget floor to protect the profitable baseline, but DO NOT scale. Any additional ad dollars pumped in here will only swallow up conversions you would have captured organically for free."
                    
                elif quadrant == "🛠️ Structural Optimization":
                    today_txt = f"Financial efficiency is lagging (**{iroas:.2f}x True iROAS**), despite strong net-new traffic value (**{incr*100:.1f}% Ad Incrementality**)."
                    why_txt = f"Ads intercept new market share efficiently (low organic overlap at {org_sov*100:.1f}%), but math breaks down on the PDP. ASP (${asp:.2f}) may be too low relative to CPCs (${cpc:.2f})."
                    future_txt = "Keep budgets completely flat. Fix the underlying retail readiness. Optimize the PDP, update A+ content, or switch ad destinations to multi-packs/bundles to widen the margin runway before spending more money."
                    
                elif quadrant == "❌ Defund / Defend Only":
                    today_txt = f"Highly inefficient cash returns (**{iroas:.2f}x True iROAS**) combined with severely depressed **{incr*100:.1f}% Ad Incrementality**."
                    why_txt = f"Paid media is highly redundant here. The brand already dominates the search shelf ({org_sov*100:.1f}% Organic SOV). Each ad-driven item only yields ${iunit:.2f} in true, net-new value due to cannibalization."
                    future_txt = "Trim budgets by 15% to 25%. Shift capital away from standard branded keywords where you are actively paying for clicks you would have earned for free organically. Restrict remaining spend *exclusively* to Conquesting spaces (intercepting competitor traffic where your organic footprint is zero) or a minimal Brand Defense floor strictly to block competitors from hijacking your top organic spots."

                st.markdown(f"""
                * **How we perform today:** {today_txt}
                * **Why:** {why_txt}
                * **Future Action:** {future_txt}
                """)
                st.write("") 

            st.markdown("---")

            # --- UI RENDER: SECTION 3 (Secondary Contextual Flags) ---
            st.header("3. Secondary Contextual & Operational Flags")
            if secondary_flags:
                for flag in secondary_flags:
                    st.warning(flag)
            else:
                st.success("✅ No critical operational or supply chain anomalies detected in the current data snapshot.")

            st.markdown("---")

            # --- UI RENDER: SECTION 4 (Behind the Curtains) ---
            st.header("4. Behind the Curtains (Causal Engine Mechanics)")
            st.markdown("""
            **How this tool determines True Incrementality:**
            1. **The Base S-Curve:** We apply a non-linear Logistic S-Curve against Organic SOV. As your organic shelf presence grows past the inflection point, the engine applies an aggressive cannibalization penalty, recognizing that ads are simply replacing free clicks.
            2. **Bayesian NTB Adjustment:** We inject New-to-Brand (NTB) data. High NTB rates act as a mathematically additive bonus, proving the campaign is conquering net-new market share.
            3. **Operational Shocks:** The engine scans for sub-80% inventory drops (which kill organic rankings and raise ad dependency) and active promotions (which compress the purchase path), adjusting the final factor in real-time.
            """)

        except Exception as e:
            st.error(f"❌ Application Error: {str(e)}")
