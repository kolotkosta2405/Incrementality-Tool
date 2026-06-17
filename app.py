import streamlit as st
import pandas as pd
import numpy as np

# --- PAGE SETUP ---
st.set_page_config(page_title="Retail Media Incrementality Engine", layout="wide")
st.title("Retail Media Incrementality & Causal Optimization Engine")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("1. Upload Master Data Layer")
uploaded_file = st.sidebar.file_uploader("Upload Unified Performance CSV", type=["csv"])

# Advanced S-Curve tuning parameter toggles hidden cleanly in sidebar expander
with st.sidebar.expander("⚙️ Advanced S-Curve Coefficients"):
    inflection_point = st.slider("Curve Inflection Point (x₀)", 0.05, 0.35, 0.20, 0.05, 
                                 help="The Organic SOV point where incrementality degradation accelerates fastest.")
    steepness = st.slider("Curve Decay Steepness (k)", 5.0, 15.0, 10.0, 0.5,
                          help="Higher numbers enforce harsher cannibalization penalties when crossing the inflection threshold.")

# --- CORE LOGIC & PROCESSING ---
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Engine Data Storage
    processed_data = []
    secondary_flags = []
    
    for index, row in df.iterrows():
        product = row['product id']
        spend = row['media_spend']
        sales = row['total_sales']
        units = row['units_sold']
        clicks = row['clicks']
        org_sov = row['organic_sov']
        ntb = row['ntb_sales_pct']
        inventory = row['inventory_status']
        promo = str(row['promo_status']).lower()
        
        # 1. Base Platform Metrics
        roas = sales / spend if spend > 0 else 0
        asp = sales / units if units > 0 else 0
        cpc = spend / clicks if clicks > 0 else 0
        
        # 2. Mathematical S-Curve (Cannibalization Penalty)
        # Using the custom sidebar inputs
        s_curve_penalty = 1 / (1 + np.exp(steepness * (org_sov - inflection_point)))
        base_incr = max(0.10, min(0.95, s_curve_penalty))
        
        # 3. Bayesian NTB Adjustment
        cpg_weight = 0.20
        incr_lift = base_incr + (cpg_weight * ntb)
        
        # 4. Contextual Multipliers
        if inventory < 80.0:
            incr_lift *= 1.12
            secondary_flags.append(f"⚠️ **{product}**: Inventory dropped to {inventory}%. Organic listings are likely suppressed; ad dependency is temporarily elevated.")
        if promo in ['yes', 'true', '1']:
            incr_lift *= 1.05
            
        # 5. Final Ad Incrementality & iROAS Calculation
        final_incrementality = min(0.98, max(0.05, incr_lift))
        iroas = roas * final_incrementality
        i_unit_contribution = asp * final_incrementality
        
        # Unit Economics Check
        breakeven_cvr = (cpc / i_unit_contribution) if i_unit_contribution > 0 else 1.0
        if breakeven_cvr > 0.50:
            secondary_flags.append(f"🚨 **{product}**: Break-Even CVR is abnormally high ({breakeven_cvr*100:.1f}%). ASP (${asp:.2f}) may be too low to support current CPCs (${cpc:.2f}). Consider multi-packs.")

        # 6. Matrix Quadrant Allocation
        # Definitions: High iROAS >= 3.0, High Incrementality >= 0.40 (40%)
        if iroas >= 3.0 and final_incrementality >= 0.40:
            quadrant = "🚀 Aggressive Scale"
        elif iroas >= 3.0 and final_incrementality < 0.40:
            quadrant = "💰 Efficiency Max / Cap Budget"
        elif iroas < 3.0 and final_incrementality >= 0.40:
            quadrant = "🛠️ Structural Optimization"
        else:
            quadrant = "❌ Defund / Defend Only"

        # Append to processed data list
        processed_data.append({
            "Product": product,
            "Media Spend": f"${spend:,.0f}",
            "Platform ROAS": f"{roas:.2f}x",
            "True iROAS": round(iroas, 2),
            "Ad Incrementality %": round(final_incrementality, 3),
            "Organic SOV %": f"{org_sov*100:.1f}%",
            "Allocation Quadrant": quadrant,
            # Hidden operational data for the text loop later
            "_asp": asp,
            "_iunit": i_unit_contribution
        })

    results_df = pd.DataFrame(processed_data)
    
    # --- UI RENDER: SECTION 1 (Unified Master Matrix) ---
    st.header("1. Unified Product Performance & Capital Matrix")
    st.markdown("This master table contrasts raw platform illusions against true causal efficiency, dynamically assigning each product to its optimal financial quadrant.")
    
    # Display columns for the dataframe (dropping hidden metrics)
    display_cols = ["Product", "Allocation Quadrant", "True iROAS", "Ad Incrementality %", "Platform ROAS", "Organic SOV %", "Media Spend"]
    st.dataframe(results_df[display_cols].style.format({
        "True iROAS": "{:.2f}x",
        "Ad Incrementality %": "{:.1%}"
    }), use_container_width=True)

    st.markdown("---")
    
    # --- UI RENDER: SECTION 2 (Product-by-Product Recommendations) ---
    st.header("2. Strategic Directives & Capital Execution Plan")
    st.markdown("Clear, product-by-product breakdowns detailing current performance reality and future capital action plans.")

    for index, row in results_df.iterrows():
        quadrant = row["Allocation Quadrant"]
        product = row["Product"]
        iroas = row["True iROAS"]
        incr = row["Ad Incrementality %"]
        org_sov = row["Organic SOV %"]
        asp = row["_asp"]
        iunit = row["_iunit"]
        
        st.subheader(f"{product}  |  {quadrant}")
        
        # Dynamic text based on quadrant categorization
        if quadrant == "🚀 Aggressive Scale":
            today_txt = f"Operating at a highly efficient **{iroas:.2f}x True iROAS** with excellent **{incr*100:.1f}% Ad Incrementality**."
            why_txt = f"Organic SOV is low ({org_sov}). The brand relies heavily on paid placements to capture market share, meaning almost every ad-driven conversion is a net-new customer."
            future_txt = "Route excess capital here immediately. Because incrementality is high, the marginal return of the next dollar spent is insulated against diminishing returns. Scale aggressively."
            
        elif quadrant == "💰 Efficiency Max / Cap Budget":
            today_txt = f"Generating strong overall efficiency (**{iroas:.2f}x True iROAS**), but Ad Incrementality has dropped to **{incr*100:.1f}%**."
            why_txt = f"The brand dominates the organic shelf ({org_sov} Organic SOV). The S-Curve cannibalization penalty is active because ads are beginning to overlap heavily with free listings."
            future_txt = "Lock the current budget floor to protect the profitable baseline, but DO NOT scale. Any additional ad dollars pumped in here will only swallow up conversions you would have captured organically for free."
            
        elif quadrant == "🛠️ Structural Optimization":
            today_txt = f"Financial efficiency is lagging (**{iroas:.2f}x True iROAS**), despite strong net-new traffic value (**{incr*100:.1f}% Ad Incrementality**)."
            why_txt = f"Ads are successfully intercepting new market share (low organic overlap at {org_sov}), but the math is breaking down on the product page. ASP (${asp:.2f}) may be too low relative to CPCs, or conversion rates are dropping."
            future_txt = "Keep budgets completely flat. Fix the underlying retail readiness. Optimize the PDP, update A+ content, or switch ad destinations to multi-packs/bundles to widen the margin runway before spending more money."
            
        elif quadrant == "❌ Defund / Defend Only":
            today_txt = f"Highly inefficient cash returns (**{iroas:.2f}x True iROAS**) combined with severely depressed **{incr*100:.1f}% Ad Incrementality**."
            why_txt = f"Paid media is highly redundant here. The brand already dominates the search shelf ({org_sov} Organic SOV). Each ad-driven item only yields ${iunit:.2f} in true, net-new value due to cannibalization."
            future_txt = "Trim budgets by 15% to 25%. Shift capital away from standard branded keywords where you are actively paying for clicks you would have earned for free organically. Restrict remaining spend *exclusively* to Conquesting spaces (intercepting competitor traffic where your organic footprint is zero) or a minimal Brand Defense floor strictly to block competitors from hijacking your top organic spots."

        # Render the 3-part format
        st.markdown(f"""
        * **How we perform today:** {today_txt}
        * **Why:** {why_txt}
        * **Future Action:** {future_txt}
        """)
        st.write("") # spacing

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

else:
    st.info("👈 Please upload the Unified Performance CSV in the sidebar to run the Incrementality Engine.")
