import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pulp

# ---- PAGE CONFIGURATION ----
st.set_page_config(
    page_title="Construction Bid Optimizer",
    page_icon="🏗️",
    layout="wide"
)

# ---- APP HEADER ----
st.title("🏗️ Construction Bid Optimizer")
st.subheader("MWBE Compliance & Cost Optimization Tool")
st.markdown("""
This tool identifies the optimal combination of subcontractor awards 
across 10 major trades — minimizing total buyout cost while meeting 
fixed MBE and WBE compliance requirements.
""")
st.divider()

# ---- PROJECT PARAMETERS ----
st.header("📋 Project Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    gsf = st.number_input(
        "Gross Square Footage (GSF)",
        min_value=10000,
        max_value=5000000,
        value=500000,
        step=10000
    )

with col2:
    mbe_goal = st.number_input(
        "MBE Goal (%)",
        min_value=0.0,
        max_value=50.0,
        value=20.0,
        step=0.5
    )

with col3:
    wbe_goal = st.number_input(
        "WBE Goal (%)",
        min_value=0.0,
        max_value=50.0,
        value=10.0,
        step=0.5
    )

# Calculate budget and targets
base_costs = {
    "Concrete": 63, "Masonry": 27, "Steel": 18, "Carpentry": 33,
    "Roofing": 10, "Plumbing": 27, "HVAC": 30, "Electrical": 34,
    "Elevators": 7, "Finishes": 24
}

total_budget = sum(base_costs.values()) * gsf
mbe_target = total_budget * (mbe_goal / 100)
wbe_target = total_budget * (wbe_goal / 100)

# Display budget summary
col_b1, col_b2, col_b3 = st.columns(3)

with col_b1:
    st.metric("Base Budget", f"${total_budget:,.0f}")

with col_b2:
    st.metric("MBE Target", f"${mbe_target:,.0f}", 
              f"{mbe_goal:.1f}% of budget",
              delta_color="off",
              delta_arrow="off")

with col_b3:
    st.metric("WBE Target", f"${wbe_target:,.0f}", 
              f"{wbe_goal:.1f}% of budget",
              delta_color="off",
              delta_arrow="off")

st.divider()

# ---- BID GENERATION ----
trades = [
    "Concrete", "Masonry", "Steel", "Carpentry", "Roofing",
    "Plumbing", "HVAC", "Electrical", "Elevators", "Finishes"
]

cert_types = [
    "Non-MWBE", "Non-MWBE", "Non-MWBE",
    "MBE", "WBE", "MWBE"
]

@st.cache_data
def generate_bids(gsf_value):
    np.random.seed(42)
    rows = []
    for trade in trades:
        base = base_costs[trade] * gsf_value
        for i, cert in enumerate(cert_types):
            if cert == "Non-MWBE":
                price = base * np.random.uniform(0.90, 1.10)
            elif cert == "MBE":
                price = base * np.random.uniform(0.90, 1.10) * np.random.uniform(1.08, 1.12)
            elif cert == "WBE":
                price = base * np.random.uniform(0.90, 1.10) * np.random.uniform(1.08, 1.12)
            else:
                price = base * np.random.uniform(0.90, 1.10) * np.random.uniform(1.10, 1.15)
            rows.append({
                "trade": trade,
                "bidder_id": f"{trade[:4].upper()}-{i+1}",
                "certification": cert,
                "bid_amount": round(price, 2)
            })
    return pd.DataFrame(rows)

df_bids = generate_bids(gsf)

# ---- PREFERRED SUBCONTRACTOR SELECTION ----
st.header("🏗️ Preferred Subcontractor Selections")
st.caption("Select preferred subcontractors to lock in — or leave as 'None' to let the optimizer decide.")

forced_awards = {}
cols = st.columns(2)

for i, trade in enumerate(trades):
    trade_bids = df_bids[df_bids["trade"] == trade]
    options = ["None (Let optimizer decide)"] + [
        f"{row['bidder_id']} ({row['certification']}) — ${row['bid_amount']:,.0f}"
        for _, row in trade_bids.iterrows()
    ]
    with cols[i % 2]:
        selected = st.selectbox(
            f"{trade}",
            options=options,
            key=f"dropdown_{trade}"
        )
        if selected != "None (Let optimizer decide)":
            bidder_id = selected.split(" ")[0]
            forced_awards[trade] = bidder_id

st.divider()

# ---- RUN OPTIMIZER BUTTON ----
if st.button("🚀 Run Optimizer", type="primary", use_container_width=True):
    
    with st.spinner("Running optimization..."):
        
        # ---- OPTIMIZATION FUNCTIONS ----
        def run_optimizer(df_bids, total_budget, mbe_target, wbe_target, forced_awards={}):
            prob = pulp.LpProblem("MWBE_Buyout_Optimizer", pulp.LpMinimize)
            
            x = {}
            for _, row in df_bids.iterrows():
                if row["certification"] != "MWBE":
                    x[row["bidder_id"], row["trade"]] = pulp.LpVariable(
                        f"x_{row['bidder_id']}_{row['trade']}".replace("-", "_"),
                        cat="Binary"
                    )
            
            x_mbe = {}
            x_wbe = {}
            for _, row in df_bids[df_bids["certification"] == "MWBE"].iterrows():
                x_mbe[row["bidder_id"], row["trade"]] = pulp.LpVariable(
                    f"xmbe_{row['bidder_id']}_{row['trade']}".replace("-", "_"),
                    cat="Binary"
                )
                x_wbe[row["bidder_id"], row["trade"]] = pulp.LpVariable(
                    f"xwbe_{row['bidder_id']}_{row['trade']}".replace("-", "_"),
                    cat="Binary"
                )
            
            prob += pulp.lpSum([
                row["bid_amount"] * x[row["bidder_id"], row["trade"]]
                for _, row in df_bids[df_bids["certification"] != "MWBE"].iterrows()
            ] + [
                row["bid_amount"] * x_mbe[row["bidder_id"], row["trade"]] +
                row["bid_amount"] * x_wbe[row["bidder_id"], row["trade"]]
                for _, row in df_bids[df_bids["certification"] == "MWBE"].iterrows()
            ])
            
            for trade in trades:
                trade_bids = df_bids[df_bids["trade"] == trade]
                prob += pulp.lpSum([
                    x[row["bidder_id"], row["trade"]]
                    for _, row in trade_bids[trade_bids["certification"] != "MWBE"].iterrows()
                ] + [
                    x_mbe[row["bidder_id"], row["trade"]] +
                    x_wbe[row["bidder_id"], row["trade"]]
                    for _, row in trade_bids[trade_bids["certification"] == "MWBE"].iterrows()
                ]) == 1, f"one_bidder_{trade}"
            
            for _, row in df_bids[df_bids["certification"] == "MWBE"].iterrows():
                prob += x_mbe[row["bidder_id"], row["trade"]] + \
                        x_wbe[row["bidder_id"], row["trade"]] <= 1, \
                        f"one_bucket_{row['bidder_id']}_{row['trade']}".replace("-", "_")
            
            prob += pulp.lpSum([
                row["bid_amount"] * x[row["bidder_id"], row["trade"]]
                for _, row in df_bids[df_bids["certification"] == "MBE"].iterrows()
            ] + [
                row["bid_amount"] * x_mbe[row["bidder_id"], row["trade"]]
                for _, row in df_bids[df_bids["certification"] == "MWBE"].iterrows()
            ]) >= mbe_target, "mbe_goal"
            
            prob += pulp.lpSum([
                row["bid_amount"] * x[row["bidder_id"], row["trade"]]
                for _, row in df_bids[df_bids["certification"] == "WBE"].iterrows()
            ] + [
                row["bid_amount"] * x_wbe[row["bidder_id"], row["trade"]]
                for _, row in df_bids[df_bids["certification"] == "MWBE"].iterrows()
            ]) >= wbe_target, "wbe_goal"
            
            for trade, bidder_id in forced_awards.items():
                cert = df_bids[(df_bids["trade"] == trade) &
                              (df_bids["bidder_id"] == bidder_id)]["certification"].values[0]
                if cert == "MWBE":
                    prob += x_mbe[bidder_id, trade] + \
                            x_wbe[bidder_id, trade] == 1, \
                            f"forced_{trade}"
                else:
                    prob += x[bidder_id, trade] == 1, f"forced_{trade}"
            
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            
            results = []
            for _, row in df_bids[df_bids["certification"] != "MWBE"].iterrows():
                var = x[row["bidder_id"], row["trade"]]
                if pulp.value(var) == 1:
                    results.append({
                        "trade": row["trade"],
                        "bidder_id": row["bidder_id"],
                        "certification": row["certification"],
                        "bucket": row["certification"] if row["certification"] != "Non-MWBE" else "Non-MWBE",
                        "bid_amount": row["bid_amount"]
                    })
            
            for _, row in df_bids[df_bids["certification"] == "MWBE"].iterrows():
                mbe_val = pulp.value(x_mbe[row["bidder_id"], row["trade"]])
                wbe_val = pulp.value(x_wbe[row["bidder_id"], row["trade"]])
                if mbe_val == 1:
                    results.append({
                        "trade": row["trade"],
                        "bidder_id": row["bidder_id"],
                        "certification": row["certification"],
                        "bucket": "MBE",
                        "bid_amount": row["bid_amount"]
                    })
                elif wbe_val == 1:
                    results.append({
                        "trade": row["trade"],
                        "bidder_id": row["bidder_id"],
                        "certification": row["certification"],
                        "bucket": "WBE",
                        "bid_amount": row["bid_amount"]
                    })
            
            df_results = pd.DataFrame(results)
            df_results["trade"] = pd.Categorical(df_results["trade"],
                                                  categories=trades, ordered=True)
            df_results = df_results.sort_values("trade").reset_index(drop=True)
            total_cost = pulp.value(prob.objective)
            status = pulp.LpStatus[prob.status]
            return df_results, total_cost, status

        def run_greedy(df_bids, mbe_target, wbe_target):
            scenario = []
            for trade in trades:
                trade_bids = df_bids[df_bids["trade"] == trade]
                lowest = trade_bids.loc[trade_bids["bid_amount"].idxmin()]
                scenario.append({
                    "trade": lowest["trade"],
                    "bidder_id": lowest["bidder_id"],
                    "certification": lowest["certification"],
                    "bucket": "MBE" if lowest["certification"] == "MBE"
                              else "WBE" if lowest["certification"] == "WBE"
                              else "Non-MWBE",
                    "bid_amount": lowest["bid_amount"]
                })
            df_greedy = pd.DataFrame(scenario)
            
            mbe_spend = df_greedy[df_greedy["bucket"] == "MBE"]["bid_amount"].sum()
            if mbe_spend < mbe_target:
                for trade in trades:
                    if mbe_spend >= mbe_target:
                        break
                    current = df_greedy[df_greedy["trade"] == trade].iloc[0]
                    if current["bucket"] != "MBE":
                        trade_bids = df_bids[df_bids["trade"] == trade]
                        mbe_bids = trade_bids[trade_bids["certification"].isin(["MBE", "MWBE"])]
                        if not mbe_bids.empty:
                            cheapest_mbe = mbe_bids.loc[mbe_bids["bid_amount"].idxmin()]
                            df_greedy.loc[df_greedy["trade"] == trade,
                                          ["bidder_id", "certification", "bucket", "bid_amount"]] = [
                                cheapest_mbe["bidder_id"], cheapest_mbe["certification"],
                                "MBE", cheapest_mbe["bid_amount"]
                            ]
                            mbe_spend = df_greedy[df_greedy["bucket"] == "MBE"]["bid_amount"].sum()
            
            wbe_spend = df_greedy[df_greedy["bucket"] == "WBE"]["bid_amount"].sum()
            if wbe_spend < wbe_target:
                for trade in trades:
                    if wbe_spend >= wbe_target:
                        break
                    current = df_greedy[df_greedy["trade"] == trade].iloc[0]
                    if current["bucket"] != "WBE" and current["bucket"] != "MBE":
                        trade_bids = df_bids[df_bids["trade"] == trade]
                        wbe_bids = trade_bids[trade_bids["certification"].isin(["WBE", "MWBE"])]
                        if not wbe_bids.empty:
                            cheapest_wbe = wbe_bids.loc[wbe_bids["bid_amount"].idxmin()]
                            df_greedy.loc[df_greedy["trade"] == trade,
                                          ["bidder_id", "certification", "bucket", "bid_amount"]] = [
                                cheapest_wbe["bidder_id"], cheapest_wbe["certification"],
                                "WBE", cheapest_wbe["bid_amount"]
                            ]
                            wbe_spend = df_greedy[df_greedy["bucket"] == "WBE"]["bid_amount"].sum()
            
            df_greedy["trade"] = pd.Categorical(df_greedy["trade"],
                                                 categories=trades, ordered=True)
            df_greedy = df_greedy.sort_values("trade").reset_index(drop=True)
            greedy_total = df_greedy["bid_amount"].sum()
            greedy_mbe = df_greedy[df_greedy["bucket"] == "MBE"]["bid_amount"].sum()
            greedy_wbe = df_greedy[df_greedy["bucket"] == "WBE"]["bid_amount"].sum()
            return df_greedy, greedy_total, greedy_mbe, greedy_wbe

        def run_lowest_bid(df_bids):
            results = []
            for trade in trades:
                trade_bids = df_bids[df_bids["trade"] == trade]
                lowest = trade_bids.loc[trade_bids["bid_amount"].idxmin()]
                results.append({
                    "trade": lowest["trade"],
                    "bidder_id": lowest["bidder_id"],
                    "certification": lowest["certification"],
                    "bucket": "Non-MWBE",
                    "bid_amount": lowest["bid_amount"]
                })
            df_lowest = pd.DataFrame(results)
            lowest_total = df_lowest["bid_amount"].sum()
            return df_lowest, lowest_total

        # ---- RUN ALL THREE SCENARIOS ----
        df_optimal, optimal_cost, status = run_optimizer(
            df_bids, total_budget, mbe_target, wbe_target, forced_awards={})
        
        df_greedy, greedy_total, greedy_mbe, greedy_wbe = run_greedy(
            df_bids, mbe_target, wbe_target)
        
        df_lowest, lowest_total = run_lowest_bid(df_bids)
        
        if forced_awards:
            df_forced, forced_cost, forced_status = run_optimizer(
                df_bids, total_budget, mbe_target, wbe_target, forced_awards=forced_awards)
        else:
            df_forced = None
            forced_cost = None

        st.success(f"Optimization complete! Status: {status}")

        # ---- RESULTS ----
        st.header("📊 Results")
        
        # Calculate MWBE totals helper
        def get_mwbe_totals(df):
            mbe = df[df["bucket"] == "MBE"]["bid_amount"].sum()
            wbe = df[df["bucket"] == "WBE"]["bid_amount"].sum()
            return mbe, wbe
        
        optimal_mbe, optimal_wbe = get_mwbe_totals(df_optimal)
        greedy_mbe_compliant = greedy_mbe >= mbe_target
        greedy_wbe_compliant = greedy_wbe >= wbe_target
        optimal_mbe_compliant = optimal_mbe >= mbe_target
        optimal_wbe_compliant = optimal_wbe >= wbe_target

        # ---- SUMMARY COMPARISON TABLE ----
        st.subheader("Strategy Comparison")
        
        comparison_data = {
            "Strategy": ["Greedy Lowest Bid (non-compliant)", 
                        "Greedy MWBE", 
                        "Pure Optimal (PuLP)"],
            "Total Cost": [f"${lowest_total:,.0f}", 
                          f"${greedy_total:,.0f}", 
                          f"${optimal_cost:,.0f}"],
            "Savings vs Budget": [f"${total_budget - lowest_total:,.0f}",
                                  f"${total_budget - greedy_total:,.0f}",
                                  f"${total_budget - optimal_cost:,.0f}"],
            "MBE %": [f"0.0%",
                      f"{greedy_mbe/total_budget*100:.1f}%",
                      f"{optimal_mbe/total_budget*100:.1f}%"],
            "WBE %": [f"0.0%",
                      f"{greedy_wbe/total_budget*100:.1f}%",
                      f"{optimal_wbe/total_budget*100:.1f}%"],
            "Compliant": ["❌ No", "✅ Yes", "✅ Yes"]
        }
        
        if df_forced is not None:
            forced_mbe, forced_wbe = get_mwbe_totals(df_forced)
            forced_mbe_compliant = forced_mbe >= mbe_target
            forced_wbe_compliant = forced_wbe >= wbe_target
            comparison_data["Strategy"].append("Your Selection")
            comparison_data["Total Cost"].append(f"${forced_cost:,.0f}")
            comparison_data["Savings vs Budget"].append(f"${total_budget - forced_cost:,.0f}")
            comparison_data["MBE %"].append(f"{forced_mbe/total_budget*100:.1f}%")
            comparison_data["WBE %"].append(f"{forced_wbe/total_budget*100:.1f}%")
            comparison_data["Compliant"].append(
                "✅ Yes" if forced_mbe_compliant and forced_wbe_compliant else "❌ No")
        
        st.dataframe(comparison_data, use_container_width=True, hide_index=True)
        st.divider()

        # ---- VALUE OF OPTIMIZATION ----
        st.subheader("💰 Value of Optimization")
        
        val_col1, val_col2 = st.columns(2)
        
        with val_col1:
            st.metric(
                "Pure Optimal vs Greedy MWBE",
                f"${greedy_total - optimal_cost:,.0f}",
                "saved by using optimizer",
                delta_color="off",
                delta_arrow="off"
            )
        
        with val_col2:
            st.metric(
                "Cost of MWBE Compliance",
                f"${optimal_cost - lowest_total:,.0f}",
                "above non-compliant baseline",
                delta_color="off",
                delta_arrow="off"
            )
        
        if df_forced is not None:
            val_col3, val_col4 = st.columns(2)
            vs_greedy = greedy_total - forced_cost
            
            with val_col3:
                if vs_greedy >= 0:
                    st.metric(
                        "Your Selection vs Greedy MWBE",
                        f"${vs_greedy:,.0f}",
                        "saved vs greedy MWBE approach",
                        delta_color="off",
                        delta_arrow="off"
                    )
                else:
                    st.metric(
                        "Your Selection vs Greedy MWBE",
                        f"${abs(vs_greedy):,.0f}",
                        "MORE expensive than greedy MWBE ⚠️",
                        delta_color="off",
                        delta_arrow="off"
                    )
            
            with val_col4:
                st.metric(
                    "Cost of Your Preferences",
                    f"${forced_cost - optimal_cost:,.0f}",
                    "above pure optimal",
                    delta_color="off",
                    delta_arrow="off"
                )
        
        st.divider()

        # ---- DETAILED RESULTS ----
        st.subheader("📋 Detailed Award Recommendations")
        
        # Tabs for each scenario
        if df_forced is not None:
            tab1, tab2, tab3, tab4 = st.tabs([
                "🤖 Pure Optimal", 
                "⭐ Your Selection", 
                "📊 Greedy MWBE",
                "❌ Greedy Non-Compliant"
            ])
        else:
            tab1, tab3, tab4 = st.tabs([
                "🤖 Pure Optimal", 
                "📊 Greedy MWBE",
                "❌ Greedy Non-Compliant"
            ])
        
        with tab1:
            df_optimal_display = df_optimal.copy()
            df_optimal_display["selection"] = "🤖 Optimizer"
            df_optimal_display["bid_amount"] = df_optimal_display["bid_amount"].apply(
                lambda x: f"${x:,.0f}")
            st.caption(f"Total Cost: ${optimal_cost:,.0f} | "
                      f"MBE: {optimal_mbe/total_budget*100:.1f}% | "
                      f"WBE: {optimal_wbe/total_budget*100:.1f}%")
            st.dataframe(df_optimal_display, use_container_width=True, hide_index=True)
        
        if df_forced is not None:
            with tab2:
                df_forced_display = df_forced.copy()
                df_forced_display["selection"] = df_forced_display["trade"].apply(
                    lambda t: "⭐ Preferred" if t in forced_awards else "🤖 Optimizer"
                )
                df_forced_display["bid_amount"] = df_forced_display["bid_amount"].apply(
                    lambda x: f"${x:,.0f}")
                st.caption(f"Total Cost: ${forced_cost:,.0f} | "
                          f"MBE: {forced_mbe/total_budget*100:.1f}% | "
                          f"WBE: {forced_wbe/total_budget*100:.1f}%")
                st.dataframe(df_forced_display, use_container_width=True, hide_index=True)
        
        with tab3:
            df_greedy_display = df_greedy.copy()
            df_greedy_display["bid_amount"] = df_greedy_display["bid_amount"].apply(
                lambda x: f"${x:,.0f}")
            st.caption(f"Total Cost: ${greedy_total:,.0f} | "
                      f"MBE: {greedy_mbe/total_budget*100:.1f}% | "
                      f"WBE: {greedy_wbe/total_budget*100:.1f}%")
            st.dataframe(df_greedy_display, use_container_width=True, hide_index=True)
        
        with tab4:
            df_lowest_display = df_lowest.copy()
            df_lowest_display["bid_amount"] = df_lowest_display["bid_amount"].apply(
                lambda x: f"${x:,.0f}")
            st.caption(f"Total Cost: ${lowest_total:,.0f} | "
                      f"MBE: 0.0% | WBE: 0.0% | ❌ Non-Compliant")
            st.dataframe(df_lowest_display, use_container_width=True, hide_index=True)
        
        st.divider()

        # ---- PREFERENCE ANALYSIS WARNING ----
        if df_forced is not None:
            st.subheader("⚠️ Preference Analysis")
            
            vs_greedy = greedy_total - forced_cost
            cost_of_prefs = forced_cost - optimal_cost
            
            if vs_greedy < 0:
                st.error(f"""
                **⚠️ WARNING: Your preferred selections exceed the greedy strategy cost.**
                
                Your selection is **${abs(vs_greedy):,.0f} MORE expensive** than what a 
                human estimator would typically spend. Consider releasing some preferences 
                to improve cost efficiency.
                """)
            elif cost_of_prefs > optimal_cost * 0.01:
                st.warning(f"""
                **⚠️ CAUTION: Your preferred selections are more than 1% above pure optimal.**
                
                Your selection is **${cost_of_prefs:,.0f} above** the pure optimal solution. 
                Minor adjustments could improve cost efficiency.
                """)
            else:
                st.success(f"""
                **✅ GOOD: Your selections are within 1% of the pure optimal solution.**
                
                Great balance of cost efficiency and relationship management. 
                Your preferences cost only **${cost_of_prefs:,.0f} above** pure optimal.
                """)
            
            st.divider()
        
        # ---- VISUALIZATION ----
        st.subheader("📈 Strategy Comparison Charts")
        
        scenario_labels = ["Greedy\nLowest Bid", "Greedy\nMWBE", "Pure\nOptimal"]
        costs = [lowest_total, greedy_total, optimal_cost]
        mbe_pcts = [0.0, greedy_mbe/total_budget*100, optimal_mbe/total_budget*100]
        wbe_pcts = [0.0, greedy_wbe/total_budget*100, optimal_wbe/total_budget*100]
        colors = ["red", "orange", "green"]
        
        if df_forced is not None:
            scenario_labels.append("Your\nSelection")
            costs.append(forced_cost)
            mbe_pcts.append(forced_mbe/total_budget*100)
            wbe_pcts.append(forced_wbe/total_budget*100)
            colors.append("blue")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Chart 1 - Total cost
        bars = axes[0].bar(scenario_labels, costs, color=colors, alpha=0.7)
        axes[0].axhline(y=total_budget, color="black", linestyle="--", label="Base Budget")
        axes[0].set_title("Total Buyout Cost by Strategy")
        axes[0].set_ylabel("Total Cost")
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1e6:.0f}M"))
        axes[0].legend()
        for bar, cost in zip(bars, costs):
            axes[0].text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + total_budget * 0.003,
                        f"${cost/1e6:.1f}M", ha="center", va="bottom",
                        fontweight="bold", fontsize=12)
        
        # Chart 2 - MBE compliance
        bars2 = axes[1].bar(scenario_labels, mbe_pcts, color=colors, alpha=0.7)
        axes[1].axhline(y=mbe_goal, color="black", linestyle="--",
                        label=f"MBE Goal ({mbe_goal:.1f}%)")
        axes[1].set_title("MBE Spend % of Base Budget")
        axes[1].set_ylabel("MBE %")
        axes[1].legend()
        for bar, pct in zip(bars2, mbe_pcts):
            axes[1].text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.3,
                        f"{pct:.1f}%", ha="center", va="bottom",
                        fontweight="bold", fontsize=12)
        
        # Chart 3 - WBE compliance
        bars3 = axes[2].bar(scenario_labels, wbe_pcts, color=colors, alpha=0.7)
        axes[2].axhline(y=wbe_goal, color="black", linestyle="--",
                        label=f"WBE Goal ({wbe_goal:.1f}%)")
        axes[2].set_title("WBE Spend % of Base Budget")
        axes[2].set_ylabel("WBE %")
        axes[2].legend()
        for bar, pct in zip(bars3, wbe_pcts):
            axes[2].text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.3,
                        f"{pct:.1f}%", ha="center", va="bottom",
                        fontweight="bold", fontsize=12)
        
        plt.suptitle("MWBE Buyout Optimization — Strategy Comparison",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)