import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="A/B Test Calculator",
    page_icon="📊",
    layout="wide"
)

st.title("📊 A/B Test Calculator")
st.markdown("Statistical significance testing, "
            "sample size planning and result visualization.")
st.markdown("---")

def two_proportion_z_test(n_a, conv_a,
                           n_b, conv_b):
    p_a = conv_a / n_a
    p_b = conv_b / n_b
    p_pool = (conv_a + conv_b) / (n_a + n_b)
    se = np.sqrt(
        p_pool * (1 - p_pool) *
        (1/n_a + 1/n_b))
    if se == 0:
        return 0, 1
    z = (p_b - p_a) / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    return round(z, 4), round(p_val, 6)

def confidence_interval(n, conversions,
                         confidence=0.95):
    p = conversions / n
    z = stats.norm.ppf(
        1 - (1 - confidence) / 2)
    margin = z * np.sqrt(p * (1 - p) / n)
    return round(p - margin, 4), \
           round(p + margin, 4)

def sample_size_required(baseline_rate,
                          min_effect,
                          alpha=0.05,
                          power=0.8):
    p1 = baseline_rate
    p2 = baseline_rate + min_effect
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta  = stats.norm.ppf(power)
    p_avg   = (p1 + p2) / 2
    n = ((z_alpha * np.sqrt(
            2 * p_avg * (1 - p_avg)) +
          z_beta * np.sqrt(
            p1 * (1 - p1) +
            p2 * (1 - p2))) ** 2) / \
        (p2 - p1) ** 2
    return int(np.ceil(n))

def relative_uplift(p_a, p_b):
    if p_a == 0:
        return 0
    return round((p_b - p_a) / p_a * 100, 2)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧪 Run A/B Test",
    "📐 Sample Size Planner",
    "📈 Multi-Variant",
    "📚 Learn Statistics",
    "🗂️ Test History"
])

# Tab 1 — Run Test
with tab1:
    st.markdown("### 🧪 A/B Test Significance Calculator")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            "#### 🅰️ Control Group (A)")
        n_a = st.number_input(
            "Visitors (A):",
            min_value=1,
            value=10000,
            step=100,
            key="n_a")
        conv_a = st.number_input(
            "Conversions (A):",
            min_value=0,
            value=500,
            step=10,
            key="conv_a")
        if n_a > 0:
            rate_a = conv_a / n_a
            st.metric(
                "Conversion Rate A",
                f"{rate_a:.2%}")
            ci_a_low, ci_a_high = \
                confidence_interval(n_a, conv_a)
            st.caption(
                "95% CI: " +
                str(round(ci_a_low * 100, 2)) +
                "% — " +
                str(round(ci_a_high * 100, 2)) +
                "%")

    with col2:
        st.markdown(
            "#### 🅱️ Variant Group (B)")
        n_b = st.number_input(
            "Visitors (B):",
            min_value=1,
            value=10000,
            step=100,
            key="n_b")
        conv_b = st.number_input(
            "Conversions (B):",
            min_value=0,
            value=550,
            step=10,
            key="conv_b")
        if n_b > 0:
            rate_b = conv_b / n_b
            st.metric(
                "Conversion Rate B",
                f"{rate_b:.2%}",
                delta=f"{relative_uplift(rate_a, rate_b):+.2f}%")
            ci_b_low, ci_b_high = \
                confidence_interval(n_b, conv_b)
            st.caption(
                "95% CI: " +
                str(round(ci_b_low * 100, 2)) +
                "% — " +
                str(round(ci_b_high * 100, 2)) +
                "%")

    confidence = st.select_slider(
        "Confidence Level:",
        options=[0.90, 0.95, 0.99],
        value=0.95,
        format_func=lambda x:
            str(int(x * 100)) + "%")

    if st.button("🔬 Run Statistical Test",
                 type="primary",
                 use_container_width=True):

        alpha = 1 - confidence
        z_score, p_value = two_proportion_z_test(
            n_a, conv_a, n_b, conv_b)

        uplift = relative_uplift(rate_a, rate_b)
        abs_diff = round(
            (rate_b - rate_a) * 100, 4)

        st.markdown("---")
        st.markdown("### 📊 Test Results")

        if p_value < alpha:
            st.success(
                "✅ **STATISTICALLY SIGNIFICANT** — "
                "The difference is real, not random!")
            verdict = "Significant"
        else:
            st.warning(
                "⚠️ **NOT SIGNIFICANT** — "
                "Need more data or the effect "
                "may not be real.")
            verdict = "Not Significant"

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Z-Score", z_score)
        c2.metric("P-Value", p_value)
        c3.metric("Confidence",
                  str(round(
                      (1 - p_value) * 100, 1)) +
                  "%")
        c4.metric("Relative Uplift",
                  str(uplift) + "%")
        c5.metric("Absolute Diff",
                  str(abs_diff) + "%")

        # Visualization
        col_v1, col_v2 = st.columns(2)

        with col_v1:
            # Conversion rate comparison
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Control (A)',
                x=['Control A', 'Variant B'],
                y=[rate_a * 100, rate_b * 100],
                marker_color=[
                    '#3498db', '#2ecc71'
                    if rate_b > rate_a
                    else '#e74c3c'],
                error_y=dict(
                    type='data',
                    array=[
                        (ci_a_high - ci_a_low)
                        * 100 / 2,
                        (ci_b_high - ci_b_low)
                        * 100 / 2
                    ],
                    visible=True
                )
            ))
            fig.update_layout(
                title='Conversion Rates with '
                      '95% Confidence Intervals',
                yaxis_title='Conversion Rate (%)',
                height=350,
                template='plotly_white',
                showlegend=False
            )
            st.plotly_chart(
                fig, use_container_width=True)

        with col_v2:
            # Normal distribution visualization
            x = np.linspace(-4, 4, 300)
            y = stats.norm.pdf(x)

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines',
                line=dict(
                    color='#3498db', width=2),
                name='Null Distribution'
            ))

            # Rejection regions
            z_crit = stats.norm.ppf(
                1 - alpha / 2)
            x_reject_r = x[x >= z_crit]
            x_reject_l = x[x <= -z_crit]

            fig2.add_trace(go.Scatter(
                x=np.concatenate(
                    [x_reject_r,
                     x_reject_r[::-1]]),
                y=np.concatenate(
                    [stats.norm.pdf(x_reject_r),
                     np.zeros(len(x_reject_r))]),
                fill='toself',
                fillcolor='rgba(231,76,60,0.3)',
                line=dict(color='rgba(0,0,0,0)'),
                name='Rejection Region'
            ))
            fig2.add_trace(go.Scatter(
                x=np.concatenate(
                    [x_reject_l,
                     x_reject_l[::-1]]),
                y=np.concatenate(
                    [stats.norm.pdf(x_reject_l),
                     np.zeros(len(x_reject_l))]),
                fill='toself',
                fillcolor='rgba(231,76,60,0.3)',
                line=dict(color='rgba(0,0,0,0)'),
                showlegend=False
            ))

            # Z-score line
            fig2.add_vline(
                x=z_score,
                line_color='#f39c12',
                line_width=3,
                annotation_text=
                    "Z=" + str(z_score))

            fig2.update_layout(
                title='Z-Test Distribution',
                xaxis_title='Z-Score',
                yaxis_title='Probability Density',
                height=350,
                template='plotly_white'
            )
            st.plotly_chart(
                fig2, use_container_width=True)

        # Recommendation
        st.markdown("#### 💡 Recommendation")
        if verdict == "Significant" and \
                uplift > 0:
            st.success(
                "🚀 **Ship Variant B!** " +
                "It outperforms control by " +
                str(uplift) + "% " +
                "with statistical confidence.")
        elif verdict == "Significant" and \
                uplift < 0:
            st.error(
                "🛑 **Keep Control A!** " +
                "Variant B is worse by " +
                str(abs(uplift)) + "%.")
        else:
            needed = sample_size_required(
                rate_a, abs(rate_b - rate_a)
                if rate_b != rate_a else 0.01)
            st.warning(
                "⏳ **Keep testing.** " +
                "Suggested sample size: " +
                str(needed) +
                " per group for 80% power.")

        # Save to history
        if 'test_history' not in \
                st.session_state:
            st.session_state.test_history = []
        st.session_state.test_history.append({
            'Date': str(
                pd.Timestamp.now())[:16],
            'N_A': n_a,
            'N_B': n_b,
            'Rate_A': str(round(
                rate_a * 100, 2)) + "%",
            'Rate_B': str(round(
                rate_b * 100, 2)) + "%",
            'Uplift': str(uplift) + "%",
            'P-Value': p_value,
            'Result': verdict
        })

# Tab 2 — Sample Size
with tab2:
    st.markdown("### 📐 Sample Size Planner")
    st.markdown(
        "How many visitors do you need "
        "before starting a test?")

    col1, col2 = st.columns(2)

    with col1:
        baseline = st.number_input(
            "Current conversion rate (%):",
            min_value=0.1,
            max_value=99.0,
            value=5.0,
            step=0.1) / 100

        min_effect = st.number_input(
            "Minimum detectable effect (%):",
            min_value=0.1,
            max_value=50.0,
            value=2.0,
            step=0.1) / 100

        alpha_ss = st.selectbox(
            "Significance level (α):",
            [0.01, 0.05, 0.10],
            index=1,
            format_func=lambda x: str(x))

        power_ss = st.selectbox(
            "Statistical power:",
            [0.70, 0.80, 0.90],
            index=1,
            format_func=lambda x:
                str(int(x * 100)) + "%")

    with col2:
        n_required = sample_size_required(
            baseline, min_effect,
            alpha_ss, power_ss)

        st.markdown("#### 📊 Required Sample Size")
        st.markdown(
            "<h1 style='color:#3498db;"
            "text-align:center'>" +
            f"{n_required:,}" +
            "</h1>",
            unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align:center;"
            "color:gray'>per group</p>",
            unsafe_allow_html=True)

        total_n = n_required * 2
        st.metric("Total Visitors Needed",
                  f"{total_n:,}")
        st.metric(
            "Expected Variant Rate",
            f"{(baseline + min_effect):.2%}")
        st.metric("Absolute Effect",
                  f"{min_effect:.2%}")

    # Power curves
    st.markdown("---")
    st.markdown("#### 📈 Power Curve")

    effects = np.linspace(0.001, 0.1, 50)
    sample_sizes = [
        sample_size_required(
            baseline, e, alpha_ss, power_ss)
        for e in effects
    ]

    fig3 = px.line(
        x=[e * 100 for e in effects],
        y=sample_sizes,
        title='Required Sample Size vs '
              'Minimum Detectable Effect',
        labels={
            'x': 'Min Detectable Effect (%)',
            'y': 'Sample Size per Group'
        }
    )
    fig3.add_vline(
        x=min_effect * 100,
        line_dash="dash",
        line_color="#e74c3c",
        annotation_text="Your MDE")
    fig3.update_layout(
        height=350,
        template='plotly_white')
    st.plotly_chart(fig3,
                    use_container_width=True)

    # Traffic calculator
    st.markdown("#### 📅 Test Duration Estimator")
    daily_traffic = st.number_input(
        "Daily visitors to test page:",
        min_value=1,
        value=1000,
        step=100)
    traffic_split = st.slider(
        "Traffic split to test (%):",
        10, 100, 50)

    daily_per_group = (
        daily_traffic *
        traffic_split / 100 / 2)
    if daily_per_group > 0:
        days_needed = int(
            np.ceil(n_required / daily_per_group))
        st.metric(
            "Estimated Test Duration",
            str(days_needed) + " days")
        st.caption(
            str(int(daily_per_group)) +
            " visitors/day per group at " +
            str(traffic_split) +
            "% traffic split")

# Tab 3 — Multi-Variant
with tab3:
    st.markdown(
        "### 📈 Multi-Variant Test (A/B/C/D)")

    n_variants = st.slider(
        "Number of variants:", 2, 4, 3)

    variants = []
    cols = st.columns(n_variants)
    labels = ['A (Control)', 'B', 'C', 'D']
    colors = ['#3498db', '#2ecc71',
              '#f39c12', '#9b59b6']

    for i, col in enumerate(
        cols[:n_variants]
    ):
        with col:
            st.markdown(
                "#### " + labels[i])
            n_v = col.number_input(
                "Visitors:",
                min_value=1,
                value=5000,
                step=100,
                key="mv_n_" + str(i))
            c_v = col.number_input(
                "Conversions:",
                min_value=0,
                value=200 + i * 25,
                step=10,
                key="mv_c_" + str(i))
            rate_v = c_v / n_v
            col.metric(
                "Rate",
                f"{rate_v:.2%}")
            variants.append({
                'label': labels[i],
                'n': n_v,
                'conv': c_v,
                'rate': rate_v,
                'color': colors[i]
            })

    if st.button("📊 Analyze All Variants",
                 type="primary"):

        # Bar chart comparison
        fig4 = go.Figure()
        for v in variants:
            ci_l, ci_h = confidence_interval(
                v['n'], v['conv'])
            fig4.add_trace(go.Bar(
                name=v['label'],
                x=[v['label']],
                y=[v['rate'] * 100],
                marker_color=v['color'],
                error_y=dict(
                    type='data',
                    array=[
                        (ci_h - ci_l) * 100 / 2],
                    visible=True)
            ))

        fig4.update_layout(
            title='Conversion Rates — '
                  'All Variants',
            yaxis_title='Conversion Rate (%)',
            height=400,
            template='plotly_white',
            showlegend=True
        )
        st.plotly_chart(
            fig4, use_container_width=True)

        # Pairwise significance
        st.markdown("#### 🔬 Pairwise Significance")
        control = variants[0]
        results_rows = []
        for v in variants[1:]:
            z, p = two_proportion_z_test(
                control['n'], control['conv'],
                v['n'], v['conv'])
            uplift = relative_uplift(
                control['rate'], v['rate'])
            sig = "✅ Significant" \
                if p < 0.05 else "❌ Not Significant"
            results_rows.append({
                'Comparison': control['label'] +
                              ' vs ' + v['label'],
                'Z-Score': z,
                'P-Value': p,
                'Uplift': str(uplift) + "%",
                'Result': sig
            })

        res_df = pd.DataFrame(results_rows)
        st.dataframe(
            res_df,
            use_container_width=True,
            hide_index=True)

        # Winner
        best = max(variants,
                   key=lambda x: x['rate'])
        st.success(
            "🏆 Best performing variant: **" +
            best['label'] + "** with " +
            f"{best['rate']:.2%}" +
            " conversion rate")

# Tab 4 — Learn
with tab4:
    st.markdown("### 📚 A/B Testing Concepts")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### 🎯 Key Concepts

        **P-Value**
        Probability of seeing this result by
        chance if there was no real difference.
        Lower p = more confident the result is real.
        Rule: p < 0.05 = significant (95% confidence)

        **Z-Score**
        How many standard deviations your result
        is from the null hypothesis (no difference).
        |Z| > 1.96 = significant at 95% level.

        **Statistical Power**
        Probability of detecting a real effect
        when one exists. Aim for 80% power minimum.

        **Type I Error (False Positive)**
        Concluding B is better when it isn't.
        Controlled by significance level α.

        **Type II Error (False Negative)**
        Missing a real improvement.
        Controlled by statistical power.
        """)

    with col2:
        st.markdown("""
        #### ⚠️ Common Mistakes

        **Peeking at results early**
        Stopping as soon as you see significance
        inflates false positive rate dramatically.

        **Too many simultaneous tests**
        Running 20 tests simultaneously means
        ~1 will appear significant by chance alone.

        **Ignoring practical significance**
        A 0.001% lift may be statistically
        significant but not worth shipping.

        **Small sample sizes**
        Underpowered tests miss real effects
        and produce unreliable results.

        **Testing too many metrics**
        Define your primary metric BEFORE
        the test starts, not after.

        #### ✅ Best Practices
        - Define hypothesis before starting
        - Calculate sample size in advance
        - Run test for full business cycles
        - Don't stop early
        - Use 95%+ confidence level
        """)

    # Quick reference table
    st.markdown("---")
    st.markdown("#### 📋 Quick Reference")
    ref_df = pd.DataFrame({
        'Confidence Level': [
            '90%', '95%', '99%'],
        'Alpha (α)': [0.10, 0.05, 0.01],
        'Z-Critical': [1.645, 1.960, 2.576],
        'Use Case': [
            'Low-stakes, early exploration',
            'Standard product decisions',
            'High-stakes, irreversible changes'
        ]
    })
    st.dataframe(ref_df,
                 use_container_width=True,
                 hide_index=True)

    # Interactive p-value explainer
    st.markdown(
        "#### 🎲 P-Value Simulator")
    st.markdown(
        "See how p-value behaves "
        "with different sample sizes.")

    true_effect = st.slider(
        "True effect size (%):", 0, 20, 5)
    base_r = 0.05
    sample_sizes_sim = [
        100, 500, 1000, 5000, 10000]
    p_values_sim = []

    for n in sample_sizes_sim:
        conv_a_sim = int(n * base_r)
        conv_b_sim = int(
            n * (base_r + true_effect / 100))
        if conv_a_sim > 0 and \
                conv_b_sim > 0 and \
                conv_b_sim <= n:
            _, p = two_proportion_z_test(
                n, conv_a_sim,
                n, conv_b_sim)
            p_values_sim.append(p)
        else:
            p_values_sim.append(1.0)

    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=sample_sizes_sim,
        y=p_values_sim,
        mode='lines+markers',
        line=dict(color='#3498db', width=2),
        marker=dict(size=10)
    ))
    fig5.add_hline(
        y=0.05,
        line_dash="dash",
        line_color="#e74c3c",
        annotation_text="p=0.05 threshold")
    fig5.update_layout(
        title='P-Value vs Sample Size '
              '(true effect = ' +
              str(true_effect) + '%)',
        xaxis_title='Sample Size',
        yaxis_title='P-Value',
        height=350,
        template='plotly_white',
        yaxis_range=[0, 1]
    )
    st.plotly_chart(fig5,
                    use_container_width=True)

# Tab 5 — History
with tab5:
    st.markdown("### 🗂️ Test History")

    if 'test_history' not in \
            st.session_state or \
            not st.session_state.test_history:
        st.info(
            "Run a test in the first tab "
            "to see history here!")
    else:
        history_df = pd.DataFrame(
            st.session_state.test_history)
        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True)

        csv = history_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download History CSV",
            csv,
            "ab_test_history.csv",
            "text/csv")

        if st.button("🗑️ Clear History"):
            st.session_state.test_history = []
            st.rerun()

        sig_count = sum(
            1 for r in
            st.session_state.test_history
            if r['Result'] == 'Significant')
        total_tests = len(
            st.session_state.test_history)

        c1, c2, c3 = st.columns(3)
        c1.metric("Tests Run", total_tests)
        c2.metric("Significant",
                  sig_count)
        c3.metric("Win Rate",
                  str(round(
                      sig_count /
                      max(total_tests, 1)
                      * 100)) + "%")

st.markdown("---")
st.markdown(
    "Built by **Jyotiraditya** | "
    "A/B Test Calculator | "
    "Z-test · Sample Size · Multi-Variant"
)