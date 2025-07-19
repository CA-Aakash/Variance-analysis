import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import openai
from datetime import datetime

# --- Page configuration & CSS ---
st.set_page_config(
    page_title="📊 Variance Analysis AI Copilot",
    layout="wide"
)
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .commentary-box {
        background: #f8f9fa;
        padding: 1rem;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True
)
st.markdown(
    '<div class="main-header"><h1>📊 Variance Analysis AI Copilot</h1>' +
    '<p>Transform your variance analysis with AI-powered insights</p></div>',
    unsafe_allow_html=True
)

# --- Full-template generator ---
def get_sample_template():
    scenarios = ["Budget", "Actual", "Forecast"]
    years     = [2024]
    months    = ["Jan", "Feb", "Mar"]
    products  = ["A", "B", "C", "D"]
    regions   = ["North America", "Europe", "Asia", "South America"]
    segments  = ["SMB", "Enterprise", "Consumer"]
    channels  = ["Online", "Distributor", "Retail"]
    rows = []
    np.random.seed(42)
    for sc in scenarios:
        for yr in years:
            for mo in months:
                for prod in products:
                    for reg in regions:
                        for seg in segments:
                            for ch in channels:
                                rows.append({
                                    "Scenario": sc,
                                    "Year": yr,
                                    "Month": mo,
                                    "Product": prod,
                                    "Region": reg,
                                    "Customer Segment": seg,
                                    "Channel": ch,
                                    "Units Sold": np.random.randint(500, 2000),
                                    "Price per Unit": round(np.random.uniform(20, 40), 2),
                                    "FX Rate": round(np.random.uniform(0.8, 1.2), 2),
                                    "COGS %": round(np.random.uniform(0.4, 0.7), 2),
                                    "Operating Expenses": round(np.random.uniform(15000, 30000), 2),
                                    "Depreciation": round(np.random.uniform(2000, 5000), 2),
                                    "Tax Rate": 0.25
                                })
    return pd.DataFrame(rows)

# --- Chart helpers ---
def plot_bar(df, x_col, measure):
    top = df.head(10)
    colors = ['green' if v > 0 else 'red' for v in top['Variance_Absolute']]
    fig = go.Figure(go.Bar(
        x=top[x_col].astype(str),
        y=top['Variance_Absolute'],
        marker_color=colors
    ))
    fig.update_layout(
        title=f"Top Variance Drivers ({measure})",
        yaxis_title=f"{measure} Variance",
        xaxis_title=x_col,
        template='plotly_white'
    )
    return fig

def plot_waterfall(df, a_col, b_col, x_col):
    data = [{'Category': a_col, 'Value': df[a_col].sum(), 'Type': 'absolute'}]
    for _, row in df.head(5).iterrows():
        data.append({
            'Category': row[x_col],
            'Value': row['Variance_Absolute'],
            'Type': 'relative'
        })
    data.append({'Category': b_col, 'Value': df[b_col].sum(), 'Type': 'absolute'})
    dff = pd.DataFrame(data)
    fig = go.Figure(go.Waterfall(
        orientation='v',
        measure=dff['Type'].tolist(),
        x=dff['Category'],
        y=dff['Value'],
        connector={'line': {'color': 'rgb(63,63,63)'}}
    ))
    fig.update_layout(
        title='Waterfall: Baseline to Comparison',
        template='plotly_white'
    )
    return fig

# New helpers

def plot_heatmap(df, x_dim, y_dim):
    pivot = df.pivot_table(
        index=y_dim,
        columns=x_dim,
        values='Variance_Percent',
        aggfunc='mean'
    )
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.astype(str),
        y=pivot.index.astype(str),
        colorscale='RdBu',
        reversescale=True,
        colorbar_title='% Var'
    ))
    fig.update_layout(title='Heatmap of Variance %', template='plotly_white')
    return fig

def plot_tornado(df, x_col):
    top = df.head(10).copy()
    top = top.sort_values('Variance_Absolute')
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=-top['Variance_Absolute'],
        y=top[x_col].astype(str),
        orientation='h',
        name='Unfavorable',
        marker_color='red'
    ))
    fig.add_trace(go.Bar(
        x=top['Variance_Absolute'],
        y=top[x_col].astype(str),
        orientation='h',
        name='Favorable',
        marker_color='green'
    ))
    fig.update_layout(title='Tornado Chart', barmode='overlay', template='plotly_white')
    return fig

def plot_trend(df, measure):
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'], format='%Y-%b')
    trend = df.groupby('Date')[measure].sum().reset_index()
    fig = px.line(trend, x='Date', y=measure, title=f'{measure} Trend', template='plotly_white')
    return fig

def plot_scatter(df, a_col, b_col):
    # Scatter of baseline vs comparison values
    fig = px.scatter(
        df,
        x=a_col,
        y=b_col,
        color='Variance_Absolute',
        title=f'{a_col} vs {b_col} Scatter',
        labels={a_col: a_col, b_col: b_col, 'Variance_Absolute': 'Variance Absolute'},
        template='plotly_white'
    )
    return fig

def plot_bullet(actual, budget):
    fig = go.Figure(go.Indicator(
        mode="number+gauge", value=actual,
        gauge={'shape': "bullet", 'axis': {'range': [0, max(budget, actual) * 1.2]}},
        delta={'reference': budget},
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    fig.update_layout(title='Bullet Chart: Actual vs Budget')
    return fig

# --- Sidebar configuration ---
st.sidebar.header("🔧 Configuration")
api_key = st.sidebar.text_input(
    "🔑 OpenAI API Key",
    type="password",
    help="Enter your OpenAI API key to enable AI commentary"
)
st.session_state.api_key = api_key

st.sidebar.subheader("📋 Sample Data Template")
if st.sidebar.button("📥 Download Full Template"):
    df_full = get_sample_template()
    buf = BytesIO()
    df_full.to_excel(buf, index=False)
    buf.seek(0)
    st.sidebar.download_button(
        "Download Full Template",
        data=buf,
        file_name="variance_full_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

uploaded = st.sidebar.file_uploader(
    "📁 Upload Excel File",
    type=["xlsx","xls"],
    help="Upload your variance data"
)

# --- State init ---
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'var_df' not in st.session_state:
    st.session_state.var_df = None
if 'commentary' not in st.session_state:
    st.session_state.commentary = ""

# --- VarianceAnalyzer class ---
class VarianceAnalyzer:
    def __init__(self):
        self.df = None
        self.variance_summary = None

    def load_data(self, file):
        df = pd.read_excel(file)
        df.columns = df.columns.str.strip()
        df['Revenue'] = df['Units Sold'] * df['Price per Unit'] * df.get('FX Rate', 1)
        df['COGS Amount'] = df['Revenue'] * df['COGS %']
        df['Gross Profit'] = df['Revenue'] - df['COGS Amount']
        df['Gross Margin %'] = df['Gross Profit'] / df['Revenue'] * 100
        df['EBITDA'] = df['Gross Profit'] - df['Operating Expenses'] + df.get('Depreciation', 0)
        df['Operating Profit'] = df['EBITDA'] - df.get('Depreciation', 0)
        df['Tax'] = df['Operating Profit'] * df.get('Tax Rate', 0)
        df['Net Income'] = df['Operating Profit'] - df['Tax']
        df['Cash Flow'] = df['Net Income'] + df.get('Depreciation', 0)
        df['Operating Margin %'] = df['Operating Profit'] / df['Revenue'] * 100
        df['Net Margin %'] = df['Net Income'] / df['Revenue'] * 100
        self.df = df
        return True, "Loaded"

    def pivot_comparison(self, measure, cmp, left, right, yA, yB, gb):
        df = self.df.copy()
        if cmp.startswith('Year'):
            df = df[df['Scenario'] == 'Actual']
            wide = df.pivot_table(
                index=gb or [],
                columns='Year',
                values=measure,
                aggfunc='sum'
            ).reset_index()
            for y in [yA, yB]:
                if y not in wide.columns:
                    wide[y] = 0
            a, b = yA, yB
        else:
            df = df[df['Scenario'].isin([left, right])]
            wide = df.pivot_table(
                index=gb or [],
                columns='Scenario',
                values=measure,
                aggfunc='sum'
            ).reset_index()
            for sc in [left, right]:
                if sc not in wide.columns:
                    wide[sc] = 0
            a, b = left, right
        wide.columns.name = None
        wide[a] = wide[a].fillna(0)
        wide[b] = wide[b].fillna(0)
        wide['Variance_Absolute'] = wide[b] - wide[a]
        wide['Variance_Percent'] = np.where(
            wide[a] != 0,
            wide['Variance_Absolute'] / wide[a] * 100,
            np.nan
        )
        tot = wide[a].sum() or 1
        wide['Impact_Percent'] = wide['Variance_Absolute'] / tot * 100
        self.variance_summary = wide.sort_values(
            'Variance_Absolute', key=lambda s: s.abs(), ascending=False
        )
        return a, b

# --- Load & preview ---
if uploaded:
    if st.session_state.analyzer is None:
        st.session_state.analyzer = VarianceAnalyzer()
    ok, msg = st.session_state.analyzer.load_data(uploaded)
    if not ok:
        st.error(msg)
    else:
        df = st.session_state.analyzer.df
        st.subheader("📊 Data Preview")
        st.dataframe(df.head().reset_index(drop=True), use_container_width=True)

# --- Analysis Configuration ---
if 'df' in locals():
    st.markdown("## ⚙️ Analysis Configuration")
    measures = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in ['Year','Month']
    ]
    measure = st.selectbox("Measure", measures)
    cmp = st.selectbox(
        "Comparison", [
            'Budget vs Actual', 'Forecast vs Actual',
            'Budget vs Forecast', 'Year-over-Year (Actual)'
        ]
    )
    if cmp.startswith('Year'):
        left = right = None
        years = sorted(df['Year'].unique())
        yA = st.selectbox('Year A', years)
        yB = st.selectbox('Year B', [y for y in years if y != yA])
    else:
        opts = list(df['Scenario'].unique())
        left = st.selectbox('Left Scenario', opts)
        right = st.selectbox(
            'Right Scenario', [o for o in opts if o != left]
        )
        yA = yB = None
    dims = [
        c for c in df.columns
        if c not in measures + ['Scenario','Year','Month']
    ]
    gb = st.multiselect('Group by', dims)
    # Mandatory group-by check
    if not gb:
        st.warning("👉 Please select at least one 'Group by' dimension (e.g. Region or Product) before analyzing.")
        st.stop()
    if st.button('🔍 Analyze'):
        a, b = st.session_state.analyzer.pivot_comparison(
            measure, cmp, left, right, yA, yB, gb
        )
        st.session_state.var_df = st.session_state.analyzer.variance_summary
        st.session_state.a_col = a
        st.session_state.b_col = b

# --- Results ---
if st.session_state.var_df is not None:
    vs = st.session_state.var_df
    a = st.session_state.a_col
    b = st.session_state.b_col
    xcol = gb[0]
    st.subheader("📈 Variance Summary")
    st.dataframe(vs.reset_index(drop=True), use_container_width=True)

    st.subheader("🎯 Key Metrics")
    tA = vs[a].sum()
    tB = vs[b].sum()
    d = tB - tA
    p = d / tA * 100 if tA else 0
    c1, c2, c3, c4 = st.columns(4)
    # Format metrics without dollar prefix for non-currency measures
    if measure.lower() == 'units sold':
        c1.metric(a, f"{tA:,.0f}")
        c2.metric(b, f"{tB:,.0f}")
        c3.metric('Diff', f"{d:,.0f}", f"{p:.1f}%")
    else:
        c1.metric(a, f"${tA:,.0f}")
        c2.metric(b, f"${tB:,.0f}")
        c3.metric('Diff', f"${d:,.0f}", f"{p:.1f}%")
    # Fav items always number
    c4.metric('Fav Items', str((vs['Variance_Absolute'] > 0).sum()))

    st.subheader("📊 Charts")
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(['Top Drivers','Waterfall','Heatmap','Tornado','Trend','Scatter','Bullet'])
    with tab1:
        fig = plot_bar(vs, xcol, measure)
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        fig = plot_waterfall(vs, a, b, xcol)
        st.plotly_chart(fig, use_container_width=True)
    with tab3:
        st.subheader("🗺️ Variance % Heatmap")
        # pick your two dims (we’ll just use the first two group‑bys here)
        x_dim = gb[0]
        y_dim = gb[1] if len(gb) > 1 else gb[0]
        fig = plot_heatmap(vs, x_dim, y_dim)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("🌪️ Tornado Chart")
        fig = plot_tornado(vs, xcol)
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.subheader("📈 Trend by Month")
        fig = plot_trend(st.session_state.analyzer.df, measure)
        st.plotly_chart(fig, use_container_width=True)

    with tab6:
        st.subheader("🔍 Variance vs Measure Scatter")
        fig = plot_scatter(vs, measure)
        st.plotly_chart(fig, use_container_width=True)

    with tab7:
        st.subheader("🎯 Bullet / Gauge")
        # show actual vs budget (or left vs right)
        actual = vs[b].sum()
        budget = vs[a].sum()
        fig = plot_bullet(actual, budget)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📝 AI Commentary")
    key = st.session_state.api_key
    if key and key.startswith('sk-'):
        if st.button('🤖 Comment'):
            cli = openai.OpenAI(api_key=key)
            lines = [
                f"- {r[xcol]}: {r['Variance_Absolute']:+,.0f} ({r['Variance_Percent']:+.1f}%)"
                for _, r in vs.head(5).iterrows()
            ]
            pr = f"FP&A commentary for {measure} {cmp}:\n" + "\n".join(lines)
            res = cli.chat.completions.create(
                model='gpt-4-turbo-preview',
                messages=[{'role':'user','content':pr}],
                max_tokens=400, temperature=0.3
            )
            st.session_state.commentary = res.choices[0].message.content
        if st.session_state.commentary:
            st.markdown(
                f"<div class='commentary-box'>{st.session_state.commentary.replace(chr(10),'<br>')}</div>",
                unsafe_allow_html=True
            )
    else:
        st.info('Enter a valid OpenAI API key in sidebar')

        # Full report download
    def gen_report(raw, dr, comp, drv, filt, gb_cols):
        buf = BytesIO()
        with pd.ExcelWriter(buf) as writer:
            dr.to_excel(writer, 'Raw_Derived', index=False)
            comp.to_excel(writer, 'Comparison', index=False)
            drv.head(10).to_excel(writer, 'Top_Drivers', index=False)
            # Heatmap: pivot variance percent if possible
            try:
                if len(gb_cols) >= 2:
                    row_dim, col_dim = gb_cols[0], gb_cols[1]
                elif 'Region' in comp.columns:
                    row_dim, col_dim = gb_cols[0] if gb_cols else None, 'Region'
                else:
                    row_dim, col_dim = None, None
                if row_dim and col_dim and row_dim in comp.columns and col_dim in comp.columns:
                    hm = comp.pivot_table(
                        index=row_dim,
                        columns=col_dim,
                        values='Variance_Percent'
                    ).reset_index()
                    hm.to_excel(writer, 'Heatmap', index=False)
            except Exception:
                # Skip heatmap if any error
                pass
            filt.to_excel(writer, 'Filtered_Data', index=False)
        buf.seek(0)
        return buf

    raw = df.copy()
    dr = raw.copy()
    comp = vs.copy()
    drv = vs.copy()
    filt = raw.copy()
    report_io = gen_report(raw, dr, comp, drv, filt, gb)
    st.download_button(
        '💾 Download Excel Report',
        report_io,
        f"var_report_{datetime.now():%Y%m%d}.xlsx",
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )