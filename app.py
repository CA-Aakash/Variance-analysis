import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import openai
from datetime import datetime
import io
import xlsxwriter
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
import base64
import json

# Configure page
st.set_page_config(
    page_title="Variance Analysis AI Copilot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .variance-positive {
        color: #28a745;
        font-weight: bold;
    }
    .variance-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .commentary-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = None
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'commentary' not in st.session_state:
    st.session_state.commentary = ""
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'selected_columns' not in st.session_state:
    st.session_state.selected_columns = {'budget': None, 'actual': None, 'groupby': []}

class VarianceAnalyzer:
    def __init__(self):
        self.df = None
        self.variance_summary = None
        
    def load_data(self, uploaded_file):
        """Load and validate Excel data"""
        try:
            self.df = pd.read_excel(uploaded_file)
            return True, "Data loaded successfully!"
        except Exception as e:
            return False, f"Error loading data: {str(e)}"
    
    def calculate_variances(self, budget_col, actual_col, groupby_cols=None):
        """Calculate absolute and percentage variances"""
        try:
            if groupby_cols:
                # Group by specified columns
                grouped = self.df.groupby(groupby_cols).agg({
                    budget_col: 'sum',
                    actual_col: 'sum'
                }).reset_index()
            else:
                grouped = self.df.copy()
            
            # Calculate variances
            grouped['Variance_Absolute'] = grouped[actual_col] - grouped[budget_col]
            grouped['Variance_Percent'] = (grouped['Variance_Absolute'] / grouped[budget_col]) * 100
            
            # Calculate impact (absolute variance as % of total budget)
            total_budget = grouped[budget_col].sum()
            grouped['Impact_Percent'] = (grouped['Variance_Absolute'] / total_budget) * 100
            
            # Sort by absolute impact
            grouped = grouped.reindex(grouped['Variance_Absolute'].abs().sort_values(ascending=False).index)
            
            self.variance_summary = grouped
            return True, "Variance calculations completed!"
        except Exception as e:
            return False, f"Error calculating variances: {str(e)}"
    
    def get_top_drivers(self, n=5):
        """Get top variance drivers"""
        if self.variance_summary is None:
            return pd.DataFrame()
        
        top_drivers = self.variance_summary.head(n).copy()
        top_drivers['Driver_Type'] = top_drivers['Variance_Absolute'].apply(
            lambda x: 'Favorable' if x > 0 else 'Unfavorable'
        )
        return top_drivers
    
    def generate_waterfall_data(self, budget_col, actual_col):
        """Generate data for waterfall chart"""
        if self.variance_summary is None:
            return None
        
        # Get top positive and negative variances
        top_positive = self.variance_summary[self.variance_summary['Variance_Absolute'] > 0].head(3)
        top_negative = self.variance_summary[self.variance_summary['Variance_Absolute'] < 0].head(3)
        
        waterfall_data = []
        
        # Starting point
        total_budget = self.variance_summary[budget_col].sum()
        waterfall_data.append({
            'Category': 'Budget',
            'Value': total_budget,
            'Type': 'absolute'
        })
        
        # Add positive variances
        for _, row in top_positive.iterrows():
            category = row.iloc[0] if len(row) > 4 else 'Positive Variance'
            waterfall_data.append({
                'Category': f"{category}",
                'Value': row['Variance_Absolute'],
                'Type': 'positive'
            })
        
        # Add negative variances
        for _, row in top_negative.iterrows():
            category = row.iloc[0] if len(row) > 4 else 'Negative Variance'
            waterfall_data.append({
                'Category': f"{category}",
                'Value': row['Variance_Absolute'],
                'Type': 'negative'
            })
        
        # Ending point
        total_actual = self.variance_summary[actual_col].sum()
        waterfall_data.append({
            'Category': 'Actual',
            'Value': total_actual,
            'Type': 'absolute'
        })
        
        return pd.DataFrame(waterfall_data)

class AICommentaryGenerator:
    def __init__(self, api_key):
        try:
            self.client = openai.OpenAI(api_key=api_key)
            print(f"DEBUG: API client initialized successfully")
        except Exception as e:
            print(f"DEBUG: API initialization failed: {str(e)}")
            st.error(f"OpenAI API initialization failed: {str(e)}")
            raise e
    
    def test_connection(self):
        """Test API connection"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            return True, "Connection successful!"
        except Exception as e:
            return False, str(e)
    
    def generate_commentary(self, variance_data, scenario_type="YTD vs Budget", budget_col=None, actual_col=None):
        """Generate AI commentary for variance analysis"""
        try:
            print(f"DEBUG: Starting commentary generation for {len(variance_data)} rows")
            
            # Prepare data summary for LLM
            top_drivers = variance_data.head(5)
            print(f"DEBUG: Top drivers shape: {top_drivers.shape}")
            print(f"DEBUG: Top drivers columns: {top_drivers.columns.tolist()}")
            
            data_summary = []
            for idx, row in top_drivers.iterrows():
                print(f"DEBUG: Processing row {idx}: {row.to_dict()}")
                
                # Find the category column (first non-numeric column or first column)
                category = 'Item'
                budget = 0
                actual = 0
                
                # Try to identify category from first column(s)
                try:
                    # Look for the first string/category column
                    for col_idx, col_name in enumerate(row.index):
                        if col_name not in ['Variance_Absolute', 'Variance_Percent', 'Impact_Percent']:
                            col_value = row.iloc[col_idx]
                            if isinstance(col_value, str) or pd.isna(col_value):
                                category = str(col_value) if not pd.isna(col_value) else 'Item'
                                break
                except:
                    category = 'Item'
                
                # Extract budget and actual values using column names if provided
                try:
                    if budget_col and budget_col in row.index:
                        budget = float(row[budget_col]) if pd.notna(row[budget_col]) else 0
                    else:
                        # Try to find budget column by looking for numeric columns
                        for col_name in row.index:
                            if 'budget' in col_name.lower() or 'Budget' in col_name:
                                budget = float(row[col_name]) if pd.notna(row[col_name]) else 0
                                break
                    
                    if actual_col and actual_col in row.index:
                        actual = float(row[actual_col]) if pd.notna(row[actual_col]) else 0
                    else:
                        # Try to find actual column by looking for numeric columns
                        for col_name in row.index:
                            if 'actual' in col_name.lower() or 'Actual' in col_name:
                                actual = float(row[col_name]) if pd.notna(row[col_name]) else 0
                                break
                except Exception as e:
                    print(f"DEBUG: Error extracting budget/actual: {e}")
                    # Fallback: use variance to calculate approximate values
                    variance_abs = float(row['Variance_Absolute']) if pd.notna(row['Variance_Absolute']) else 0
                    if variance_abs != 0:
                        # Estimate budget as actual minus variance
                        actual = variance_abs + 100000  # Rough estimate
                        budget = actual - variance_abs
                    else:
                        actual = 0
                        budget = 0
                
                variance_abs = float(row['Variance_Absolute']) if pd.notna(row['Variance_Absolute']) else 0
                variance_pct = float(row['Variance_Percent']) if pd.notna(row['Variance_Percent']) else 0
                
                data_summary.append(f"- {category}: Actual ${actual:,.0f} vs Budget ${budget:,.0f} → {variance_abs:+,.0f} ({variance_pct:+.1f}%)")
            
            data_text = "\n".join(data_summary)
            print(f"DEBUG: Data summary prepared:\n{data_text}")
            
            prompt = f"""
You're a senior financial analyst writing variance commentary for executive leadership. 
Analyze this {scenario_type} variance data and provide clear, actionable insights.

VARIANCE DATA:
{data_text}

REQUIREMENTS:
1. Start with an executive summary (2-3 sentences)
2. Highlight the top 3 most significant variances
3. For each significant variance, indicate likely drivers (volume, price, timing, costs)
4. Use professional FP&A language
5. Keep it concise but insightful
6. Format with clear sections

TONE: Professional, analytical, action-oriented
LENGTH: 250-400 words
"""
            
            print(f"DEBUG: Sending request to OpenAI...")
            
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.3
            )
            
            print(f"DEBUG: Received response from OpenAI")
            commentary = response.choices[0].message.content
            print(f"DEBUG: Commentary length: {len(commentary)} characters")
            
            return commentary
            
        except openai.RateLimitError as e:
            error_msg = f"Rate limit exceeded. Please wait and try again. Error: {str(e)}"
            print(f"DEBUG: Rate limit error: {error_msg}")
            return error_msg
        except openai.AuthenticationError as e:
            error_msg = f"Authentication failed. Please check your API key. Error: {str(e)}"
            print(f"DEBUG: Auth error: {error_msg}")
            return error_msg
        except openai.APIError as e:
            error_msg = f"OpenAI API error: {str(e)}"
            print(f"DEBUG: API error: {error_msg}")
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error generating commentary: {str(e)}"
            print(f"DEBUG: Unexpected error: {error_msg}")
            return error_msg

def create_variance_chart(variance_data, chart_type="bar"):
    """Create variance visualization"""
    if variance_data.empty:
        return None
    
    top_10 = variance_data.head(10)
    
    if chart_type == "bar":
        fig = px.bar(
            top_10,
            x=top_10.columns[0],
            y='Variance_Absolute',
            color='Variance_Absolute',
            color_continuous_scale=['red', 'white', 'green'],
            title="Top 10 Variance Drivers",
            labels={'Variance_Absolute': 'Variance ($)'}
        )
        fig.update_layout(
            height=500,
            showlegend=False,
            xaxis_tickangle=-45
        )
        return fig
    
    elif chart_type == "waterfall":
        # Create waterfall chart
        fig = go.Figure()
        
        x_categories = top_10.iloc[:, 0].tolist()
        y_values = top_10['Variance_Absolute'].tolist()
        
        colors = ['green' if x > 0 else 'red' for x in y_values]
        
        fig.add_trace(go.Waterfall(
            name="Variance",
            orientation="v",
            measure=["relative"] * len(y_values),
            x=x_categories,
            y=y_values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "green"}},
            decreasing={"marker": {"color": "red"}},
        ))
        
        fig.update_layout(
            title="Variance Waterfall Analysis",
            height=500,
            xaxis_tickangle=-45
        )
        return fig
    
    return None

def export_to_excel(variance_data, commentary, filename="variance_analysis.xlsx"):
    """Export analysis to Excel"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Write variance data
        variance_data.to_excel(writer, sheet_name='Variance Analysis', index=False)
        
        # Write commentary
        commentary_df = pd.DataFrame({'Commentary': [commentary]})
        commentary_df.to_excel(writer, sheet_name='Commentary', index=False)
        
        # Get workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets['Variance Analysis']
        
        # Format headers
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        
        # Apply header format
        for col_num, value in enumerate(variance_data.columns.values):
            worksheet.write(0, col_num, value, header_format)
    
    output.seek(0)
    return output

def display_analysis_results():
    """Display analysis results - separated into its own function"""
    analyzer = st.session_state.analyzer
    
    if analyzer is None or analyzer.variance_summary is None:
        return
    
    # Get column selections from session state
    budget_col = st.session_state.selected_columns['budget']
    actual_col = st.session_state.selected_columns['actual']
    groupby_cols = st.session_state.selected_columns['groupby']
    
    # Show variance summary
    st.subheader("📈 Variance Summary")
    st.dataframe(analyzer.variance_summary, use_container_width=True)
    
    # Key metrics
    st.subheader("🎯 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    total_budget = analyzer.variance_summary[budget_col].sum()
    total_actual = analyzer.variance_summary[actual_col].sum()
    total_variance = total_actual - total_budget
    variance_pct = (total_variance / total_budget) * 100
    
    with col1:
        st.metric("Total Budget", f"${total_budget:,.0f}")
    
    with col2:
        st.metric("Total Actual", f"${total_actual:,.0f}")
    
    with col3:
        st.metric("Total Variance", f"${total_variance:,.0f}", f"{variance_pct:.1f}%")
    
    with col4:
        favorable_count = len(analyzer.variance_summary[analyzer.variance_summary['Variance_Absolute'] > 0])
        st.metric("Favorable Items", str(favorable_count))
    
    # Visualizations
    st.subheader("📊 Variance Analysis Charts")
    
    tab1, tab2 = st.tabs(["Bar Chart", "Top Drivers"])
    
    with tab1:
        chart = create_variance_chart(analyzer.variance_summary, "bar")
        if chart:
            st.plotly_chart(chart, use_container_width=True)
    
    with tab2:
        top_drivers = analyzer.get_top_drivers(10)
        if not top_drivers.empty:
            st.dataframe(top_drivers, use_container_width=True)

def handle_ai_commentary():
    """Handle AI commentary generation - separated into its own function"""
    api_key = st.session_state.openai_api_key
    scenario_type = st.session_state.get('scenario_type', 'YTD vs Budget')
    
    st.subheader("🧠 AI-Generated Commentary")
    
    if not api_key:
        st.info("🔑 Please enter your OpenAI API key in the sidebar to generate AI commentary")
        return
    elif not api_key.startswith('sk-'):
        st.error("⚠️ Invalid API key format. OpenAI API keys should start with 'sk-'")
        return
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Generate commentary button
        if st.button("🤖 Generate AI Commentary", type="primary", key="generate_commentary"):
            with st.spinner("🔄 Generating AI commentary..."):
                try:
                    st.info("📊 Analyzing variance data...")
                    ai_generator = AICommentaryGenerator(api_key)
                    
                    st.info("🧠 Generating commentary with AI...")
                    commentary = ai_generator.generate_commentary(
                        st.session_state.analyzer.variance_summary, 
                        scenario_type,
                        budget_col=st.session_state.selected_columns['budget'],
                        actual_col=st.session_state.selected_columns['actual']
                    )
                    
                    if commentary.startswith("Error") or commentary.startswith("Rate limit") or commentary.startswith("Authentication"):
                        st.error(f"❌ {commentary}")
                        st.info("💡 **Troubleshooting tips:**\n- Check your API key is correct\n- Ensure you have OpenAI credits\n- Try again in a few moments")
                    else:
                        st.session_state.commentary = commentary
                        st.success("✅ Commentary generated successfully!")
                        st.rerun()  # Refresh to show the commentary
                        
                except Exception as e:
                    st.error(f"❌ Failed to generate commentary: {str(e)}")
                    st.info("💡 **Debug info:**\n- Check your internet connection\n- Verify API key is active\n- Try refreshing the page")
        
        # Test API connection button
        st.markdown("---")
        if st.button("🔍 Test API Connection", type="secondary", key="test_connection"):
            with st.spinner("Testing API connection..."):
                try:
                    ai_generator = AICommentaryGenerator(api_key)
                    success, message = ai_generator.test_connection()
                    if success:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ API connection failed: {message}")
                except Exception as e:
                    st.error(f"❌ API connection failed: {str(e)}")
    
    with col2:
        if st.session_state.commentary:
            st.info(f"📝 Commentary ready! ({len(st.session_state.commentary)} characters)")
        else:
            st.info("💭 Click the button to generate AI commentary")
    
    # Display commentary if available
    if st.session_state.commentary and not st.session_state.commentary.startswith("Error"):
        st.markdown("### 📝 Generated Commentary")
        st.markdown(f"""
        <div class="commentary-box">
            {st.session_state.commentary.replace(chr(10), '<br>')}
        </div>
        """, unsafe_allow_html=True)
        
        # Allow editing
        st.markdown("### ✏️ Edit Commentary")
        edited_commentary = st.text_area(
            "Edit the generated commentary:",
            value=st.session_state.commentary,
            height=200,
            key="commentary_edit",
            help="You can edit the AI-generated commentary before exporting"
        )
        
        if edited_commentary != st.session_state.commentary:
            st.session_state.commentary = edited_commentary
            st.success("✅ Commentary updated!")

# Main App
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Variance Analysis AI Copilot</h1>
        <p>Transform your variance analysis with AI-powered insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("🔧 Configuration")
    
    # API Key input
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.openai_api_key,
        help="Enter your OpenAI API key for AI commentary generation"
    )
    st.session_state.openai_api_key = api_key
    
    # Show API key status
    if api_key:
        if api_key.startswith('sk-'):
            st.sidebar.success("✅ API Key format looks valid")
        else:
            st.sidebar.error("❌ Invalid API key format")
    else:
        st.sidebar.info("🔑 Enter your OpenAI API key to enable AI commentary")
    
    # Analysis type
    scenario_type = st.sidebar.selectbox(
        "Analysis Scenario",
        ["YTD vs Budget", "Month vs Forecast", "Quarter vs Plan", "Actual vs Previous Year"]
    )
    st.session_state.scenario_type = scenario_type
    
    # File upload
    st.sidebar.subheader("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Excel File",
        type=['xlsx', 'xls'],
        help="Upload your variance analysis data"
    )
    
    if uploaded_file is not None:
        # Initialize analyzer if not exists or if new file
        if st.session_state.analyzer is None:
            st.session_state.analyzer = VarianceAnalyzer()
        
        # Load data
        success, message = st.session_state.analyzer.load_data(uploaded_file)
        
        if success:
            st.sidebar.success(message)
            
            # Show data preview
            st.subheader("📊 Data Preview")
            st.dataframe(st.session_state.analyzer.df.head(), use_container_width=True)
            
            # Column selection
            st.subheader("⚙️ Analysis Configuration")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                budget_col = st.selectbox("Budget Column", st.session_state.analyzer.df.columns, key="budget_col_select")
                st.session_state.selected_columns['budget'] = budget_col
            
            with col2:
                actual_col = st.selectbox("Actual Column", st.session_state.analyzer.df.columns, key="actual_col_select")
                st.session_state.selected_columns['actual'] = actual_col
            
            with col3:
                groupby_cols = st.multiselect(
                    "Group By Columns",
                    [col for col in st.session_state.analyzer.df.columns if col not in [budget_col, actual_col]],
                    key="groupby_cols_select"
                )
                st.session_state.selected_columns['groupby'] = groupby_cols
            
            # Calculate variances
            if st.button("🔍 Analyze Variances", type="primary", key="analyze_variances"):
                with st.spinner("Calculating variances..."):
                    success, message = st.session_state.analyzer.calculate_variances(budget_col, actual_col, groupby_cols)
                    
                    if success:
                        st.success(message)
                        st.session_state.analysis_complete = True
                        st.rerun()  # Refresh to show results
                    else:
                        st.error(message)
                        st.session_state.analysis_complete = False
            
            # Display results if analysis is complete
            if st.session_state.analysis_complete and st.session_state.analyzer.variance_summary is not None:
                display_analysis_results()
                handle_ai_commentary()
                
                # Export functionality
                st.subheader("📤 Export Analysis")
                
                if st.button("📊 Export to Excel", type="secondary", key="export_excel"):
                    excel_buffer = export_to_excel(
                        st.session_state.analyzer.variance_summary,
                        st.session_state.commentary,
                        f"variance_analysis_{scenario_type.lower().replace(' ', '_')}.xlsx"
                    )
                    
                    st.download_button(
                        label="💾 Download Excel Report",
                        data=excel_buffer,
                        file_name=f"variance_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        else:
            st.sidebar.error(message)
    
    # Sample data template
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Sample Data Template")
    
    if st.sidebar.button("📥 Download Sample Template"):
        sample_data = pd.DataFrame({
            'Product': ['Product A', 'Product B', 'Product C', 'Product D'],
            'Region': ['North', 'South', 'East', 'West'],
            'Budget': [100000, 150000, 120000, 80000],
            'Actual': [110000, 140000, 125000, 85000],
            'Units_Budget': [500, 750, 600, 400],
            'Units_Actual': [520, 700, 625, 425]
        })
        
        buffer = io.BytesIO()
        sample_data.to_excel(buffer, index=False)
        buffer.seek(0)
        
        st.sidebar.download_button(
            label="💾 Download Sample Data",
            data=buffer,
            file_name="variance_analysis_sample.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    main()