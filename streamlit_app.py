import streamlit as st
import pandas as pd
import numpy as np
import os
from src.prediction import predict_category, load_artifacts
from src.utils import clean_text
from src.expense_advisor import ExpenseAdvisor
from src.data_manager import append_transaction_to_csv, sync_database_to_csv, get_data_stats, validate_data_integrity
from src.model_evaluation import ModelEvaluator
from src.db import (
    init_db,
    insert_transaction as db_insert_transaction,
    fetch_all as db_fetch_all,
    fetch_for_user as db_fetch_for_user,
    list_users as db_list_users,
)
import matplotlib.pyplot as plt
import datetime
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    px = None
    go = None
    PLOTLY_AVAILABLE = False

st.set_page_config(page_title='Expense Assistant', layout='wide')

# paths
DATA_CSV = 'data/improved_transactions.csv'  # Use improved dataset with realistic merchant names
DB_PATH = 'data/expenses.db'

# initialize sqlite DB (migrates CSV into DB if DB empty)
conn = init_db(DB_PATH, DATA_CSV)

# Initialize expense advisor
advisor = ExpenseAdvisor()

# model artifacts: initialize to None and try to load safely
model = None
vectorizer = None
scaler = None
label_encoder = None
tokenizer = None
try:
    model, vectorizer, scaler, label_encoder, tokenizer = load_artifacts()
except Exception as _:
    # don't fail import if artifacts are missing; UI will allow reload
    model = None

def load_dataset(path=DATA_CSV, nrows=None):
    # load from sqlite DB
    try:
        rows = db_fetch_all(conn)
        if not rows:
            return pd.DataFrame(columns=['trans_date_trans_time','merchant','category','amt','first','last'])
        df = pd.DataFrame([dict(r) for r in rows])
        if 'trans_date_trans_time' in df.columns:
            df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
        if 'amt' in df.columns:
            df['amt'] = pd.to_numeric(df['amt'], errors='coerce').fillna(0.0)
        return df
    except Exception:
        return pd.DataFrame(columns=['trans_date_trans_time','merchant','category','amt','first','last'])

def append_transaction(row: dict):
    # insert into sqlite DB and CSV file
    try:
        result = db_insert_transaction(conn, row)
        if result:
            # Also append to CSV file for ML training data
            append_transaction_to_csv(row)
        return result
    except Exception as e:
        st.error(f"Failed to save transaction: {str(e)}")
        return None

def get_user_transactions(first: str, last: str) -> pd.DataFrame:
    """Fetch transactions for a given user from the DB and return as DataFrame."""
    try:
        rows = db_fetch_for_user(conn, first, last)
        if not rows:
            return pd.DataFrame(columns=['trans_date_trans_time','merchant','category','amt','first','last'])
        df = pd.DataFrame([dict(r) for r in rows])
        if 'trans_date_trans_time' in df.columns:
            df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
        if 'amt' in df.columns:
            df['amt'] = pd.to_numeric(df['amt'], errors='coerce').fillna(0.0)
        return df
    except Exception:
        return pd.DataFrame(columns=['trans_date_trans_time','merchant','category','amt','first','last'])

def get_user_suggestions(query: str) -> list:
    """Return list of 'First Last' strings from DB matching query case-insensitively."""
    try:
        all_users = db_list_users(conn)
        q = (query or '').strip().lower()
        results = []
        for f, l in all_users:
            name = f"{(f or '').strip()} {(l or '').strip()}".strip()
            if not q:
                results.append(name)
            else:
                if q in name.lower() or q in (f or '').lower() or q in (l or '').lower():
                    results.append(name)
        # unique preserving order
        seen = set()
        uniq = []
        for r in results:
            if r not in seen:
                seen.add(r)
                uniq.append(r)
        return uniq
    except Exception:
        return []

def predict_category_safe(text, amount):
    try:
        return predict_category(text, amount)
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

def reload_model_artifacts():
    """Reload model artifacts and return status"""
    try:
        global model, vectorizer, scaler, label_encoder, tokenizer
        model, vectorizer, scaler, label_encoder, tokenizer = load_artifacts()
        return True, "Artifacts reloaded successfully"
    except Exception as e:
        return False, f"Failed to reload artifacts: {str(e)}"

# Initialize user selection state
if 'user_selected' not in st.session_state:
    st.session_state['user_selected'] = False

# Main user selection page
if not st.session_state['user_selected']:
    st.title('🏦 Expense Tracker')
    st.markdown('### Welcome! Please select your user type:')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="padding: 20px; border: 2px solid #4CAF50; border-radius: 10px; text-align: center; margin: 10px;">
            <h3>🔐 Existing User</h3>
            <p>Sign in to access your expense history</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button('Sign In as Existing User', key='existing_user_btn', use_container_width=True):
            st.session_state['user_type'] = 'existing'
            st.session_state['user_selected'] = True
            st.rerun()
    
    with col2:
        st.markdown("""
        <div style="padding: 20px; border: 2px solid #2196F3; border-radius: 10px; text-align: center; margin: 10px;">
            <h3>✨ New User</h3>
            <p>Create a new account and start tracking</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button('Create New Account', key='new_user_btn', use_container_width=True):
            st.session_state['user_type'] = 'new'
            st.session_state['user_selected'] = True
            st.rerun()

else:
    # Load dataset after user selection
    df_all = load_dataset()
    
    # Show user type and back button
    st.sidebar.markdown(f"**User Type:** {st.session_state.get('user_type', 'Unknown').title()}")
    if st.sidebar.button('← Back to User Selection'):
        st.session_state['user_selected'] = False
        st.session_state.pop('current_user', None)
        st.rerun()
    
    # --- Main app tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(['➕ Add Expense', '📈 Analysis', '📊 Dashboard', '🤖 ML Models', '⚙️ Settings'])

    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader('Add Expense')
        
        user_type = st.session_state.get('user_type', 'existing')
        
        if user_type == 'existing':
            st.markdown('**🔐 Sign in**')
            # search / suggestion helper
            search = st.text_input('Search user by name (type any part of first or last)', key='search_user')
            suggestions = get_user_suggestions(search)
            if suggestions:
                sel = st.selectbox('Matching users (select to autofill)', options=[''] + suggestions, key='suggest_select')
                if sel:
                    parts = sel.split(' ', 1)
                    sf = parts[0]
                    sl = parts[1] if len(parts) > 1 else ''
                    # prefill signin inputs via session state
                    st.session_state['signin_first'] = sf
                    st.session_state['signin_last'] = sl

            e1, e2 = st.columns(2)
            with e1:
                first = st.text_input('First name', key='signin_first')
            with e2:
                last = st.text_input('Last name', key='signin_last')
            if st.button('Load my data', key='load_user'):
                if not first or not last:
                    st.error('Please enter both first and last name')
                else:
                    # try to find a case-insensitive match in DB and normalize to stored casing
                    matched = None
                    try:
                        all_users = db_list_users(conn)
                        for fu, lu in all_users:
                            if fu and lu and fu.strip().lower() == first.strip().lower() and lu.strip().lower() == last.strip().lower():
                                matched = (fu, lu)
                                break
                    except Exception:
                        matched = None
                    if matched:
                        st.session_state['current_user'] = {'first': matched[0], 'last': matched[1]}
                        st.success(f'Loaded user {matched[0]} {matched[1]}')
                    else:
                        # no exact match in DB; still set current user (will create new rows on save)
                        st.session_state['current_user'] = {'first': first, 'last': last}
                        st.info(f'No existing user found; proceeding with name {first} {last} (will create on save)')

            # If a user is loaded (from above or earlier), show the Add Expense form
            if 'current_user' in st.session_state:
                cu = st.session_state['current_user']
                # show recent transactions for this user (exclude merchant column)
                try:
                    tx = get_user_transactions(cu.get('first'), cu.get('last'))
                    if not tx.empty:
                        st.markdown('**Recent transactions**')
                        # Display only relevant columns, exclude merchant
                        display_cols = ['trans_date_trans_time', 'category', 'amt']
                        if all(col in tx.columns for col in display_cols):
                            display_tx = tx[display_cols].sort_values('trans_date_trans_time', ascending=False).head(10)
                            display_tx.columns = ['Date & Time', 'Category', 'Amount ($)']
                            st.dataframe(display_tx, use_container_width=True)
                        else:
                            st.dataframe(tx.sort_values('trans_date_trans_time', ascending=False).head(10))
                    else:
                        st.info('No transactions yet for this user')
                except Exception:
                    pass
                st.markdown('---')
                st.write(f"Adding expense for {cu.get('first')} {cu.get('last')}")
                
                # Use a form to handle the expense input properly
                with st.form("expense_form", clear_on_submit=True):
                    desc = st.text_input('Description', placeholder='e.g., Starbucks Coffee, Car Loan Payment, Grocery Shopping')
                    amt = st.number_input('Amount', min_value=0.0, value=0.0)
                    dcol, tcol = st.columns([1,1])
                    with dcol:
                        add_date = st.date_input('Date', value=datetime.datetime.now().date())
                    with tcol:
                        add_time = st.time_input('Time', value=datetime.datetime.now().time())
                
                    # show prediction preview and suggestions
                    if desc and amt > 0:
                        try:
                            pred_cat, conf = predict_category_safe(desc, amt)
                            if pred_cat:
                                st.info(f'🤖 Predicted category: **{pred_cat}** (confidence: {conf:.2f})')
                                
                                # Show category-specific insights
                                insights = advisor.get_category_insights(pred_cat, amt)
                                if insights:
                                    st.markdown("**💡 Smart Savings Tips:**")
                                    for insight in insights:
                                        st.markdown(f"• {insight}")
                        except Exception as e:
                            st.warning(f'Prediction failed: {str(e)}')
                    
                    # Form submit button
                    submitted = st.form_submit_button('💾 Save Expense')
                    
                    if submitted:
                        if not desc or amt <= 0:
                            st.error('Please provide a valid description and amount')
                        else:
                            first = cu.get('first')
                            last = cu.get('last')
                            try:
                                pred_cat, conf = predict_category_safe(desc, amt) if desc else (None, None)
                                # Use only ML predicted category
                                final_category = pred_cat if pred_cat else 'misc_pos'
                                ts = datetime.datetime.combine(add_date, add_time)
                                row = {
                                    'trans_date_trans_time': ts.strftime('%Y-%m-%d %H:%M:%S'),
                                    'merchant': desc,
                                    'category': final_category,
                                    'amt': amt,
                                    'first': first,
                                    'last': last
                                }
                                result = append_transaction(row)
                                if result is not None:
                                    st.success(f'✅ Expense saved! Category: {final_category}')
                                    st.rerun()
                            except Exception as e:
                                st.error(f'Failed to save expense: {str(e)}')

        else:  # new user
            st.markdown('**✨ Quick onboarding**')
            if 'onboard_step' not in st.session_state:
                st.session_state['onboard_step'] = 1
                st.session_state['new_user_data'] = {}

            step = st.session_state['onboard_step']
            if step == 1:
                st.write('Step 1 — Tell us your name')
                nf, nl = st.columns(2)
                with nf:
                    nf_val = st.text_input('First name', key='onb_first')
                with nl:
                    nl_val = st.text_input('Last name', key='onb_last')
                if st.button('Next', key='onb_next1'):
                    if not nf_val or not nl_val:
                        st.error('Please provide names to continue')
                    else:
                        st.session_state['new_user_data']['first'] = nf_val
                        st.session_state['new_user_data']['last'] = nl_val
                        st.session_state['onboard_step'] = 2
                        st.rerun()
            elif step == 2:
                st.write('Step 2 — Monthly income (used for budgets)')
                mi = st.number_input('Monthly income', min_value=0.0, value=0.0, key='onb_income')
                if st.button('Next', key='onb_next2'):
                    st.session_state['new_user_data']['monthly_income'] = mi
                    st.session_state['onboard_step'] = 3
                    st.rerun()
            else:
                st.write('Step 3 — Add first expense (optional)')
                
                with st.form("onboarding_expense_form"):
                    desc = st.text_input('Description', placeholder='e.g., Starbucks Coffee, Car Loan Payment')
                    amt = st.number_input('Amount', min_value=0.0, value=0.0)
                    dcol, tcol = st.columns([1,1])
                    with dcol:
                        onb_date = st.date_input('Date', value=datetime.datetime.now().date())
                    with tcol:
                        onb_time = st.time_input('Time', value=datetime.datetime.now().time())
                    
                    # Show prediction for new user too
                    if desc and amt > 0:
                        try:
                            pred_cat, conf = predict_category_safe(desc, amt)
                            if pred_cat:
                                st.info(f'🤖 Predicted category: **{pred_cat}** (confidence: {conf:.2f})')
                        except Exception as e:
                            st.warning(f'Prediction failed: {str(e)}')
                    
                    create_account = st.form_submit_button('🎉 Create Account')
                    
                    if create_account:
                        first = st.session_state['new_user_data'].get('first')
                        last = st.session_state['new_user_data'].get('last')
                        
                        if desc and amt > 0:
                            try:
                                pred_cat, conf = predict_category_safe(desc, amt)
                                final_category = pred_cat if pred_cat else 'misc_pos'
                                ts = datetime.datetime.combine(onb_date, onb_time)
                                row = {
                                    'trans_date_trans_time': ts.strftime('%Y-%m-%d %H:%M:%S'),
                                    'merchant': desc,
                                    'category': final_category,
                                    'amt': amt,
                                    'first': first,
                                    'last': last
                                }
                                append_transaction(row)
                            except Exception as e:
                                st.error(f'Failed to save expense: {str(e)}')
                        
                        st.balloons()
                        st.success('Account created — you can now use Add Expense or Analysis')
                        st.session_state['current_user'] = {'first': first, 'last': last}
                        st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader('💡 Smart Analysis & Savings Tips')
        
        # Use current_user if present, else global
        if 'current_user' in st.session_state:
            cu = st.session_state['current_user']
            first = cu.get('first')
            last = cu.get('last')
            # fetch from DB so newly added transactions are visible immediately
            user_df = get_user_transactions(first, last)
            st.write(f'Analyzing expenses for {first} {last}')
            
            # Get monthly income if available
            monthly_income = st.session_state.get('new_user_data', {}).get('monthly_income')
            if not monthly_income:
                monthly_income = st.number_input('Monthly Income (optional - for better analysis)', 
                                               min_value=0.0, value=0.0, key='analysis_income')
                if monthly_income > 0:
                    if 'new_user_data' not in st.session_state:
                        st.session_state['new_user_data'] = {}
                    st.session_state['new_user_data']['monthly_income'] = monthly_income
        else:
            user_df = df_all.copy()
            monthly_income = None

        if user_df.empty:
            st.info('No transactions to analyze. Add some expenses to get personalized insights!')
            
            # Show demo graphs with sample data
            st.markdown("### 📊 Demo Analysis (Sample Data)")
            st.info("👆 This is what your analysis will look like once you add expenses!")
            
            # Create sample data for demo
            sample_categories = ['food_dining', 'gas_transport', 'grocery_pos', 'shopping_pos', 'entertainment']
            sample_amounts = [250.50, 180.75, 320.25, 150.00, 95.30]
            
            col1, col2 = st.columns(2)
            
            with col1:
                if PLOTLY_AVAILABLE:
                    # Demo Pie Chart
                    fig_pie = px.pie(
                        values=sample_amounts, 
                        names=sample_categories,
                        title='📊 Spending Distribution by Category (Demo)',
                        color_discrete_sequence=px.colors.qualitative.Set3,
                        hole=0.4
                    )
                    fig_pie.update_traces(
                        textposition='inside', 
                        textinfo='percent+label',
                        textfont_size=10
                    )
                    fig_pie.update_layout(
                        showlegend=True,
                        height=450,
                        font=dict(size=11),
                        title_font_size=16
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                if PLOTLY_AVAILABLE:
                    # Demo Bar Chart
                    fig_bar = px.bar(
                        x=sample_amounts, 
                        y=sample_categories,
                        title='💰 Top Categories by Spending (Demo)',
                        labels={'x': 'Amount ($)', 'y': 'Category'},
                        color=sample_amounts,
                        color_continuous_scale='viridis',
                        orientation='h'
                    )
                    fig_bar.update_layout(
                        height=450,
                        showlegend=False,
                        title_font_size=16,
                        yaxis={'categoryorder': 'total ascending'},
                        xaxis_title="Spending Amount ($)",
                        yaxis_title="Expense Category"
                    )
                    fig_bar.update_traces(
                        texttemplate='$%{x:.0f}',
                        textposition='outside'
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
        else:
            # Get personalized analysis and tips
            with st.spinner('Analyzing your spending patterns...'):
                tips_data = advisor.get_personalized_tips(user_df, monthly_income)
                analysis = tips_data['analysis']
                tips = tips_data['tips']
                
                # Display key metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Spending", f"${analysis['total_spending']:.2f}")
                with col2:
                    st.metric("Transactions", analysis['transaction_count'])
                with col3:
                    st.metric("Avg Transaction", f"${analysis['avg_transaction']:.2f}")
                with col4:
                    trend_emoji = {"increasing": "📈", "decreasing": "📉", "stable": "➡️"}
                    st.metric("Trend", f"{trend_emoji.get(analysis['trend'], '➡️')} {analysis['trend'].title()}")
                
                # Two Key Professional Graphs
                st.markdown("### 📊 Spending Analysis - Key Insights")
                category_data = analysis['category_spending']
                if category_data:
                    category_df = pd.DataFrame.from_dict(category_data, orient='index')
                    category_df = category_df.sort_values('sum', ascending=False)
                    
                    # Create two columns for the 2 main graphs
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if PLOTLY_AVAILABLE:
                            # Graph 1: Professional Pie Chart - Spending Distribution
                            fig_pie = px.pie(
                                values=category_df['sum'], 
                                names=category_df.index,
                                title='📊 Spending Distribution by Category',
                                color_discrete_sequence=px.colors.qualitative.Set3,
                                hole=0.4  # Make it a donut chart for modern look
                            )
                            fig_pie.update_traces(
                                textposition='inside', 
                                textinfo='percent+label',
                                textfont_size=10
                            )
                            fig_pie.update_layout(
                                showlegend=True,
                                height=450,
                                font=dict(size=11),
                                title_font_size=16,
                                legend=dict(
                                    orientation="v",
                                    yanchor="middle",
                                    y=0.5,
                                    xanchor="left",
                                    x=1.05
                                )
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col2:
                        if PLOTLY_AVAILABLE:
                            # Graph 2: Professional Bar Chart - Top Categories
                            top_categories = category_df.head(8)
                            fig_bar = px.bar(
                                x=top_categories['sum'], 
                                y=top_categories.index,
                                title='💰 Top Categories by Spending Amount',
                                labels={'x': 'Amount ($)', 'y': 'Category'},
                                color=top_categories['sum'],
                                color_continuous_scale='viridis',
                                orientation='h'  # Horizontal bar chart for better readability
                            )
                            fig_bar.update_layout(
                                height=450,
                                showlegend=False,
                                title_font_size=16,
                                yaxis={'categoryorder': 'total ascending'},
                                xaxis_title="Spending Amount ($)",
                                yaxis_title="Expense Category"
                            )
                            fig_bar.update_traces(
                                texttemplate='$%{x:.0f}',
                                textposition='outside'
                            )
                            st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # Professional summary table
                    st.markdown("**📈 Category Summary Table:**")
                    display_df = category_df.head(6).copy()
                    display_df.columns = ['Total Spent ($)', 'Transaction Count', 'Average per Transaction ($)']
                    display_df['Total Spent ($)'] = display_df['Total Spent ($)'].round(2)
                    display_df['Average per Transaction ($)'] = display_df['Average per Transaction ($)'].round(2)
                    
                    # Add percentage column
                    total_spending = display_df['Total Spent ($)'].sum()
                    display_df['% of Total Spending'] = (display_df['Total Spent ($)'] / total_spending * 100).round(1)
                    
                    st.dataframe(display_df, use_container_width=True)
                
                # Enhanced Time-Based Analysis with Dropdown
                st.markdown("### 📅 Time-Based Expense Analysis")
                
                # Dropdown for time period selection
                time_period = st.selectbox(
                    "📊 Select Analysis Period:",
                    ["Monthly", "Weekly", "Yearly"],
                    key="time_period_selector"
                )
                
                # Prepare data based on selected time period
                if not user_df.empty and 'trans_date_trans_time' in user_df.columns:
                    user_df_time = user_df.copy()
                    user_df_time['trans_date_trans_time'] = pd.to_datetime(user_df_time['trans_date_trans_time'])
                    
                    # Create time-based groupings
                    if time_period == "Monthly":
                        user_df_time['period'] = user_df_time['trans_date_trans_time'].dt.to_period('M')
                        period_format = '%Y-%m'
                        period_name = "Month"
                        comparison_text = "month"
                    elif time_period == "Weekly":
                        user_df_time['period'] = user_df_time['trans_date_trans_time'].dt.to_period('W')
                        period_format = '%Y-W%U'
                        period_name = "Week"
                        comparison_text = "week"
                    else:  # Yearly
                        user_df_time['period'] = user_df_time['trans_date_trans_time'].dt.to_period('Y')
                        period_format = '%Y'
                        period_name = "Year"
                        comparison_text = "year"
                    
                    # Group by period and calculate totals
                    period_spending = user_df_time.groupby('period')['amt'].agg(['sum', 'count', 'mean']).reset_index()
                    period_spending['period_str'] = period_spending['period'].astype(str)
                    period_spending = period_spending.sort_values('period')
                    
                    if len(period_spending) > 0:
                        # Create two columns for chart and statistics
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            if PLOTLY_AVAILABLE and len(period_spending) > 0:
                                # Create interactive time series chart
                                fig_time = px.line(
                                    period_spending,
                                    x='period_str',
                                    y='sum',
                                    title=f'📈 {time_period} Spending Trend',
                                    labels={'period_str': period_name, 'sum': 'Amount ($)'},
                                    markers=True,
                                    line_shape='spline'
                                )
                                
                                # Add trend line if enough data points
                                if len(period_spending) > 2:
                                    from scipy import stats
                                    x_numeric = range(len(period_spending))
                                    slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, period_spending['sum'])
                                    trend_line = [slope * x + intercept for x in x_numeric]
                                    
                                    fig_time.add_scatter(
                                        x=period_spending['period_str'],
                                        y=trend_line,
                                        mode='lines',
                                        name='Trend Line',
                                        line=dict(dash='dash', color='red', width=2)
                                    )
                                
                                # Enhance chart appearance
                                fig_time.update_layout(
                                    height=400,
                                    hovermode='x unified',
                                    showlegend=True,
                                    xaxis_title=period_name,
                                    yaxis_title="Spending Amount ($)",
                                    title_font_size=16
                                )
                                
                                # Add hover information
                                fig_time.update_traces(
                                    hovertemplate=f'<b>{period_name}: %{{x}}</b><br>' +
                                                'Amount: $%{y:.2f}<br>' +
                                                '<extra></extra>'
                                )
                                
                                st.plotly_chart(fig_time, use_container_width=True)
                        
                        with col2:
                            # Statistics and trend analysis
                            st.markdown(f"**📊 {time_period} Statistics:**")
                            
                            avg_spending = period_spending['sum'].mean()
                            max_period = period_spending.loc[period_spending['sum'].idxmax(), 'period_str']
                            min_period = period_spending.loc[period_spending['sum'].idxmin(), 'period_str']
                            max_amount = period_spending['sum'].max()
                            min_amount = period_spending['sum'].min()
                            
                            st.metric(f"Average {period_name}", f"${avg_spending:.2f}")
                            st.metric(f"Highest {period_name}", f"{max_period}")
                            st.metric("Peak Amount", f"${max_amount:.2f}")
                            st.metric(f"Lowest {period_name}", f"{min_period}")
                            st.metric("Minimum Amount", f"${min_amount:.2f}")
                            
                            # Trend Analysis and Comparison
                            if len(period_spending) >= 2:
                                current_amount = period_spending['sum'].iloc[-1]
                                previous_amount = period_spending['sum'].iloc[-2]
                                change_amount = current_amount - previous_amount
                                change_pct = (change_amount / previous_amount) * 100 if previous_amount != 0 else 0
                                
                                st.markdown(f"**📈 {period_name} Comparison:**")
                                
                                # Current vs Previous period comparison
                                current_period = period_spending['period_str'].iloc[-1]
                                previous_period = period_spending['period_str'].iloc[-2]
                                
                                if change_pct > 5:
                                    st.error(f"📈 **Spending Increased!**\n\n"
                                           f"Current {comparison_text} ({current_period}): ${current_amount:.2f}\n\n"
                                           f"Previous {comparison_text} ({previous_period}): ${previous_amount:.2f}\n\n"
                                           f"**Increase: ${change_amount:.2f} ({change_pct:.1f}%)**")
                                elif change_pct < -5:
                                    st.success(f"📉 **Spending Decreased!**\n\n"
                                             f"Current {comparison_text} ({current_period}): ${current_amount:.2f}\n\n"
                                             f"Previous {comparison_text} ({previous_period}): ${previous_amount:.2f}\n\n"
                                             f"**Decrease: ${abs(change_amount):.2f} ({abs(change_pct):.1f}%)**")
                                else:
                                    st.info(f"➡️ **Spending Stable**\n\n"
                                           f"Current {comparison_text} ({current_period}): ${current_amount:.2f}\n\n"
                                           f"Previous {comparison_text} ({previous_period}): ${previous_amount:.2f}\n\n"
                                           f"**Change: ${change_amount:.2f} ({change_pct:.1f}%)**")
                                
                                # Long-term trend analysis (if enough data)
                                if len(period_spending) >= 4:
                                    # Calculate trend over last 4 periods
                                    recent_periods = period_spending.tail(4)
                                    first_period_avg = recent_periods['sum'].iloc[:2].mean()
                                    last_period_avg = recent_periods['sum'].iloc[-2:].mean()
                                    long_term_change = ((last_period_avg - first_period_avg) / first_period_avg) * 100 if first_period_avg != 0 else 0
                                    
                                    st.markdown("**📊 Long-term Trend:**")
                                    if long_term_change > 10:
                                        st.warning(f"⚠️ Upward trend: +{long_term_change:.1f}% over recent periods")
                                    elif long_term_change < -10:
                                        st.success(f"✅ Downward trend: {long_term_change:.1f}% over recent periods")
                                    else:
                                        st.info(f"📊 Stable trend: {long_term_change:.1f}% over recent periods")
                        
                        # Detailed period breakdown table
                        st.markdown(f"### 📋 Detailed {time_period} Breakdown")
                        
                        # Prepare display dataframe
                        display_df = period_spending.copy()
                        display_df = display_df.sort_values('period', ascending=False)  # Most recent first
                        display_df['Total Spent'] = display_df['sum'].round(2)
                        display_df['Transactions'] = display_df['count']
                        display_df['Average per Transaction'] = display_df['mean'].round(2)
                        
                        # Add percentage of total
                        total_all_periods = display_df['sum'].sum()
                        display_df['% of Total'] = (display_df['sum'] / total_all_periods * 100).round(1)
                        
                        # Add trend indicators
                        display_df['Trend'] = ''
                        for i in range(1, len(display_df)):
                            current = display_df.iloc[i-1]['sum']
                            previous = display_df.iloc[i]['sum']
                            if previous != 0:
                                change = ((current - previous) / previous) * 100
                                if change > 5:
                                    display_df.iloc[i-1, display_df.columns.get_loc('Trend')] = f"📈 +{change:.1f}%"
                                elif change < -5:
                                    display_df.iloc[i-1, display_df.columns.get_loc('Trend')] = f"📉 {change:.1f}%"
                                else:
                                    display_df.iloc[i-1, display_df.columns.get_loc('Trend')] = f"➡️ {change:.1f}%"
                        
                        # Display the table
                        final_df = display_df[['period_str', 'Total Spent', 'Transactions', 'Average per Transaction', '% of Total', 'Trend']]
                        final_df.columns = [period_name, 'Total Spent ($)', 'Transactions', 'Avg per Transaction ($)', '% of Total', 'Trend vs Previous']
                        
                        st.dataframe(final_df, use_container_width=True)
                        
                        # Summary insights
                        st.markdown(f"### 💡 {time_period} Insights")
                        
                        insights = []
                        
                        # Spending pattern insights
                        if len(period_spending) >= 3:
                            # Find most consistent period
                            std_dev = period_spending['sum'].std()
                            avg_spending = period_spending['sum'].mean()
                            cv = (std_dev / avg_spending) * 100 if avg_spending != 0 else 0
                            
                            if cv < 20:
                                insights.append(f"✅ **Consistent Spending**: Your {comparison_text}ly spending is very consistent (variation: {cv:.1f}%)")
                            elif cv > 50:
                                insights.append(f"⚠️ **Variable Spending**: Your {comparison_text}ly spending varies significantly (variation: {cv:.1f}%)")
                            
                            # Seasonal patterns (for monthly/weekly data)
                            if time_period in ["Monthly", "Weekly"] and len(period_spending) >= 6:
                                recent_avg = period_spending.tail(3)['sum'].mean()
                                earlier_avg = period_spending.head(3)['sum'].mean()
                                
                                if recent_avg > earlier_avg * 1.2:
                                    insights.append(f"📈 **Increasing Pattern**: Recent {comparison_text}s show 20%+ higher spending")
                                elif recent_avg < earlier_avg * 0.8:
                                    insights.append(f"📉 **Decreasing Pattern**: Recent {comparison_text}s show 20%+ lower spending")
                        
                        # Display insights
                        for insight in insights:
                            st.markdown(insight)
                        
                        if not insights:
                            st.info(f"💡 Add more transactions over multiple {comparison_text}s to get detailed spending insights!")
                    
                    else:
                        st.info(f"📊 No {comparison_text}ly data available yet. Add more transactions to see trends!")
                
                else:
                    st.info("📅 Add some transactions to see time-based analysis!")
                
                # Personalized Tips Section
                st.markdown("### 💰 Your Personalized Savings Tips")
                st.markdown("Based on your spending patterns, here are tailored recommendations:")
                
                # Display tips in an organized way
                for i, tip in enumerate(tips, 1):
                    st.markdown(f"**{i}.** {tip}")
                
                # Professional Budget Analysis
                if monthly_income and monthly_income > 0:
                    st.markdown("### 💰 Professional Budget Analysis")
                    monthly_spending = analysis['total_spending'] / max(1, len(analysis['monthly_spending']))
                    spending_ratio = monthly_spending / monthly_income
                    remaining = monthly_income - monthly_spending
                    
                    # Key metrics in a professional layout
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Monthly Income", f"${monthly_income:.2f}")
                    with col2:
                        st.metric("Monthly Spending", f"${monthly_spending:.2f}")
                    with col3:
                        delta_color = "normal" if remaining >= 0 else "inverse"
                        st.metric("Net Surplus/Deficit", f"${remaining:.2f}", delta=f"${remaining:.2f}")
                    with col4:
                        st.metric("Spending Ratio", f"{spending_ratio:.1%}")
                    

                    
                    # Professional budget recommendations with scoring
                    st.markdown("**🎯 Budget Health Score & Recommendations:**")
                    
                    # Calculate budget health score
                    if spending_ratio <= 0.5:
                        score = 100
                        grade = "A+"
                        color = "success"
                        message = "🏆 Excellent! You're a financial superstar with great spending discipline."
                    elif spending_ratio <= 0.7:
                        score = 85
                        grade = "A"
                        color = "success"
                        message = "✅ Great job! You're maintaining very healthy spending habits."
                    elif spending_ratio <= 0.8:
                        score = 70
                        grade = "B"
                        color = "info"
                        message = "💡 Good spending habits. Consider the 50/30/20 budgeting rule for optimization."
                    elif spending_ratio <= 0.9:
                        score = 50
                        grade = "C"
                        color = "warning"
                        message = "⚠️ Caution: You're spending 80-90% of income. Look for areas to reduce expenses."
                    else:
                        score = 25
                        grade = "D"
                        color = "error"
                        message = "🚨 Critical: You're spending over 90% of income. Immediate action required!"
                    
                    # Display score with appropriate styling
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if color == "success":
                            st.success(f"**Budget Health Score: {score}/100 (Grade: {grade})**")
                        elif color == "info":
                            st.info(f"**Budget Health Score: {score}/100 (Grade: {grade})**")
                        elif color == "warning":
                            st.warning(f"**Budget Health Score: {score}/100 (Grade: {grade})**")
                        else:
                            st.error(f"**Budget Health Score: {score}/100 (Grade: {grade})**")
                        
                        st.write(message)
                    
                    # 50/30/20 Rule Analysis
                    st.markdown("**📊 50/30/20 Rule Analysis:**")
                    needs_budget = monthly_income * 0.5
                    wants_budget = monthly_income * 0.3
                    savings_budget = monthly_income * 0.2
                    
                    rule_col1, rule_col2, rule_col3 = st.columns(3)
                    with rule_col1:
                        st.metric("Needs (50%)", f"${needs_budget:.2f}", 
                                help="Essential expenses: housing, utilities, groceries, transportation")
                    with rule_col2:
                        st.metric("Wants (30%)", f"${wants_budget:.2f}",
                                help="Entertainment, dining out, hobbies, non-essential shopping")
                    with rule_col3:
                        st.metric("Savings (20%)", f"${savings_budget:.2f}",
                                help="Emergency fund, retirement, investments, debt repayment")

        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader('Dashboard Overview')
        
        # Show basic stats only, no charts on main dashboard
        if 'current_user' in st.session_state:
            cu = st.session_state['current_user']
            user_df = get_user_transactions(cu.get('first'), cu.get('last'))
            st.write(f'Dashboard for {cu.get("first")} {cu.get("last")}')
            
            if not user_df.empty:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Expenses", f"${user_df['amt'].sum():.2f}")
                with col2:
                    st.metric("Total Transactions", len(user_df))
                with col3:
                    st.metric("Average Amount", f"${user_df['amt'].mean():.2f}")
                with col4:
                    st.metric("Categories", user_df['category'].nunique())
                
                st.markdown("**Recent Transactions**")
                # Display only relevant columns, exclude merchant
                display_cols = ['trans_date_trans_time', 'category', 'amt']
                if all(col in user_df.columns for col in display_cols):
                    display_tx = user_df[display_cols].sort_values('trans_date_trans_time', ascending=False).head(5)
                    display_tx.columns = ['Date & Time', 'Category', 'Amount ($)']
                    st.dataframe(display_tx, use_container_width=True)
                else:
                    st.dataframe(user_df.sort_values('trans_date_trans_time', ascending=False).head(5))
            else:
                st.info("No transactions found. Add some expenses to see your dashboard!")
        else:
            st.info("Please sign in or create an account to view your dashboard.")
        
        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader('🤖 Machine Learning Model Analytics')
        
        # Load benchmark report
        evaluator = ModelEvaluator()
        benchmark_report = evaluator.load_benchmark_report()
        
        if benchmark_report:
            st.markdown("### 🏆 Benchmark Model Performance")
            
            # Display benchmark model info
            benchmark_model = benchmark_report.get('benchmark_model', 'Unknown')
            benchmark_metrics = benchmark_report.get('benchmark_metrics', {})
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🏆 Best Model", benchmark_model)
            with col2:
                accuracy = benchmark_metrics.get('accuracy', 0)
                st.metric("📊 Accuracy", f"{accuracy:.4f}")
            with col3:
                f1_score = benchmark_metrics.get('f1_weighted', 0)
                st.metric("📈 F1-Score", f"{f1_score:.4f}")
            with col4:
                precision = benchmark_metrics.get('precision_weighted', 0)
                st.metric("🎯 Precision", f"{precision:.4f}")
            
            # Detailed Model Results Section
            st.markdown("### 📋 Comprehensive Model Results")
            
            model_results = benchmark_report.get('model_results', {})
            
            # Create tabs for each model
            if model_results:
                model_names = list(model_results.keys())
                model_tabs = st.tabs([f"📊 {name.replace('_', ' ').title()}" for name in model_names])
                
                for i, (model_name, tab) in enumerate(zip(model_names, model_tabs)):
                    with tab:
                        model_data = model_results[model_name]
                        
                        # Key metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Accuracy", f"{model_data.get('accuracy', 0):.4f}")
                        with col2:
                            st.metric("F1-Score", f"{model_data.get('f1_weighted', 0):.4f}")
                        with col3:
                            st.metric("Precision", f"{model_data.get('precision_weighted', 0):.4f}")
                        with col4:
                            st.metric("Recall", f"{model_data.get('recall_weighted', 0):.4f}")
                        
                        # Per-class performance
                        if 'precision_per_class' in model_data and 'class_names' in model_data:
                            st.markdown("**📊 Per-Category Performance:**")
                            
                            class_names = model_data['class_names']
                            precision_per_class = model_data.get('precision_per_class', [])
                            recall_per_class = model_data.get('recall_per_class', [])
                            f1_per_class = model_data.get('f1_per_class', [])
                            
                            if len(precision_per_class) == len(class_names):
                                class_performance = pd.DataFrame({
                                    'Category': class_names,
                                    'Precision': [round(p, 4) for p in precision_per_class],
                                    'Recall': [round(r, 4) for r in recall_per_class],
                                    'F1-Score': [round(f, 4) for f in f1_per_class]
                                })
                                
                                # Sort by F1-Score descending
                                class_performance = class_performance.sort_values('F1-Score', ascending=False)
                                st.dataframe(class_performance, use_container_width=True)
                                
                                # Visualization of per-class performance
                                if PLOTLY_AVAILABLE:
                                    fig_class = px.bar(
                                        class_performance.head(10),  # Top 10 categories
                                        x='Category',
                                        y='F1-Score',
                                        title=f'{model_data["model_name"]} - F1-Score by Category',
                                        color='F1-Score',
                                        color_continuous_scale='viridis'
                                    )
                                    fig_class.update_layout(
                                        height=400,
                                        xaxis_tickangle=-45,
                                        showlegend=False
                                    )
                                    st.plotly_chart(fig_class, use_container_width=True)
                        
                        # Special handling for RNN results
                        if model_name == 'rnn' and 'training_history' in model_data:
                            st.markdown("**🧠 RNN Training Progress:**")
                            training_history = model_data['training_history']
                            
                            hist_col1, hist_col2 = st.columns(2)
                            with hist_col1:
                                st.metric("Final Training Loss", f"{training_history.get('final_loss', 0):.4f}")
                                st.metric("Final Training Accuracy", f"{training_history.get('final_accuracy', 0):.4f}")
                            with hist_col2:
                                st.metric("Final Validation Loss", f"{training_history.get('final_val_loss', 0):.4f}")
                                st.metric("Final Validation Accuracy", f"{training_history.get('final_val_accuracy', 0):.4f}")
                        
                        # Model-specific insights
                        st.markdown("**💡 Model Insights:**")
                        if model_name == 'logistic_regression':
                            st.info("🎯 **Logistic Regression**: Fast, interpretable, and excellent for this classification task. Best overall performance.")
                        elif model_name == 'knn':
                            st.info("🔍 **K-Nearest Neighbors**: Instance-based learning, good for local patterns but slower for large datasets.")
                        elif model_name == 'rnn':
                            st.info("🧠 **RNN (LSTM)**: Deep learning approach, captures sequential patterns but requires more training data.")
                        elif model_name == 'elm':
                            st.info("⚡ **Extreme Learning Machine**: Fast training, single hidden layer network with random weights.")
            
            # Overall comparison table
            st.markdown("### 📊 Model Comparison Summary")
            comparison_data = benchmark_report.get('model_comparison', [])
            if comparison_data:
                df_comparison = pd.DataFrame(comparison_data)
                df_comparison.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Test Samples']
                
                # Format numerical columns
                for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
                    df_comparison[col] = df_comparison[col].round(4)
                
                # Add ranking
                df_comparison['Rank'] = range(1, len(df_comparison) + 1)
                df_comparison = df_comparison[['Rank', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Test Samples']]
                
                st.dataframe(df_comparison, use_container_width=True)
                
                # Visualization of model performance
                if PLOTLY_AVAILABLE:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # F1-Score comparison
                        fig_f1 = px.bar(
                            df_comparison,
                            x='Model',
                            y='F1-Score',
                            title='📊 F1-Score Comparison',
                            color='F1-Score',
                            color_continuous_scale='viridis'
                        )
                        fig_f1.update_layout(height=400, xaxis_tickangle=-45)
                        st.plotly_chart(fig_f1, use_container_width=True)
                    
                    with col2:
                        # Accuracy comparison
                        fig_acc = px.bar(
                            df_comparison,
                            x='Model',
                            y='Accuracy',
                            title='🎯 Accuracy Comparison',
                            color='Accuracy',
                            color_continuous_scale='plasma'
                        )
                        fig_acc.update_layout(height=400, xaxis_tickangle=-45)
                        st.plotly_chart(fig_acc, use_container_width=True)
            
            # RNN Results (if available)
            model_results = benchmark_report.get('model_results', {})
            rnn_results = model_results.get('rnn')
            
            if rnn_results:
                st.markdown("### 🧠 RNN (LSTM) Detailed Results")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("RNN Accuracy", f"{rnn_results.get('accuracy', 0):.4f}")
                with col2:
                    st.metric("RNN F1-Score", f"{rnn_results.get('f1_weighted', 0):.4f}")
                with col3:
                    st.metric("RNN Precision", f"{rnn_results.get('precision_weighted', 0):.4f}")
                with col4:
                    st.metric("RNN Recall", f"{rnn_results.get('recall_weighted', 0):.4f}")
                
                # Training history if available
                training_history = rnn_results.get('training_history')
                if training_history:
                    st.markdown("**🔄 RNN Training History:**")
                    hist_col1, hist_col2 = st.columns(2)
                    with hist_col1:
                        st.metric("Final Training Loss", f"{training_history.get('final_loss', 0):.4f}")
                        st.metric("Final Training Accuracy", f"{training_history.get('final_accuracy', 0):.4f}")
                    with hist_col2:
                        st.metric("Final Validation Loss", f"{training_history.get('final_val_loss', 0):.4f}")
                        st.metric("Final Validation Accuracy", f"{training_history.get('final_val_accuracy', 0):.4f}")
            
            # Dataset information
            st.markdown("### 📊 Dataset Information")
            dataset_info = benchmark_report.get('dataset_info', {})
            
            info_col1, info_col2, info_col3, info_col4 = st.columns(4)
            with info_col1:
                st.metric("Total Samples", dataset_info.get('total_samples', 0))
            with info_col2:
                st.metric("Features", dataset_info.get('features', 0))
            with info_col3:
                st.metric("Classes", dataset_info.get('classes', 0))
            with info_col4:
                st.metric("Test Samples", dataset_info.get('test_samples', 0))
            
            # Class names
            class_names = dataset_info.get('class_names', [])
            if class_names:
                st.markdown("**📋 Expense Categories:**")
                st.write(", ".join(class_names))
            
            # Timestamp
            timestamp = benchmark_report.get('timestamp', 'Unknown')
            st.markdown(f"**⏰ Last Benchmark:** {timestamp}")
            
        else:
            st.warning("⚠️ No benchmark report found. Run model training to generate benchmark results.")
            
            if st.button("🚀 Run Model Benchmarking Now"):
                with st.spinner("Running comprehensive model evaluation... This may take a few minutes."):
                    try:
                        benchmark_report = evaluator.benchmark_all_models()
                        st.success("✅ Benchmarking completed! Refresh the page to see results.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Benchmarking failed: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

    with tab5:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader('Settings')
        
        # Quick demo data for testing graphs
        if 'current_user' in st.session_state:
            st.markdown("**🎯 Quick Test Data**")
            if st.button('Add Sample Expenses (for testing graphs)', key='add_sample'):
                cu = st.session_state['current_user']
                first = cu.get('first')
                last = cu.get('last')
                
                # Sample expenses with realistic data across different time periods
                import datetime
                base_date = datetime.datetime.now()
                
                sample_expenses = [
                    # Current month
                    {'merchant': 'Starbucks Coffee', 'category': 'food_dining', 'amt': 5.50, 'date_offset': 0},
                    {'merchant': 'Shell Gas Station', 'category': 'gas_transport', 'amt': 45.00, 'date_offset': -2},
                    {'merchant': 'Walmart Grocery', 'category': 'grocery_pos', 'amt': 85.30, 'date_offset': -5},
                    {'merchant': 'Netflix Subscription', 'category': 'entertainment', 'amt': 15.99, 'date_offset': -7},
                    
                    # Previous month
                    {'merchant': 'Amazon Purchase', 'category': 'shopping_net', 'amt': 29.99, 'date_offset': -35},
                    {'merchant': 'McDonalds Restaurant', 'category': 'food_dining', 'amt': 12.50, 'date_offset': -40},
                    {'merchant': 'Target Store', 'category': 'shopping_pos', 'amt': 67.25, 'date_offset': -45},
                    
                    # Two months ago
                    {'merchant': 'Planet Fitness Gym', 'category': 'health_fitness', 'amt': 22.99, 'date_offset': -65},
                    {'merchant': 'Costco Wholesale', 'category': 'grocery_pos', 'amt': 120.75, 'date_offset': -70},
                    {'merchant': 'Home Depot Store', 'category': 'home', 'amt': 89.50, 'date_offset': -75},
                    
                    # Three months ago
                    {'merchant': 'Uber Ride', 'category': 'gas_transport', 'amt': 18.25, 'date_offset': -95},
                    {'merchant': 'Chipotle Mexican', 'category': 'food_dining', 'amt': 11.75, 'date_offset': -100}
                ]
                
                for expense in sample_expenses:
                    # Calculate date based on offset
                    expense_date = base_date + datetime.timedelta(days=expense['date_offset'])
                    
                    row = {
                        'trans_date_trans_time': expense_date.strftime('%Y-%m-%d %H:%M:%S'),
                        'merchant': expense['merchant'],
                        'category': expense['category'],
                        'amt': expense['amt'],
                        'first': first,
                        'last': last
                    }
                    append_transaction(row)
                
                st.success('✅ Added 8 sample expenses! Go to Analysis tab to see the graphs.')
                st.rerun()
        
        st.markdown("---")
        
        # Model settings
        if st.button('Reload model artifacts'):
            success, message = reload_model_artifacts()
            if success:
                st.success(message)
            else:
                st.error(message)
        st.write('Model status: ', 'Loaded' if model is not None else 'Not loaded')
        
        # Database info
        st.markdown("**Database Information**")
        try:
            total_transactions = len(db_fetch_all(conn))
            total_users = len(db_list_users(conn))
            st.write(f"Total transactions: {total_transactions}")
            st.write(f"Total users: {total_users}")
        except Exception as e:
            st.error(f"Database error: {str(e)}")
        
        # Plotly status
        st.markdown("**Visualization Status**")
        st.write(f"Plotly available: {'✅ Yes' if PLOTLY_AVAILABLE else '❌ No'}")
        
        st.markdown("---")
        
        # Data Management Section
        st.markdown("**📊 Data Management**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Sync Database to CSV"):
                with st.spinner("Synchronizing database to CSV..."):
                    success = sync_database_to_csv(conn)
                    if success:
                        st.success("✅ Database synchronized to CSV successfully!")
                    else:
                        st.error("❌ Failed to sync database to CSV")
        
        with col2:
            if st.button("🔍 Validate Data Integrity"):
                integrity_report = validate_data_integrity(conn)
                if "error" in integrity_report:
                    st.error(f"❌ {integrity_report['error']}")
                else:
                    if integrity_report['in_sync']:
                        st.success(f"✅ Data is in sync! DB: {integrity_report['database_transactions']}, CSV: {integrity_report['csv_transactions']}")
                    else:
                        st.warning(f"⚠️ Data mismatch! DB: {integrity_report['database_transactions']}, CSV: {integrity_report['csv_transactions']}")
        
        # CSV Statistics
        st.markdown("**📈 CSV File Statistics**")
        csv_stats = get_data_stats()
        
        if "error" in csv_stats:
            st.error(f"❌ {csv_stats['error']}")
        else:
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            with stat_col1:
                st.metric("Total Transactions", csv_stats.get('total_transactions', 0))
                st.metric("Unique Users", csv_stats.get('unique_users', 0))
            with stat_col2:
                st.metric("Categories", csv_stats.get('categories', 0))
                st.metric("File Size (MB)", f"{csv_stats.get('file_size_mb', 0):.2f}")
            with stat_col3:
                st.metric("Total Amount", f"${csv_stats.get('total_amount', 0):.2f}")
                st.metric("Avg Amount", f"${csv_stats.get('avg_amount', 0):.2f}")
        
        st.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown('---')
st.sidebar.write('Model metadata and artifacts loaded from models/ directory')