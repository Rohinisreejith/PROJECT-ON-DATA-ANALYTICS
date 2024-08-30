import streamlit as st
import pandas as pd
import plotly.express as px


st.set_page_config(page_title="	🛍️Sales Dashboard🛍️", layout="wide")

st.title("🛍️Sales Dashboard🛍️")

df=pd.read_csv("sample_sales_data.csv")
st.header("🛒Sales Dataset🛒")
st.dataframe(df.head())



df['Date'] = pd.to_datetime(df['Date'])

s1=st.sidebar


# Date filter
start_date, end_date = s1.date_input("Select date range", [df['Date'].min(), df['Date'].max()])

# Region filter
regions = s1.multiselect("Select Region(s)", options=df['Region'].unique(), default=df['Region'].unique())

# Product filter
products = s1.multiselect("Select Product(s)", options=df['Product Name'].unique(), default=df['Product Name'].unique())

# Button to slice data
slice_button = s1.button("Slice Data")

# Filter data based on inputs
filtered_df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
filtered_df = filtered_df[filtered_df['Region'].isin(regions)]
filtered_df = filtered_df[filtered_df['Product Name'].isin(products)]

# Display filtered data
if slice_button:
    st.write(f"Data Sliced. Showing {len(filtered_df)} records.")
    st.write(filtered_df)
col1, col2 = st.columns(2)

# Sales Over Time (Bar Chart)
with col1:
    st.subheader("Sales Over Time")
    sales_fig = px.bar(filtered_df, x='Date', y='Sales Amount', color='Region', title="Sales Over Time")
    st.plotly_chart(sales_fig, use_container_width=True)

# Profit by Product (Bar Chart)
with col2:
    st.subheader("Units Sold by Product")
    profit_fig = px.bar(filtered_df, x='Product Name', y='Units Sold', color='Product Name', title="units sold vs Product")
    st.plotly_chart(profit_fig, use_container_width=True)

col3, col4 = st.columns(2)

# Sales Trend Over Time (Line Chart)
with col3:
    st.subheader("Sales Trend Over Time")
    trend_fig = px.line(filtered_df, x='Date', y='Sales Amount', color='Region', title="Sales Trend Over Time")
    st.plotly_chart(trend_fig, use_container_width=True)

# Regional Sales Distribution (Pie Chart)
with col4:
    st.subheader("Regional Sales Distribution")
    pie_fig = px.pie(filtered_df, names='Region', values='Sales Amount', title="Regional Sales Distribution")
    st.plotly_chart(pie_fig, use_container_width=True)

# Additional summary
st.subheader("Summary")
total_sales = filtered_df['Sales Amount'].sum()

st.metric("Total Sales", f"${total_sales}")
