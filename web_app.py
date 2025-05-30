import streamlit as st
import pandas as pd
import numpy as np
import base64
from streamlit_option_menu import option_menu
import os
from warehouse_management import run_warehouse_management  # Import your warehouse management function
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import io
import random
import gym
from gym import spaces
import joblib
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="OptiFow AI", page_icon="🚚")
from contextlib import contextmanager

@contextmanager
def loading_screen(message="Loading..."):
    """Context manager to show loading screen during operations"""
    loading_placeholder = st.empty()
    try:
        with loading_placeholder.container():
            st.markdown(f"""
            <style>
                .loading-container {{
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background-color: rgba(0, 0, 0, 0.8);
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    z-index: 9999;
                }}
                .loading-spinner {{
                    border: 5px solid #f3f3f3;
                    border-top: 5px solid #00ffff;
                    border-radius: 50%;
                    width: 50px;
                    height: 50px;
                    animation: spin 1s linear infinite;
                }}
                @keyframes spin {{
                    0% {{ transform: rotate(0deg); }}
                    100% {{ transform: rotate(360deg); }}
                }}
                .loading-text {{
                    color: white;
                    margin-top: 20px;
                    font-size: 18px;
                }}
            </style>
            <div class="loading-container">
                <div class="loading-spinner"></div>
                <div class="loading-text">{message}</div>
            </div>
            """, unsafe_allow_html=True)
        yield
    finally:
        loading_placeholder.empty()
# Function to encode image as base64
def get_base64_from_file(file_path):
    try:
        with open(file_path, "rb") as file:
            return base64.b64encode(file.read()).decode("utf-8")
    except FileNotFoundError:
        st.error("Background image file not found. Please check the file path.")
        return None

# Path to the background image
background_image = "images/warehouse.jpg"  # Replace with your image file path
base64_image = get_base64_from_file(background_image)


# Register the custom metric if needed
@tf.keras.utils.register_keras_serializable()
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Function to save DataFrame to Excel
def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=True, sheet_name="Forecast")
    output.seek(0)
    return output

# Apply custom CSS if the image is valid

if base64_image:
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{base64_image}");
        background-size: 100% 100%;
        background-repeat: no-repeat;
        background-position: center;
        background-color: rgba(0, 0, 0, 0.4);
        background-blend-mode: overlay; 
    }}
    header {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .content-box {{
        background-color: rgba(0, 0, 0, 0.8);
        color: white;
        font-size: 18px;
        text-align: justify;
        padding: 10px;
        border-radius: 10px;
        margin: 10px auto;
        width: 100%;
    }}
    .content-box h2 {{
        text-align: center;
        font-size: 28px;
        margin-bottom: 20px;
    }}
    .stButton > button {{
        background-color: rgba(0, 0, 0, 0.6);
        color: white;
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 20%;
        border: 2px solid transparent;
        border-radius: 5px;
        padding: 10px;
        font-size: 16px;
        cursor: pointer;
        transition: transform 0.2s ease-in-out, color 0.3s, border-color 0.3s;
        position: relative;
        overflow: hidden;
    }}
    .stButton > button::after {{
        content: '';
        background-color: rgba(0, 255, 255, 0.6);
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        transform: translateX(-100%) rotate(10deg);
        transform-origin: top left;
        transition: transform 0.2s ease-out;
        z-index: -1;
    }}
    .stButton > button:hover {{
        border-color: transparent;
        color: black;
        background-color: rgba(0, 255, 255, 0.6);  /* Transparent white on hover */
        transform: scale(1.05);
    }}
    .stButton > button:hover::after {{
        transform: translateX(0) rotate(0deg);
    }}
    .stButton > button:active {{
        color: black;
        border-color: black;
        background-color: rgba(0, 0, 0, 0.6);
    }}
    .stSidebar {{
        background-color: rgba(0, 0, 0, 1);
        color: white;
        font-size: 18px;
        text-align: left;
        padding: 20px;
    }}
    </style>
    """, unsafe_allow_html=True)

# Initialize session state for button click
if "explore_clicked" not in st.session_state:
    st.session_state.explore_clicked = False

# Page Content
if not st.session_state.explore_clicked:
    st.markdown(
        """
        <div class="content-box">
            <h2><b>👷🏼 Welcome to OptiFlow AI</b></h2>
            <p><b>OptiFlow AI is designed to help improve warehouse operations by making tasks more efficient and reducing costs. It achieves this by focusing on key areas such as planning ahead, maintaining stock levels, better organization, and improving customer satisfaction 🎯.</b></p>
            <p><b>The app predicts which products are likely to be needed soon and when returns may occur, helping warehouses stay ahead of demand. It also ensures that stock is replenished in a timely manner, which helps avoid delays and reduce unnecessary labor costs 💰. Additionally, it improves warehouse organization by strategically placing items in easy-to-reach locations based on how often they are picked, making it easier for staff to locate products quickly.</b></p>
            <p><b>OptiFlow AI helps to identify customers who might stop buying, allowing businesses to take action and keep their customers happy and loyal. This overall approach leads to smoother warehouse operations, better customer service, and lower costs 🚀.</b></p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Explore"):
        st.session_state.explore_clicked = True
        st.rerun()

else:
    with st.sidebar:
        selected = option_menu(
            menu_title="OptiFlow AI - Intelligent Supply Chain Management System",
            options=[
                "Home",
                "Demand Forecasting",
                "Return Forecasting",
                "Inventory Management",
                "Warehouse Management",
                "Customer Churn Prediction"
            ],
            icons=[
                "house-fill",       
                "graph-up-arrow", 
                "graph-up-arrow",    
                "boxes",             
                "building",         
                "person-circle"      
            ],
            default_index=0,
            menu_icon="robot",
            styles={
                "container": {
                    "background-color": "rgba(0, 0, 0, 1)",
                },
                "nav-link": {
                    "color": "white",
                    "font-size": "18px"
                },
                "nav-link-selected": {
                    "background-color": "rgba(0, 255, 255, 0.8)"
                },
                "menu-title": {
                    "color": "white",
                    "font-size": "22px",
                    "font-weight": "bold"
                },
            }
        )

    # Main content area
    if selected == "Home":
        # Check if the GIF file exists
        gif_path = "GIFS/Hello Animation.gif"
        if os.path.exists(gif_path):
            st.markdown(
                f"""
                <div class="content-box" style="
                    background-color: rgba(0, 0, 0, 0.8);
                    color: white;
                    border-radius: 10px;
                    padding: 20px;
                    margin: 20px auto;
                    max-width: 900px;
                    display: flex;
                    flex-direction: column;
                    gap: 20px;
                ">
                    <!-- Title Section -->
                    <h1 style="font-size: 32px; margin-bottom: 10px; text-align: center;">
                        <b>OptiFlow AI - Revolutionizing the Modern Supply Chain!</b>
                    </h1>
                    <div style="display: flex; gap: 20px; align-items: center;">
                        <div style="flex: 1; text-align: justify; font-size: 18px; line-height: 1.6;">
                            <p>
                                <b>OptiFlow AI</b> is an AI-powered supply chain management system designed to optimize warehouse operations. 
                                Streamline processes, minimize costs, and enhance customer satisfaction with our intelligent features
                            </p>
                        </div>
                        <div style="flex: 0.2; text-align: center;">
                            <img src="data:image/gif;base64,{get_base64_from_file(gif_path)}" 
                                alt="Optimizing Your Warehouse Operations" 
                                style="width: 120%; max-width: 300px; border-radius: 10px;">
                        </div>
                    </div>
                    <div style="text-align: left; font-size: 18px; line-height: 1.6;">
                        <ul style="list-style-type: none; padding: 0; margin: 0;">
                            <li style="margin: 10px 0;">🚀 <b>Demand and Return Prediction:</b> Minimize waste and improve planning.</li>
                            <li style="margin: 10px 0;">📦 <b>Inventory Management:</b> Reduce costs and prioritize replenishments.</li>
                            <li style="margin: 10px 0;">🏗 <b>Warehouse Organization:</b> Efficiently manage stock placement using AI.</li>
                            <li style="margin: 10px 0;">🔗 <b>Customer Retention:</b> Identify and engage at-risk customers.</li>
                        </ul>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("GIF file not found. Please check the file path.")


    elif selected == "Warehouse Management":
        with loading_screen("Loading warehouse data..."):
            # Hide the sidebar and remove the background image for this section
            st.markdown(
                """
                <style>
                .stApp {{
                    background-image: none !important;
                    background-color: white !important;
                }}
                .stSidebar {{
                    display: none !important;
                }}
                .stButton > button {{
                    background-color: rgba(0, 0, 0, 0.6);
                    color: white;
                    display: block;
                    margin-left: auto;
                    margin-right: auto;
                    width: 20%;
                    border: 2px solid transparent;
                    border-radius: 5px;
                    padding: 10px;
                    font-size: 16px;
                    cursor: pointer;
                    transition: transform 0.2s ease-in-out, color 0.3s, border-color 0.3s;
                    position: relative;
                    overflow: hidden;
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )
            # Run the warehouse management code
            run_warehouse_management()


    elif selected == "Demand Forecasting":
            with loading_screen("Loading demand forecasting data..."):

                st.markdown(
                    """
                    <div class="content-box">
                        <h2>Demand Forecasting</h2>
                        <p>Upload your transaction data now and unlock AI-powered demand forecasting for the next 7 days!</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("""
                    <style>
                        div.stFileUploader {
                            background-color: rgba(0, 0, 0, 0.8); /
                            border-radius: 10px;
                            padding: 10px;
                        }
                        div.stFileUploader > label {
                            color: white;
                            font-weight: bold;
                        }
                    </style>
                """, unsafe_allow_html=True)
                file1 = st.file_uploader("Upload 'KTI Transactions 1.xlsx'", type="xlsx")
                file2 = st.file_uploader("Upload 'KTI Transactions 2.xlsx'", type="xlsx")
                if file1 and file2:
                    # Load the data
                    df1 = pd.read_excel(file1)
                    df2 = pd.read_excel(file2)

                    # Combine DataFrames
                    combined_df = pd.concat([df1, df2], ignore_index=True)

                    # Data Preprocessing for transactions
                    columns_to_drop = ['Owner', 'Lot', 'User', 'PACK KEY:', 'Source Key', 'ASN NO:', 'OWNER:']
                    df = combined_df.drop(columns=columns_to_drop, errors='ignore').dropna().drop_duplicates()
                    df['Date'] = pd.to_datetime(df['Date']).dt.date
                    df['Quantity'] = pd.to_numeric(df['Quantity'].str.replace(',', ''), errors='coerce').abs().astype(int)
                    df = df.dropna()

                    # Filter shipment data
                    shipment_data = df[
                        (df['Activity'] == 'Shipment') &
                        (df['Source Type'].isin(['ntrPickDetailUpdate', 'ntrTransferDetailAdd']))
                    ]
                    shipment_data['Date'] = pd.to_datetime(shipment_data['Date'])

                    # Aggregate daily demand per item
                    daily_item_demand = shipment_data.groupby(['Date', 'Item'])['Quantity'].sum().reset_index()

                    # Complete date range
                    date_range = pd.date_range(start=daily_item_demand['Date'].min(), end=daily_item_demand['Date'].max())
                    complete_item_demand = pd.DataFrame({
                        'Date': np.tile(date_range, len(daily_item_demand['Item'].unique())),
                        'Item': np.repeat(daily_item_demand['Item'].unique(), len(date_range))
                    })
                    merged_data = pd.merge(complete_item_demand, daily_item_demand, on=['Date', 'Item'], how='left')
                    merged_data['Quantity'].fillna(0, inplace=True)
                    pivoted_data = merged_data.pivot(index='Date', columns='Item', values='Quantity').fillna(0)

                    st.markdown(
                    """
                    <div class="content-box">
                        <h2>Uploaded Data Preview</h2>
                        
                    </div>
                    """,
                    unsafe_allow_html=True,
                    )
                    st.dataframe(df.head())
                    st.markdown(
                    """
                    <div class="content-box">
                        <h2>Processed Demand for each Item by Date</h2>
                        
                    </div>
                    """,
                    unsafe_allow_html=True,
                    )
                    st.dataframe(pivoted_data.head())
                    # Normalize data
                    scaler = MinMaxScaler()
                    normalized_data = scaler.fit_transform(pivoted_data)

                    # LSTM Model and Forecast
                    sequence_length = 30
                    forecast_horizon = 7

                    def create_sequences(data, sequence_length):
                        X = []
                        for i in range(len(data) - sequence_length):
                            X.append(data[i : i + sequence_length])
                        return np.array(X)

                    # Prepare last sequence for forecasting
                    last_sequence = normalized_data[-sequence_length:]
                    last_sequence = np.expand_dims(last_sequence, axis=0)

                    # Load pre-trained model
                    try:
                        model = tf.keras.models.load_model("forecast_model.h5", custom_objects={"mse": mse})
                        # Forecast
                        forecast_normalized = model.predict(last_sequence)
                        forecast_denormalized = scaler.inverse_transform(forecast_normalized[0])

                        # Round the forecasted demand to the nearest whole number and make values positive
                        forecast_denormalized = np.abs(forecast_denormalized).round()

                        # Create forecast DataFrame
                        last_date = pivoted_data.index[-1]
                        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon)
                        forecast_df = pd.DataFrame(forecast_denormalized, index=forecast_dates, columns=pivoted_data.columns)
                        st.markdown(
                                    """
                                    <div class="content-box">
                                        <h2>Forecasted Demand for Next 7 Days per each Item</h2>
                                        
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                        )
                        st.dataframe(forecast_df)

                        # Plot forecast
                        st.markdown(
                                    """
                                    <div class="content-box">
                                        <h2>Forecast Visualization by Item</h2>
                                        <p>Select Item to View Forecast</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                        )
                        selected_item = st.selectbox("Select Item to View Forecast", forecast_df.columns)

                        plt.figure(figsize=(10, 5))
                        plt.plot(forecast_df.index, forecast_df[selected_item], marker='o', label=f"Forecast for {selected_item}")
                        plt.title(f"Forecasted Demand for {selected_item}")
                        plt.xlabel("Date")
                        plt.ylabel("Demand")
                        plt.xticks(rotation=45)
                        plt.legend()
                        st.pyplot(plt)
                        st.markdown(
                                            """
                                            <style>
                                                .stDownloadButton {
                                                    display: flex;
                                                    justify-content: center;
                                                    align-items: center;
                                                }
                                                .stDownloadButton button {
                                                    background-color: rgba(0, 0, 0, 0.8);
                                                    color: white !important;
                                                    border-radius: 5px !important;
                                                    border: none !important;
                                                    padding: 10px !important;
                                                }
                                            </style>
                                            """,
                                            unsafe_allow_html=True,
                        )
                        # Provide download button for the forecast data
                        excel_file = to_excel(forecast_df)
                        st.download_button(
                            label="Download Forecast Data as Excel",
                            data=excel_file,
                            file_name="forecasted_demand.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    except Exception as e:
                            st.error(f"Error loading the model: {e}")
                    
                    # Load stock report file
                    stock_report_file = st.file_uploader("Upload Stock Report", type="xlsx")

                    # If stock report file is uploaded
                    if stock_report_file:
                        # Load stock report data
                        stock_report = pd.read_excel(stock_report_file)

                        # Display unique materials for selection
                        unique_materials = stock_report['Material'].unique()
                        st.markdown(
                                    """
                                    <div class="content-box">
                                        <p>Select Materials</p>
                                        
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                        )
                        selected_materials = st.multiselect("Select Materials", unique_materials)

                        if selected_materials:
                            # Filter stock report data based on selected materials
                            selected_stock_data = stock_report[stock_report['Material'].isin(selected_materials)]

                            # Display the relevant stock data
                            st.markdown(
                                    """
                                    <div class="content-box">
                                        <h2>Selected Materials Stock Report</h2>
                                        
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                            )
                            st.dataframe(selected_stock_data)

                            # Forecast output for selected stock report materials
                            stock_forecast_data = forecast_df[selected_materials]
                            stock_excel_file = to_excel(stock_forecast_data)
                            st.markdown(
                                            """
                                            <style>
                                                .stDownloadButton {
                                                    display: flex;
                                                    justify-content: center;
                                                    align-items: center;
                                                }
                                                .stDownloadButton button {
                                                    background-color: rgba(0, 0, 0, 0.8);
                                                    color: white !important;
                                                    border-radius: 5px !important;
                                                    border: none !important;
                                                    padding: 10px !important;
                                                }
                                            </style>
                                            """,
                                            unsafe_allow_html=True,
                            )
                            # Pro  vide download button for stock forecast data
                            st.download_button(
                                label="Download Stock Forecast Data as Excel",
                                data=stock_excel_file,
                                file_name="selected_stock_forecast.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )

                            # Plot individual material forecast
                            for material in selected_materials:
                                st.markdown(
                                """
                                <div class="content-box">
                                    <h2>Demand Forecast for selected materials</h2>
                                    
                                </div>
                                """,
                                unsafe_allow_html=True,
                                )
                                if material in forecast_df.columns:
                                    plt.figure(figsize=(10, 5))
                                    plt.plot(forecast_df.index, forecast_df[material], marker='o', label=f"Forecast for {material}")
                                    plt.title(f"Forecasted Demand for {material}")
                                    plt.xlabel("Date")
                                    plt.ylabel("Demand")
                                    plt.xticks(rotation=45)
                                    plt.legend()
                                    st.pyplot(plt)
                                else:
                                    st.warning(f"No forecast data available for material: {material}")

    elif selected == "Return Forecasting":
        with loading_screen("Loading return forecasting data..."):
    
            st.markdown(
                """
                <div class="content-box">
                    <h2>Return Forecasting</h2>
                    <p>Upload your return data now and gain AI-driven insights to forecast product returns for the next 7 days!</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("""
                <style>
                    div.stFileUploader {
                        background-color: rgba(0, 0, 0, 0.8); /
                        border-radius: 10px;
                        padding: 10px;
                    }
                    div.stFileUploader > label {
                        color: white;
                        font-weight: bold;
                    }
                </style>
            """, unsafe_allow_html=True)
            file1 = st.file_uploader("Upload 'KTI Transactions 1.xlsx'", type="xlsx")
            file2 = st.file_uploader("Upload 'KTI Transactions 2.xlsx'", type="xlsx")
            if file1 and file2:
                # Load the data
                df1 = pd.read_excel(file1)
                df2 = pd.read_excel(file2)

                # Combine DataFrames
                combined_df = pd.concat([df1, df2], ignore_index=True)

                # Data Preprocessing for transactions
                columns_to_drop = ['Owner', 'Lot', 'User', 'PACK KEY:', 'Source Key', 'ASN NO:', 'OWNER:']
                df = combined_df.drop(columns=columns_to_drop, errors='ignore').dropna().drop_duplicates()
                df['Date'] = pd.to_datetime(df['Date']).dt.date
                df['Quantity'] = pd.to_numeric(df['Quantity'].str.replace(',', ''), errors='coerce').abs().astype(int)
                df = df.dropna()

                # Filter shipment data
                shipment_data = df[
                    (df['Activity'] == 'Receipt') &
                    (df['To Location'] == 'RETURN')
                ]
                shipment_data['Date'] = pd.to_datetime(shipment_data['Date'])

                # Aggregate daily demand per item
                daily_item_demand = shipment_data.groupby(['Date', 'Item'])['Quantity'].sum().reset_index()

                # Complete date range
                date_range = pd.date_range(start=daily_item_demand['Date'].min(), end=daily_item_demand['Date'].max())
                complete_item_demand = pd.DataFrame({
                    'Date': np.tile(date_range, len(daily_item_demand['Item'].unique())),
                    'Item': np.repeat(daily_item_demand['Item'].unique(), len(date_range))
                })
                merged_data = pd.merge(complete_item_demand, daily_item_demand, on=['Date', 'Item'], how='left')
                merged_data['Quantity'].fillna(0, inplace=True)
                pivoted_data = merged_data.pivot(index='Date', columns='Item', values='Quantity').fillna(0)

                st.markdown(
                """
                <div class="content-box">
                    <h2>Processed Returns for each Item by Date</h2>
                    
                </div>
                """,
                unsafe_allow_html=True,
                )
                st.dataframe(pivoted_data.head())
                # Normalize data
                scaler = MinMaxScaler()
                normalized_data = scaler.fit_transform(pivoted_data)

                # LSTM Model and Forecast
                sequence_length = 30
                forecast_horizon = 7

                def create_sequences(data, sequence_length):
                    X = []
                    for i in range(len(data) - sequence_length):
                        X.append(data[i : i + sequence_length])
                    return np.array(X)

                # Prepare last sequence for forecasting
                last_sequence = normalized_data[-sequence_length:]
                last_sequence = np.expand_dims(last_sequence, axis=0)

                # Load pre-trained model
                try:
                    model = tf.keras.models.load_model("return_forecast_model.h5", custom_objects={"mse": mse})
                    # Forecast
                    forecast_normalized = model.predict(last_sequence)
                    forecast_denormalized = scaler.inverse_transform(forecast_normalized[0])

                    # Round the forecasted demand to the nearest whole number and make values positive
                    forecast_denormalized = np.abs(forecast_denormalized).round()

                    # Create forecast DataFrame
                    last_date = pivoted_data.index[-1]
                    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon)
                    forecast_df = pd.DataFrame(forecast_denormalized, index=forecast_dates, columns=pivoted_data.columns)

                    st.markdown(
                        """
                        <div class="content-box">
                            <h2>Forecasted Returns for Next 7 Days per each Item</h2>
                            
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.dataframe(forecast_df)

                    # Plot forecast
                    st.markdown(
                        """
                        <div class="content-box">
                            <h2>Forecast Visualization by Item</h2>
                            <p>Select Item to View Forecast</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    selected_item = st.selectbox("Select Item to View Forecast", forecast_df.columns)

                    plt.figure(figsize=(10, 5))
                    plt.plot(forecast_df.index, forecast_df[selected_item], marker='o', label=f"Forecast for {selected_item}")
                    plt.title(f"Forecasted Return for {selected_item}")
                    plt.xlabel("Date")
                    plt.ylabel("Return")
                    plt.xticks(rotation=45)
                    plt.legend()
                    st.pyplot(plt)
                    st.markdown(
                                        """
                                        <style>
                                            .stDownloadButton {
                                                display: flex;
                                                justify-content: center;
                                                align-items: center;
                                            }
                                            .stDownloadButton button {
                                                background-color: rgba(0, 0, 0, 0.8);
                                                color: white !important;
                                                border-radius: 5px !important;
                                                border: none !important;
                                                padding: 10px !important;
                                            }
                                        </style>
                                        """,
                                        unsafe_allow_html=True,
                    )
                    # Provide download button for the forecast data
                    excel_file = to_excel(forecast_df)
                    st.download_button(
                        label="Download Forecast Data as Excel",
                        data=excel_file,
                        file_name="forecasted_return.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except Exception as e:
                        st.error(f"Error loading the model: {e}")
                
                # Load stock report file
                stock_report_file = st.file_uploader("Upload Stock Report", type="xlsx")

                # If stock report file is uploaded
                if stock_report_file:
                    # Load stock report data
                    stock_report = pd.read_excel(stock_report_file)

                    # Display unique materials for selection
                    unique_materials = stock_report['Material'].unique()
                    st.markdown(
                                """
                                <div class="content-box">
                                    <p>Select Materials</p>
                                    
                                </div>
                                """,
                                unsafe_allow_html=True,
                    )
                    selected_materials = st.multiselect("Select Materials", unique_materials)

                    if selected_materials:
                        # Filter stock report data based on selected materials
                        selected_stock_data = stock_report[stock_report['Material'].isin(selected_materials)]

                        # Display the relevant stock data
                        st.markdown(
                        """
                        <div class="content-box">
                            <h2>Selected Materials Stock Report</h2>
                            
                        </div>
                        """,
                        unsafe_allow_html=True,
                        )
                        st.dataframe(selected_stock_data)

                        # Forecast output for selected stock report materials
                        stock_forecast_data = forecast_df[selected_materials]
                        stock_excel_file = to_excel(stock_forecast_data)
                        st.markdown(
                                        """
                                        <style>
                                            .stDownloadButton {
                                                display: flex;
                                                justify-content: center;
                                                align-items: center;
                                            }
                                            .stDownloadButton button {
                                                background-color: rgba(0, 0, 0, 0.8);
                                                color: white !important;
                                                border-radius: 5px !important;
                                                border: none !important;
                                                padding: 10px !important;
                                            }
                                        </style>
                                        """,
                                        unsafe_allow_html=True,
                        )
                        # Provide download button for stock forecast data
                        st.download_button(
                            label="Download Stock Forecast Data as Excel",
                            data=stock_excel_file,
                            file_name="selected_stock_forecast.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                        # Plot individual material forecast
                        for material in selected_materials:
                            st.markdown(
                            """
                            <div class="content-box">
                                <h2>Return Forecast for selected materials</h2>
                                
                            </div>
                            """,
                            unsafe_allow_html=True,
                            )
                            if material in forecast_df.columns:
                                plt.figure(figsize=(10, 5))
                                plt.plot(forecast_df.index, forecast_df[material], marker='o', label=f"Forecast for {material}")
                                plt.title(f"Forecasted Return for {material}")
                                plt.xlabel("Date")
                                plt.ylabel("Return")
                                plt.xticks(rotation=45)
                                plt.legend()
                                st.pyplot(plt)
                            else:
                                st.warning(f"No forecast data available for material: {material}")

    elif selected == "Inventory Management":
        with loading_screen("Loading inventory management data..."):
            st.markdown(
                """
                <div class="content-box">
                    <h2>Inventory Management</h2>
                    <p>Upload the current stock report and generated forecast report to dynamically prioritize replenishments.</p>
                    <div id="success-message" style="display: none; color: #FFFFFF; font-weight: bold;">
                    Replenishment Plan Generated!
                    </div>
                    <style>
                    .stButton button {
                        width: auto !important;
                        white-space: nowrap;
                    }
                    </style>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <style>
                    div.stFileUploader {
                        background-color: rgba(0, 0, 0, 0.8);
                        border-radius: 10px;
                        padding: 10px;
                    }
                    div.stFileUploader > label {
                        color: white;
                        font-weight: bold;
                    }
                    .content-box {
                        padding: 15px;
                        border: 1px solid #ddd;
                        border-radius: 10px;
                        background-color: rgba(0, 0, 0, 0.8);
                    }
                </style>
                """,
                unsafe_allow_html=True,
            )

            # Streamlit UI for uploading files
            stock_file = st.file_uploader("Upload Stock Report (CSV)", type="csv")
            forecast_file = st.file_uploader("Upload Demand Forecast Report (Excel)", type="xlsx")

            if stock_file and forecast_file:
                stock_data = pd.read_csv(stock_file)
                forecasted_demand = pd.read_excel(forecast_file, sheet_name=0, index_col=0)

                stock_data['On Hand'] = pd.to_numeric(stock_data['On Hand'], errors='coerce').fillna(0)

                if st.button("Generate Replenishment Plan"):
                    class ReplenishmentEnv(gym.Env):
                        def __init__(self, forecast, stock_mapping):
                            super(ReplenishmentEnv, self).__init__()
                            self.forecast = forecast
                            self.stock_mapping = stock_mapping
                            self.items = list(stock_mapping.keys())
                            self.action_space = spaces.Discrete(len(self.items))
                            self.observation_space = spaces.Box(
                                low=0, high=np.inf, shape=(len(self.items),), dtype=np.float32
                            )
                            self.reset()

                        def reset(self):
                            self.remaining_demand = self.forecast.sum(axis=0).to_dict()
                            self.stock_levels = {
                                item: locations["high_rack"]["On Hand"].sum()
                                for item, locations in self.stock_mapping.items()
                            }
                            return np.array(list(self.remaining_demand.values()), dtype=np.float32)

                        def step(self, action):
                            item = self.items[action]
                            if item not in self.remaining_demand or self.remaining_demand[item] <= 0:
                                return self._get_obs(), -10, True, {}

                            demand = self.remaining_demand[item]
                            stock = self.stock_levels.get(item, 0)

                            if demand == 0:
                                replenishment_quantity = 0
                                reward = 0
                            else:
                                replenishment_quantity = min(demand, stock)
                                if replenishment_quantity > 0:
                                    self.remaining_demand[item] -= replenishment_quantity
                                    self.stock_levels[item] -= replenishment_quantity
                                    reward = replenishment_quantity
                                else:
                                    reward = -5

                            done = all(v <= 0 for v in self.remaining_demand.values())
                            return self._get_obs(), reward, done, {}

                        def _get_obs(self):
                            return np.array(list(self.remaining_demand.values()), dtype=np.float32)

                    def create_mapping(high_rack, pick_piece):
                        mapping = {}
                        for item in high_rack["Material"].unique():
                            high_rack_locations = high_rack[high_rack["Material"] == item]
                            pick_piece_locations = pick_piece[pick_piece["Material"] == item]
                            if not high_rack_locations.empty and not pick_piece_locations.empty:
                                mapping[item] = {
                                    "high_rack": high_rack_locations,
                                    "pick_piece": pick_piece_locations.iloc[0]["Location"]
                                }
                        return mapping

                    def train_model(stock_data, forecasted_demand):
                        high_rack_stock = stock_data[stock_data["Location Type (Not Location)"] == "High Rack"]
                        pick_piece_stock = stock_data[stock_data["Location Type (Not Location)"] == "PICK - Piece"]
                        mapped_locations = create_mapping(high_rack_stock, pick_piece_stock)

                        env = ReplenishmentEnv(forecasted_demand, mapped_locations)
                        q_table = {}
                        learning_rate = 0.1
                        discount_factor = 0.9
                        epsilon = 1.0
                        min_epsilon = 0.01
                        epsilon_decay = 0.995

                        num_episodes = 1000
                        for episode in range(num_episodes):
                            state = env.reset()
                            discrete_state = tuple(state.astype(int))
                            done = False
                            while not done:
                                if random.uniform(0, 1) < epsilon:
                                    action = env.action_space.sample()
                                else:
                                    action = np.argmax(q_table.get(discrete_state, np.zeros(env.action_space.n)))

                                next_state, reward, done, _ = env.step(action)
                                discrete_next_state = tuple(next_state.astype(int))

                                if discrete_state not in q_table:
                                    q_table[discrete_state] = np.zeros(env.action_space.n)
                                if discrete_next_state not in q_table:
                                    q_table[discrete_next_state] = np.zeros(env.action_space.n)

                                q_table[discrete_state][action] = (
                                    (1 - learning_rate) * q_table[discrete_state][action] +
                                    learning_rate * (reward + discount_factor * np.max(q_table[discrete_next_state]))
                                )

                                discrete_state = discrete_next_state

                            epsilon = max(min_epsilon, epsilon * epsilon_decay)

                        replenishment_plan = []
                        for item in env.items:
                            high_rack_location = mapped_locations[item]["high_rack"].iloc[0]["Location"]
                            pick_piece_location = mapped_locations[item]["pick_piece"]
                            replenishment_plan.append({
                                "item": item,
                                "from_location": high_rack_location,
                                "to_location": pick_piece_location,
                                "quantity": env.stock_levels[item]
                            })

                        replenishment_plan_df = pd.DataFrame(replenishment_plan)
                        replenishment_plan_df = replenishment_plan_df.sort_values(by="quantity", ascending=False)

                        return replenishment_plan_df

                    replenishment_plan = train_model(stock_data, forecasted_demand)

                    st.markdown(
                        """
                        <div class="content-box">
                            <h4 style="color: white;">Replenishment Plan Generated!</h4>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.dataframe(replenishment_plan)
                    st.markdown(
                        """
                        <style>
                            .stDownloadButton {
                                display: flex;
                                justify-content: center;
                                align-items: center;
                            }
                            .stDownloadButton button {
                                background-color: rgba(0, 0, 0, 0.8);
                                color: white !important;
                                border-radius: 5px !important;
                                border: none !important;
                                padding: 10px !important;
                            }
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )
                    replenishment_plan_file = "replenishment_plan.xlsx"
                    replenishment_plan.to_excel(replenishment_plan_file, index=False)
                    with open(replenishment_plan_file, "rb") as file:
                        st.download_button(
                            label="Download Replenishment Plan",
                            data=file,
                            file_name=replenishment_plan_file,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    
                    # Visualization: Top 10 Stock Availability
                    stock_summary = stock_data.groupby('Material')['On Hand'].sum().reset_index()
                    top_stock_summary = stock_summary.nlargest(10, 'On Hand')
                    st.markdown(
                                """
                                <div class="content-box">
                                    <h2>Stock Availability by Material (Top 10)</h2>
                                    
                                </div>
                                """,
                                unsafe_allow_html=True,
                    )
                    fig1, ax1 = plt.subplots(figsize=(12, 6))
                    sns.barplot(data=top_stock_summary, x='Material', y='On Hand', ax=ax1)
                    plt.xticks(rotation=45)
                    st.pyplot(fig1)

                    # Visualization: Top 10 Forecasted Demand
                    forecast_summary = forecasted_demand.sum(axis=0).reset_index()
                    forecast_summary.columns = ['Material', 'Total Demand']
                    top_forecast_summary = forecast_summary.nlargest(10, 'Total Demand')
                    st.markdown(
                                """
                                <div class="content-box">
                                    <h2>Forecasted Demand by Material (Top 10)</h2>
                                    
                                </div>
                                """,
                                unsafe_allow_html=True,
                    )
                    fig2, ax2 = plt.subplots(figsize=(12, 6))
                    sns.barplot(data=top_forecast_summary, x='Material', y='Total Demand', ax=ax2)
                    plt.xticks(rotation=45)
                    st.pyplot(fig2)

                    # Visualization: Top 5 Replenishment Priorities
                    top_priorities = replenishment_plan.nlargest(5, 'quantity')
                    st.markdown(
                                """
                                <div class="content-box">
                                    <h2>Top 5 Replenishment Priorities</h2>
                                    
                                </div>
                                """,
                                unsafe_allow_html=True,
                    )
                    fig3, ax3 = plt.subplots(figsize=(12, 6))
                    sns.barplot(data=top_priorities, x='item', y='quantity', ax=ax3)
                    plt.xticks(rotation=45)
                    st.pyplot(fig3)


    elif selected == "Customer Churn Prediction":
        with loading_screen("Loading customer churn data..."):
            st.markdown(
                """
                <style>
                .churn-container {
                    background-color: rgba(0, 0, 0, 0.8); /* Dark semi-transparent background */
                    padding: 20px;
                    border-radius: 10px;
                    margin: 20px 0;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            try:
                # Load pre-trained models and scaler
                rf_model = joblib.load("Churn_Models/random_forest_model.pkl")
                nn_model = tf.keras.models.load_model(
                    "Churn_Models/neural_network_model.h5")
                scaler = joblib.load("Churn_Models/scaler.pkl")
            except Exception as e:
                st.error(f"Error loading models: {e}")
                st.stop()

            # Load dataset for company filtering
            try:
                df = pd.read_csv("ShipmentDetailsKTI.csv")
                df["DeliveryDelay"] = (pd.to_datetime(
                    df["ActualShipDate"]) - pd.to_datetime(df["RequestedDeliveryDate"])).dt.days
            except Exception as e:
                st.error(f"Error loading dataset: {e}")
                st.stop()

            # Create tabs for individual and company-level predictions
            tab1, tab2 = st.tabs(
                ["Individual Prediction", "Company-Level Prediction"])

            with tab1:
                # Individual Prediction Section
                with st.container():
                    st.markdown("""<div class="content-box">
                    <h2>Individual Customer Churn Prediction</h2>
                    <p>Predict customer churn based on shipment details and delivery performance.</p>
                    <p>Predict churn for a single customer based on shipment details.</p>
                </div>""", unsafe_allow_html=True)
                    st.markdown("""<div class="content-box">
                    <p>Actual Ship Date &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Requested Delivery Date</p>
                    </div>""", unsafe_allow_html=True)
                    # Create two columns for date inputs
                    col1, col2 = st.columns(2)
                    with col1:
                        actual_ship_date = st.date_input(
                            "Actual Ship Date", value=pd.to_datetime("today"))
                    with col2:
                        requested_delivery_date = st.date_input(
                            "Requested Delivery Date", value=pd.to_datetime("today"))

                    # Calculate delivery delay
                    delivery_delay = (actual_ship_date -
                                    requested_delivery_date).days
                    st.write(
                        f"**Calculated Delivery Delay:** {delivery_delay} days")

                    # Numerical input fields
                    st.markdown("""<div class="content-box">
                    <p>Total Quantity</p>
                    </div>""", unsafe_allow_html=True)
                    total_quantity = st.number_input(
                        "Total Quantity", min_value=0, value=100)
                    st.markdown("""<div class="content-box">
                    <p>Total Order Lines</p>
                    </div>""", unsafe_allow_html=True)
                    total_order_lines = st.number_input(
                        "Total Order Lines", min_value=0, value=5)
                    st.markdown("""<div class="content-box">
                    <p>Total Cube (CBM)</p>
                    </div>""", unsafe_allow_html=True)
                    total_cube = st.number_input(
                        "Total Cube (CBM)", min_value=0.0, value=10.0)
                    st.markdown("""<div class="content-box">
                    <p>Total Gross Weight (KG)</p>
                    </div>""", unsafe_allow_html=True)
                    total_gross_weight = st.number_input(
                        "Total Gross Weight (KG)", min_value=0.0, value=50.0)

                    if st.button("Predict Individual Churn"):
                        # Prepare input DataFrame
                        input_data = pd.DataFrame([[total_quantity, total_order_lines, total_cube,
                                                    total_gross_weight, delivery_delay]],
                                                columns=["TotalQuantity", "TotalOrderLines",
                                                        "TotalCube", "TotalGrossWeight", "DeliveryDelay"])

                        # Scale features
                        scaled_input = scaler.transform(input_data)

                        # Get predictions
                        rf_proba = rf_model.predict_proba(scaled_input)[0][1]
                        nn_proba = nn_model.predict(scaled_input, verbose=0)[0][0]
                        hybrid_proba = (rf_proba + nn_proba) / 2
                        final_prediction = "Churn" if hybrid_proba > 0.5 else "No Churn"

                        # Display results with styling
                        st.markdown(
        f"""
        <div class="content-box">
            <h2>Prediction Results</h2>
            <div style="display: flex; justify-content: space-around;">
                <div style="text-align: center;">
                    <h4>Random Forest Prediction</h4>
                    <p style="font-size: 20px; font-weight: bold;">{rf_proba:.1%}</p>
                    <p>Churn Probability</p>
                </div>
                <div style="text-align: center;">
                    <h4>Neural Network Prediction</h4>
                    <p style="font-size: 20px; font-weight: bold;">{nn_proba:.1%}</p>
                    <p>Churn Probability</p>
                </div>
                <div style="text-align: center;">
                    <h4>Hybrid AI Prediction</h4>
                    <p style="font-size: 20px; font-weight: bold;">{final_prediction}</p>
                    <p>Confidence: {hybrid_proba:.1%}</p>
                </div>
            </div>
            <div style="margin-top: 20px; padding: 10px; border-radius: 5px; 
                        {"background-color: #ffdddd; color: red;" if final_prediction == "Churn" else "background-color: #ddffdd; color: green;"}">
                <p style="font-size: 16px; font-weight: bold;">
                    {"High risk of customer churn! Consider proactive retention measures." if final_prediction == "Churn" else "Low churn risk detected. Customer appears satisfied with service."}
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


                        # # Create columns for metrics
                        # col1, col2, col3 = st.columns(3)

                        # with col1:
                        #     st.metric(label="Random Forest Prediction",
                        #               value=f"{rf_proba:.1%}",
                        #               delta="Churn Probability")

                        # with col2:
                        #     st.metric(label="Neural Network Prediction",
                        #               value=f"{nn_proba:.1%}",
                        #               delta="Churn Probability")

                        # with col3:
                        #     st.metric(label="Hybrid AI Prediction",
                        #               value=final_prediction,
                        #               delta=f"Confidence: {hybrid_proba:.1%}",
                        #               delta_color="off")

                        # # Visual feedback
                        # if final_prediction == "Churn":
                        #     st.error(
                        #         "High risk of customer churn! Consider proactive retention measures.")
                        # else:
                        #     st.success(
                        #         "Low churn risk detected. Customer appears satisfied with service.")

                        # Add explanation section
                        st.markdown(
        f"""
        <div class="content-box">
            <h2>Prediction Results</h2>
            <div style="margin-top: 20px;">
                <h3>Understanding the Predictions</h3>
                <p><strong>Prediction Breakdown</strong></p>
                <ul>
                    <li><strong>Random Forest:</strong> Ensemble decision tree model (Accuracy: ~74%)</li>
                    <li><strong>Neural Network:</strong> Deep learning model with 3 hidden layers</li>
                    <li><strong>Hybrid AI:</strong> Combined prediction average for enhanced accuracy</li>
                </ul>
                <p><strong>Key Factors Considered</strong></p>
                <ul>
                    <li>Delivery timeliness (delay calculation)</li>
                    <li>Order volume and complexity</li>
                    <li>Shipment dimensions and weight</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

                    #     with st.expander("Understanding the Predictions"):
                    #         st.markdown("""
                    #         **Prediction Breakdown:**
                    #         - **Random Forest:** Ensemble decision tree model (Accuracy: ~74%)
                    #         - **Neural Network:** Deep learning model with 3 hidden layers
                    #         - **Hybrid AI:** Combined prediction average for enhanced accuracy
                            
                    #         **Key Factors Considered:**
                    #         - Delivery timeliness (delay calculation)
                    #         - Order volume and complexity
                    #         - Shipment dimensions and weight
                    #         """)
                    # st.markdown('</div>', unsafe_allow_html=True)

            with tab2:
                # Company-Level Prediction Section
                with st.container():
                    st.markdown(
                        """<div class="content-box">
                    <h2>Company Level Churn Prediction</h2>
                    <p>Predict customer churn based on shipment details and delivery performance.</p>
                    <p>Predict churn for all records of a selected company.</p>
                </div>
                """, unsafe_allow_html=True)
                    

                    # Dropdown for company selection
                    st.markdown(
                        """<div class="content-box">
                    <p>Select a Company</p>
                </div>
                """, unsafe_allow_html=True)
                    company_list = df["Company"].unique().tolist()
                    selected_company = st.selectbox(
                        "Select a Company", company_list)

                    if st.button("Predict Company Churn"):
                        # Filter records for the selected company
                        company_records = df[df["Company"] == selected_company]

                        if company_records.empty:
                            st.warning(
                                f"No records found for company: {selected_company}")
                        else:
                            # Prepare input features
                            X_company = company_records[[
                                "TotalQuantity", "TotalOrderLines", "TotalCube", "TotalGrossWeight", "DeliveryDelay"]]
                            X_company_scaled = scaler.transform(X_company)

                            # Get predictions from both models
                            rf_preds = rf_model.predict_proba(
                                X_company_scaled)[:, 1]
                            nn_preds = nn_model.predict(X_company_scaled).flatten()

                            # Hybrid Model: Average predictions
                            hybrid_preds = (rf_preds + nn_preds) / 2
                            # Convert to binary classification
                            hybrid_preds_binary = np.where(
                                hybrid_preds > 0.5, 1, 0)

                            # Count occurrences of Churn (1) and No Churn (0)
                            churn_count = np.sum(hybrid_preds_binary)
                            no_churn_count = len(hybrid_preds_binary) - churn_count

                            # Final Decision: Majority vote
                            final_decision = "Churn" if churn_count > no_churn_count else "No Churn"
                            st.markdown(
                                f"""
                                <div class="content-box">
                                    <h2>Prediction Results</h2>
                                    <div style="text-align: center; margin-bottom: 20px;">
                                        <h3>Final Decision</h3>
                                        <p style="font-size: 24px; font-weight: bold;">{final_decision}</p>
                                    </div>
                                    <div style="display: flex; justify-content: space-around; margin-bottom: 20px;">
                                        <div style="text-align: center;">
                                            <h4>Total Records</h4>
                                            <p style="font-size: 20px; font-weight: bold;">{len(hybrid_preds_binary)}</p>
                                        </div>
                                        <div style="text-align: center;">
                                            <h4>Churn Predictions</h4>
                                            <p style="font-size: 20px; font-weight: bold;">{int(churn_count)}</p>
                                        </div>
                                    </div>
                                    <div style="margin-top: 20px; padding: 10px; border-radius: 5px; 
                                                {"background-color: #ffdddd; color: red;" if final_decision == "Churn" else "background-color: #ddffdd; color: green;"}">
                                        <p style="font-size: 16px; font-weight: bold;">
                                            {"**Warning:** " + selected_company + " has a high risk of churn. Proactive measures are recommended."
                                            if final_decision == "Churn" else
                                            "**Good News:** " + selected_company + " has a low risk of churn."}
                                        </p>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            # # Display results
                            # st.markdown("### Prediction Results")
                            # st.metric(label="Final Decision", value=final_decision)

                            # # Create columns for detailed metrics
                            # col1, col2 = st.columns(2)
                            # with col1:
                            #     st.metric(label="Total Records",
                            #               value=len(hybrid_preds_binary))
                            # with col2:
                            #     st.metric(label="Churn Predictions",
                            #               value=int(churn_count))

                            # # Visual feedback
                            # if final_decision == "Churn":
                            #     st.error(
                            #         f"**Warning:** {selected_company} has a high risk of churn. Proactive measures are recommended.")
                            # else:
                            #     st.success(
                            #         f"**Good News:** {selected_company} has a low risk of churn.")

                            # Show detailed predictions in an expandable section
                            st.markdown(
                        """<div class="content-box">
                    <p>View Detailed Predictions</p>
                </div>
                """, unsafe_allow_html=True)
                            with st.expander("Detailed Predictions"):
                                st.markdown(
                        """<div class="content-box">
                    <p>Detailed Predictions for Each Record</p>
                </div>
                """, unsafe_allow_html=True)
                                results_df = pd.DataFrame({
                                    "Record ID": company_records.index,
                                    "Random Forest Probability": rf_preds,
                                    "Neural Network Probability": nn_preds,
                                    "Hybrid Probability": hybrid_preds,
                                    "Prediction": ["Churn" if x == 1 else "No Churn" for x in hybrid_preds_binary]
                                })
                                st.dataframe(results_df)

                    st.markdown('</div>', unsafe_allow_html=True)
