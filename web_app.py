import streamlit as st
import pandas as pd
import numpy as np
import base64
from streamlit_option_menu import option_menu
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import io

# Set page configuration
st.set_page_config(page_title="OptiFow AI", page_icon="🚚")

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
            menu_title="OptiFlow AI     Intelligent Supply Chain Management System",
            options=[
                "Home",
                "Demand Forecasting",
                "Inventory Management",
                "Warehouse Management",
                "Customer Churn Prediction"
            ],
            icons=[
                "house-fill",       
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

    elif selected == "Demand Forecasting":
    
        st.markdown(
            """
            <div class="content-box">
                <h2>Demand Forecasting</h2>
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

            st.write("### Processed Data Preview:")
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

                st.write("### Forecasted Demand for Next 7 Days:")
                st.dataframe(forecast_df)

                # Plot forecast
                st.subheader("Forecast Visualization")
                selected_item = st.selectbox("Select Item to View Forecast", forecast_df.columns)

                plt.figure(figsize=(10, 5))
                plt.plot(forecast_df.index, forecast_df[selected_item], marker='o', label=f"Forecast for {selected_item}")
                plt.title(f"Forecasted Demand for {selected_item}")
                plt.xlabel("Date")
                plt.ylabel("Demand")
                plt.xticks(rotation=45)
                plt.legend()
                st.pyplot(plt)

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
                selected_materials = st.multiselect("Select Materials", unique_materials)

                if selected_materials:
                    # Filter stock report data based on selected materials
                    selected_stock_data = stock_report[stock_report['Material'].isin(selected_materials)]

                    # Display the relevant stock data
                    st.write("### Selected Materials Stock Report:")
                    st.dataframe(selected_stock_data)

                    # Forecast output for selected stock report materials
                    stock_forecast_data = forecast_df[selected_materials]
                    stock_excel_file = to_excel(stock_forecast_data)

                    # Provide download button for stock forecast data
                    st.download_button(
                        label="Download Stock Forecast Data as Excel",
                        data=stock_excel_file,
                        file_name="selected_stock_forecast.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                    # Plot individual material forecast
                    for material in selected_materials:
                        st.subheader(f"Demand Forecast for {material}")
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
            

    elif selected == "Inventory Management":
        st.markdown(
            """
            <div class="content-box">
                <h2>Inventory Management</h2>
                <p>Details about inventory management go here.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif selected == "Warehouse Management":
        st.markdown(
            """
            <div class="content-box">
                <h2>Warehouse Management</h2>
                <p>Details about warehouse management go here.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif selected == "Customer Churn Prediction":
        st.markdown(
            """
            <div class="content-box">
                <h2>Customer Churn Prediction</h2>
                <p>Details about customer churn prediction go here.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
