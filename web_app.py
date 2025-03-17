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

    elif selected == "Return Forecasting":
    
        st.markdown(
            """
            <div class="content-box">
                <h2>Return Forecasting</h2>
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

                st.write("### Forecasted Return for Next 7 Days:")
                st.dataframe(forecast_df)

                # Plot forecast
                st.subheader("Forecast Visualization")
                selected_item = st.selectbox("Select Item to View Forecast", forecast_df.columns)

                plt.figure(figsize=(10, 5))
                plt.plot(forecast_df.index, forecast_df[selected_item], marker='o', label=f"Forecast for {selected_item}")
                plt.title(f"Forecasted Return for {selected_item}")
                plt.xlabel("Date")
                plt.ylabel("Return")
                plt.xticks(rotation=45)
                plt.legend()
                st.pyplot(plt)

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
                        st.subheader(f"Return Forecast for {material}")
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