import streamlit as st
import base64
from streamlit_option_menu import option_menu
import os
from warehouse_management import run_warehouse_management  # Import your warehouse management function

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
                "Demand & Return Forecasting",
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

    elif selected == "Demand & Return Forecasting":
        st.markdown(
            """
            <div class="content-box">
                <h2>Demand & Return Forecasting</h2>
                <p>Details about demand forecasting go here.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

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