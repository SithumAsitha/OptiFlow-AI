import streamlit as st
import base64

# Set page configuration
st.set_page_config(page_title="OPTIWARE", page_icon="🚚", layout="centered")

# Function to encode image as base64
def get_base64_from_file(file_path):
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")

# Path to the background image
background_image = "warehouse.jpg"  # Replace with your image file path
base64_image = get_base64_from_file(background_image)

# Apply custom CSS for background image, button positioning, and hover effect
st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{base64_image}");
        background-size: 100% 100%;
        background-repeat: no-repeat;
        background-position: center;
    }}
    .content-box {{
        background-color: rgba(0, 0, 0, 0.6);
        color: white;
        font-size: 18px;
        text-align: justify;
        padding: 20px;
        border-radius: 10px;
        margin: 20px auto;
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
        background-color: white;
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
        background-color: rgba(255, 255, 255, 0);  /* Transparent white on hover */
        transform: scale(1.05);
    }}
    .stButton > button:hover::after {{
        transform: translateX(0) rotate(0deg);
    }}
    /* Override the active state to keep the color black */
    .stButton > button:active {{
        color: black; /* Set text color to black on click */
        border-color: black; /* Set border color to black on click */
        background-color: rgba(0, 0, 0, 0.6); /* Ensure background remains consistent */
    }}
    </style>
""", unsafe_allow_html=True)


# Page Content
st.markdown(
    """
    <div class="content-box">
        <h2><b>👷🏼 Welcome to OPTIWARE</b></h2>
        <p><b>OPTIWARE is designed to help improve warehouse operations by making tasks more efficient and reducing costs. It achieves this by focusing on key areas such as planning ahead, maintaining stock levels, better organization, and improving customer satisfaction 🎯.</b></p>
        <p><b>The app predicts which products are likely to be needed soon and when returns may occur, helping warehouses stay ahead of demand. It also ensures that stock is replenished in a timely manner, which helps avoid delays and reduce unnecessary labor costs 💰. Additionally, it improves warehouse organization by strategically placing items in easy-to-reach locations based on how often they are picked, making it easier for staff to locate products quickly.</b></p>
        <p><b>OPTIWARE helps to identify customers who might stop buying, allowing businesses to take action and keep their customers happy and loyal. This overall approach leads to smoother warehouse operations, better customer service, and lower costs 🚀.</b></p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Explore Button
if st.button("Explore"):
    st.write("Redirecting to the app...")
    # Add redirection logic here if needed
