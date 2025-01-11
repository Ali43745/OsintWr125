import streamlit as st
import pandas as pd
import os
import plotly.express as px
from sqlalchemy import create_engine
import toml
from fpdf import FPDF
from pathlib import Path
import os
import torch
from sklearn.preprocessing import LabelEncoder
import requests
import zipfile
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Database connection details
DB_HOST = "junction.proxy.rlwy.net"
DB_USER = "root"
DB_PASSWORD = "GKesHFOMJkurJYvpaVNuqRgTEGYOgFQN"
DB_NAME = "railway"
DB_PORT = "27554"

# Define the database connection
@st.cache_resource
def get_engine():
    engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    return engine

engine = get_engine()

# Load data from weapon_data1 and join with dbo_images
# Load data from weapon_data1
@st.cache_data
def load_data():
    query = """
    SELECT * 
    FROM dbo_final_text1
    WHERE Weapon_Name IN (
        SELECT DISTINCT Weapon_Name
        FROM dbo_final_text1
    );
    """
    return pd.read_sql(query, engine)

data = load_data()

# Resolve the directory path
current_dir = Path(__file__).resolve().parent  # Use resolve() to get the absolute path
os.chdir(current_dir)  # Change the current working directory

# Load the .toml configuration
try:
    pages_config = toml.load(current_dir / ".streamlit/pages.toml")
except Exception as e:
    st.error(f"Error loading pages.toml: {e}")
    st.stop()



# Validate 'pages' key in the configuration
if "pages" not in pages_config:
    st.error("'pages' key not found in the pages.toml file.")
    st.stop()



# Handle Page Navigation
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

# Sidebar Navigation
st.sidebar.markdown("### Navigation")
page_names = ["Home"] + [page["name"] for page in pages_config["pages"]]
selected_page = st.sidebar.selectbox("Go to", page_names, key="page_selector")

if selected_page != st.session_state.current_page:
    st.session_state.current_page = selected_page
    st.experimental_set_query_params(page=selected_page)

# Main Content Rendering Based on Selected Page
if st.session_state.current_page == "Home":
    # Dashboard Page
    st.title("Weapon Insights Dashboard")
    st.write("Explore weapon specifications, search, and visualize data interactively.")

    # Display filtered data
    st.write("### Filtered Data Table")
    st.dataframe(data)

   # Filter the data to exclude origins starting with "source: "
    filtered_data = data[~data['Origin'].str.startswith('Source:', na=False)].drop(columns=['Source'], errors='ignore')

    # Threat Distribution by Origin
    st.write("### Threat Distribution by Origin")
    if not filtered_data.empty:
        fig = px.bar(
            filtered_data,
            x="Origin",
            y="Weapon_Name",
            color="Type",
            title="Threat Distribution by Origin",
            labels={"Weapon_Name": "Weapon Name", "Origin": "Country of Origin"},
        )
        st.plotly_chart(fig)
    else:
        st.warning("No data available for visualization.")

    
    # Load Top 5 Countries Data
    @st.cache_data
    def load_top_countries():
        query = """
        SELECT Origin, COUNT(*) as Weapon_Count
        FROM dbo_final_text1
        GROUP BY Origin
        ORDER BY Weapon_Count DESC
        LIMIT 5;
        """
        return pd.read_sql(query, engine)

    # Display Top 5 Countries Map
    top_countries_data = load_top_countries()
    st.title("Top 5 Countries by Weapon Production")
    st.write("This map shows the top 5 countries that produce the highest number of weapons.")

    if not top_countries_data.empty:
        # Display the map
        st.write("### Top 5 Countries Map")
        fig = px.choropleth(
            top_countries_data,
            locations="Origin",
            locationmode="country names",
            color="Weapon_Count",
            hover_name="Origin",
            title="Top 5 Countries by Weapon Production",
            color_continuous_scale=px.colors.sequential.Plasma,
        )
        fig.update_layout(geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth"))
        st.plotly_chart(fig)

        # Display the data in a table
        st.write("### Top 5 Countries Data")
        st.dataframe(top_countries_data)
    else:
        st.warning("No data available to display.")

    

    # Weapon Categories by Origin (New Graph)
    top_countries_data = (
        filtered_data.groupby("Origin").size().reset_index(name="Weapon_Count")
    )

    if not top_countries_data.empty:
        st.write("### Weapon Categories by Origin")
        weapon_origin_distribution = (
            filtered_data.groupby(["Origin", "Type"])
            .size()
            .reset_index(name="Count")
        )
        fig = px.bar(
            weapon_origin_distribution,
            x="Origin",
            y="Count",
            color="Type",
            title="Distribution of Weapon Categories by Origin",
            labels={"Origin": "Country of Origin", "Count": "Number of Weapons"},
            barmode="stack",
            color_discrete_sequence=px.colors.sequential.Viridis,
        )
        st.plotly_chart(fig)
    else:
        st.warning("No data available for visualization.")

    


   # 1. Weapon Production Over Time
    st.write("### Weapon Production Over Time")
    if not data.empty:
        fig = px.line(
            data,
            x="Development",
            y="Weapon_Name",
            color="Type",
            title="Weapon Production Over Time",
            labels={"Weapon_Name": "Number of Weapons", "Development": "Production Year Range"},
        )
        st.plotly_chart(fig)
    else:
        st.warning("No data available for visualization.")

  
    # Define a function to clean and extract numeric weights
    def clean_weight_column(weight):
        """Extract the first numeric value from the weight string, if available."""
        import re
        if isinstance(weight, str):
            match = re.search(r"\d+(\.\d+)?", weight)  # Match numbers with optional decimals
            if match:
                return float(match.group())
        return None  # Return None if no valid number is found


    # Display Categories with Representative Images
   # Display Categories with Representative Images
    st.write("### Weapon Categories")
    IMAGE_FOLDER = "normalized_images"
    placeholder_image_path = os.path.join(IMAGE_FOLDER, "placeholder.jpeg")
    categories = sorted(data["Type"].dropna().unique())

    cols_per_row = 3
    rows = [categories[i:i + cols_per_row] for i in range(0, len(categories), cols_per_row)]

    # Function to normalize the category name for the redirection URL
    def normalize_name_for_url(name):
        """Normalize category names for URL redirection."""
        return name.replace(" ", "+").replace("_", "+").title()

    # Function to clean up the category name for button labels
    def clean_category_name(name):
        parts = name.split("_")
        cleaned_name = " ".join(parts[1:]).title()
        return cleaned_name
   

    for row in rows:
        cols = st.columns(len(row))
        for col, category in zip(cols, row):
            # Image logic remains the same
            category_dir = os.path.join(IMAGE_FOLDER, category.replace(" ", "_"))
            category_image = None

            if os.path.exists(category_dir) and os.path.isdir(category_dir):
                for file_name in os.listdir(category_dir):
                    if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                        category_image = os.path.join(category_dir, file_name)
                        break

            # Display image or placeholder
            if category_image and os.path.exists(category_image):
                col.image(category_image, caption=category, use_container_width=True)
                with col:
                    with open(category_image, "rb") as file:
                      col.download_button(
                        label="Download as PNG",
                        data=file,
                        file_name=os.path.basename(category_image),
                        mime="image/png"
                    )
                # Add navigation button
                cleaned_name =(category)
                col.button(f"Go to {cleaned_name} Category through navigation bar")
                    
                
                
            elif os.path.exists(placeholder_image_path):
                col.image(placeholder_image_path, caption=f"{category} (Placeholder)", use_container_width=True)
            else:
                col.error(f"No image available for {category}")

        
    st.write("### News Section")

    # Prepare the data for the news
    news_data = data[["Weapon_Name", "Type", "Development", "Weight", "Downloaded_Image_Name"]].dropna().reset_index(
        drop=True
    )
    total_news_items = len(news_data)

    # State to keep track of the current news index
    if "news_index" not in st.session_state:
        st.session_state.news_index = 0
    
    # Function to move to the next news item
    def next_news():
        st.session_state.news_index = (st.session_state.news_index + 1) % total_news_items

    # Function to move to the previous news item
    def prev_news():
        st.session_state.news_index = (st.session_state.news_index - 1) % total_news_items


    def generate_single_news_pdf(news_item, image_path):
     pdf = FPDF()
     pdf.set_auto_page_break(auto=True, margin=15)
     pdf.add_page()

     # Add the title
     pdf.set_font("Arial", size=16, style="B")
     pdf.cell(0, 10, f"News: {news_item['Weapon_Name']}", ln=True, align="C")
     pdf.ln(10)

     # Add the image
     if image_path and os.path.exists(image_path):
         pdf.image(image_path, x=10, y=pdf.get_y(), w=100)
         pdf.ln(50)

     # Add the details
     pdf.set_font("Arial", size=12)
     for key in ["Weapon_Name", "Weapon_Category", "Development", "Weight", "Status"]:
         pdf.cell(0, 10, f"{key.replace('_', ' ')}: {news_item[key]}", ln=True)

     # Save the PDF
     pdf_output_path = f"{news_item['Weapon_Name']}_news.pdf"
     pdf.output(pdf_output_path)
     return pdf_output_path
       

    # Display the current news item
    current_news = news_data.iloc[st.session_state.news_index]

    # Get the image for the current news item
    image_path = None
    if pd.notnull(current_news["Downloaded_Image_Name"]):
        image_name = current_news["Downloaded_Image_Name"]
        weapon_category = current_news["Type"].replace(" ", "_")  # Use Weapon_Category from the data

        # Construct the folder path (include the subfolder structure)
        category_folder = os.path.join(IMAGE_FOLDER, weapon_category)  # Only use Weapon_Category for folder

        # Normalize filenames for better matching
        def normalize_name(name):
            # Remove leading underscores, lowercase, replace "_", and strip extensions
            return (
                name.lower()
                .strip()
                .replace("_", " ")
                .replace(".jpg", "")
                .replace(".jpeg", "")
            )

        # Normalized image name (retain numbers)
        normalized_image_name = normalize_name(image_name)
        if os.path.exists(category_folder) and os.path.isdir(category_folder):
            available_files = [normalize_name(f) for f in os.listdir(category_folder)]

            # Match normalized names
            matching_file = next((f for f in os.listdir(category_folder) if normalize_name(f) == normalized_image_name), None)
            if matching_file:
                image_path = os.path.join(category_folder, matching_file)
            else:
                st.write(f"Image {image_name} not found in {category_folder}. Using placeholder.")
        else:
            st.write(f"Category folder does not exist: {category_folder}")

    # Use placeholder if image path is not found
    if not image_path or not os.path.exists(image_path):
        image_path = placeholder_image_path

    # Display the news image
    st.image(
        image_path,
        caption=f"Image for {current_news['Weapon_Name']}",
        use_container_width=True,
    )

    
    # Display the news description
    st.write(
        f"**Here is {current_news['Weapon_Name']}**, developed in **{current_news['Development']}**"
    )

    # Navigation buttons for the news
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ Previous"):
            prev_news()
    with col3:
        if st.button("➡️ Next"):
            next_news()
       # Button to download the current news item
    with col2:
        if st.button("Download Current News as PDF"):
            pdf_path = generate_single_news_pdf(current_news, image_path)
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="Download Current News",
                    data=f,
                    file_name=os.path.basename(pdf_path),
                    mime="application/pdf"
                )
            os.remove(pdf_path)  # Clean up temporary file

