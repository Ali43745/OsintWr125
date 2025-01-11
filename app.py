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
    SELECT Weapon_Name,Type, Weapon_Category, Origin, Development, Caliber, Length, Barrel_Length, Weight, Width, Height, Action, Beam, Downloaded_Image_Name FROM dbo_final_text1
    """
    return pd.read_sql(query, engine)

page = st.number_input("Page", min_value=1, max_value=10, step=1)
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

        
    
else:
    import os
    import pandas as pd
    import streamlit as st
    from fpdf import FPDF

    # Dynamically get the current page
    current_page = st.session_state.current_page

    # Display the Page Title
    st.title(f"{current_page}")

    # Display the Category Heading
    st.header(f"Category: {current_page}")

    # Add description or further dynamic content
    st.write(f"This is the dynamically created page for **{current_page}**.")

    # Base image directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    IMAGE_FOLDER = os.path.join(BASE_DIR, "normalized_images")
    placeholder_image_path = os.path.join(IMAGE_FOLDER, "placeholder.jpeg")

    # Function to find images directly based on the folder matching the category
    def find_images_for_category(base_folder, category_name):
        """Find all images in the folder directly matching the category name."""
        category_folder = os.path.join(base_folder, category_name)
        if os.path.exists(category_folder) and os.path.isdir(category_folder):
            images = [
                (os.path.join(category_folder, file), file)
                for file in os.listdir(category_folder)
                if file.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            return images
        return []

    # Function to load details for the images from the database
    def load_image_details(file_name):
        file_name_escaped = file_name.replace("'", "''")  # Escape single quotes for SQL
        try:
            query = f"""
            SELECT Weapon_Name AS 'Weapon Name', Development AS 'Development Era', Origin,
               Weapon_Category AS 'Weapon Category', Type, Caliber
            FROM dbo_final_text1
            WHERE Downloaded_Image_Name = '{file_name_escaped}'
            """
            result = pd.read_sql(query, engine)
            if not result.empty:
                details = result.iloc[0].dropna().to_dict()  # Drop any columns with NaN values
                return details
        except Exception as e:
            print(f"Error loading details for {file_name}: {e}")
        return {}

    # Function to create a PDF with image details
    def create_pdf(images_with_details, output_file):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        for image_path, details in images_with_details:
            pdf.add_page()

            # Add the image
            if os.path.exists(image_path):
                pdf.image(image_path, x=10, y=10, w=100)

            # Add the details
            pdf.set_font("Arial", size=12)
            pdf.ln(110)  # Move below the image
            for key, value in details.items():
                safe_value = str(value).encode('latin-1', 'ignore').decode('latin-1')  # Handle unsupported characters
                pdf.cell(0, 10, f"{key}: {safe_value}", ln=True)

        pdf.output(output_file)

    # Get images for the current category
    images = find_images_for_category(IMAGE_FOLDER, current_page)

    # Load details for all images
    image_details = []
    for image_path, file_name in images:
        details = load_image_details(file_name)
        if details:
            image_details.append((image_path, file_name, details))

    # Filter options for the current page based on loaded image details
    if image_details:
        # Initialize dynamic filters
        filtered_image_details = image_details

        # Extract available years and origins dynamically
        def get_filter_options(filtered_details, field):
            return ["All"] + sorted(
                {details.get(field) for _, _, details in filtered_details if field in details and details.get(field)}
            )

        selected_year = "All"
        selected_origin = "All"

        while True:
            available_years = get_filter_options(filtered_image_details, "Development Era")
            available_origins = get_filter_options(filtered_image_details, "Origin")

            col1, col2 = st.columns(2)
            with col1:
                selected_year = st.selectbox("Filter by Year", options=available_years, key="year_filter")
            with col2:
                selected_origin = st.selectbox("Filter by Origin", options=available_origins, key="origin_filter")

            # Apply filters
            filtered_image_details = [
                (image_path, file_name, details)
                for image_path, file_name, details in image_details
                if (selected_year == "All" or details.get("Development Era") == selected_year)
                and (selected_origin == "All" or details.get("Origin") == selected_origin)
            ]

            # Stop infinite loop
            break

        
        # Display images and their details with improved layout

    # Display images and their details with improved layout
    if filtered_image_details:
        st.write("### Weapon Images")
        cols_per_row = 3
        rows = [
            filtered_image_details[i: i + cols_per_row] for i in range(0, len(filtered_image_details), cols_per_row)
        ]

        # Combined Download: Create a ZIP file for all filtered images and their PDFs
        import zipfile
        from io import BytesIO

        combined_zip_buffer = BytesIO()
        with zipfile.ZipFile(combined_zip_buffer, "w") as zip_file:
            for image_path, file_name, details in filtered_image_details:
                if os.path.exists(image_path):
                    zip_file.write(image_path, arcname=file_name)

                # Generate a PDF for the current image and its details
                pdf = FPDF()
                pdf.set_auto_page_break(auto=True, margin=15)
                pdf.add_page()

                # Add image to the PDF
                if os.path.exists(image_path):
                    pdf.image(image_path, x=10, y=10, w=100)

                # Add details to the PDF
                pdf.set_font("Arial", size=12)
                pdf.ln(110)  # Move below the image
                for key, value in details.items():
                    safe_value = str(value).encode('latin-1', 'ignore').decode('latin-1')  # Handle unsupported characters
                    pdf.cell(0, 10, f"{key}: {safe_value}", ln=True)

                # Save the PDF to the ZIP
                pdf_file_path = f"{file_name}_details.pdf"
                pdf.output(pdf_file_path)
                zip_file.write(pdf_file_path, arcname=pdf_file_path)

        combined_zip_buffer.seek(0)

        # Add a download button for all filtered images and PDFs
        st.download_button(
            label="Download All Filtered Images and Details",
            data=combined_zip_buffer,
            file_name="filtered_images_and_details.zip",
            mime="application/zip",
        )

        # Display each image in a grid layout
        for row in rows:
            cols = st.columns(len(row))
            for col, (image_path, file_name, details) in zip(cols, row):
                with col:
                    # Display image with consistent height
                    if os.path.exists(image_path):
                        st.image(image_path, caption="", use_container_width=True, output_format="JPEG")
                    else:
                        st.image(placeholder_image_path, caption="Image Not Available", use_container_width=True)

                    # Display the file name in one line below the image
                    st.markdown(
                        f"<div style='text-align: center; font-size: 14px; font-weight: bold; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;'>{file_name}</div>",
                        unsafe_allow_html=True,
                    )

                    # Add a details button
                    if st.button(f"Details: {file_name}", key=f"details_button_{file_name}"):
                        st.markdown("<br>", unsafe_allow_html=True)  # Add space after the button
                        with st.expander(f"Details of {file_name}", expanded=True):
                            for key, value in details.items():
                                st.write(f"**{key}:** {value}")

                            # Individual Downloads
                            # Create a PDF for the selected image and its details
                            pdf = FPDF()
                            pdf.set_auto_page_break(auto=True, margin=15)
                            pdf.add_page()

                            # Add the image to the PDF
                            if os.path.exists(image_path):
                                pdf.image(image_path, x=10, y=10, w=100)

                            # Add the details to the PDF
                            pdf.set_font("Arial", size=12)
                            pdf.ln(110)  # Move below the image
                            for key, value in details.items():
                                safe_value = str(value).encode('latin-1', 'ignore').decode('latin-1')  # Handle unsupported characters
                                pdf.cell(0, 10, f"{key}: {safe_value}", ln=True)

                            # Save the PDF
                            pdf_file_path = os.path.join(BASE_DIR, f"{file_name}_details.pdf")
                            pdf.output(pdf_file_path)

                            # Provide a download button for the PDF
                            with open(pdf_file_path, "rb") as f:
                                st.download_button(
                                    label="Download PDF with Details",
                                    data=f,
                                    file_name=f"{file_name}_details.pdf",
                                    mime="application/pdf",
                                )

                            # Provide a download button for the image itself
                            with open(image_path, "rb") as img_file:
                                st.download_button(
                                    label="Download Image",
                                    data=img_file,
                                    file_name=file_name,
                                    mime="image/jpeg",
                                )
    else:
        st.warning("No images match the selected filters.")
