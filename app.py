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
from io import BytesIO



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
    SELECT Weapon_Name, Source, Type, Weapon_Category, Origin, Development, Caliber, Length, Barrel_Length, Weight, Width, Height, Action, Complement, Speed, Downloaded_Image_Name FROM dbo_final_text1
    """
    data = pd.read_sql(query, engine)
    
    # Ensure all "Origin" entries are in title format
    data['Origin'] = data['Origin'].str.title()
    
    # Exclude rows where "Origin" contains "7.5 cm Feldkanone 18 int."
    data = data[~data['Origin'].str.contains("7.5 Cm Feldkanone 18", case=False, na=False)]
    
    return data
    
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

# Function to get current page from URL
def get_current_page():
    params = st.experimental_get_query_params()
    return params.get("page", ["home"])[0]  # Default to "home"

# Handle Page Navigation
if "current_page" not in st.session_state:
    st.session_state.current_page = get_current_page()

# Sidebar Navigation
st.sidebar.markdown("### Navigation")
page_names = [page["name"] for page in pages_config["pages"]]
selected_page = st.sidebar.selectbox("Select Page", page_names, key="page_selector")

if selected_page != st.session_state.current_page:
    st.session_state.current_page = selected_page.title().replace(" ", "-")
    st.experimental_set_query_params(page=st.session_state.current_page)

# Separate buttons for News Section and AI Prediction Visualizations
if st.sidebar.button("📜 News Section"):
    st.session_state.current_page = "News-Section"
    st.experimental_set_query_params(page="News-Section")

if st.sidebar.button("🔍 AI Prediction Visualizations"):
    st.session_state.current_page = "ai-prediction"
    st.experimental_set_query_params(page="ai-prediction")

# Render pages based on URL
current_page = st.session_state.current_page


if st.session_state.current_page == "Home":
    # Dropdown for weapon types
   
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
        filtered_data['Type'] = filtered_data['Type'].str.replace("_", " ")

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
        filtered_data['Type'] = filtered_data['Type'].str.replace("_", " ")
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
        data['Type'] = data['Type'].str.replace("_", " ")
        fig = px.line(
            data,
            x="Development",
            y="Weapon_Name",
            color="Type",
            title="Weapon Production Over Time",
            labels={"Weapon_Name": "Name of Weapons", "Development": "Production Year Range"},
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

    # Function to safely add text to the PDF
    def safe_add_text(pdf, text):
        try:
            pdf.cell(0, 10, text, ln=True)
        except UnicodeEncodeError:
            text = text.encode('latin-1', 'ignore').decode('latin-1')
            pdf.cell(0, 10, text, ln=True)


    # Function to clean up the category name for button labels
    def clean_category_name(name):
        parts = name.split("_")
        cleaned_name = " ".join(parts[1:]).title()
        return cleaned_name

    for row in rows:
        cols = st.columns(len(row))
        for col, category in zip(cols, row):
            # Clean the category name by replacing "_" with " " and capitalizing words
            cleaned_category_name = category.replace("_", " ").title()

            category_dir = os.path.join(IMAGE_FOLDER, category.replace(" ", "_"))
            images_in_category = []

            if os.path.exists(category_dir) and os.path.isdir(category_dir):
                for file_name in os.listdir(category_dir):
                    if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                        images_in_category.append(file_name)

            # Display image or placeholder
            if images_in_category:
                first_image = os.path.join(category_dir, images_in_category[0])
                col.image(first_image, caption=cleaned_category_name, use_container_width=True)
                with col:
                    with open(first_image, "rb") as file:
                        col.download_button(
                            label="Download as PNG",
                            data=file,
                            file_name=os.path.basename(first_image),
                            mime="image/png"
                        )

                # Add expander for details
                with col.expander(f"Details of {cleaned_category_name} \u2b07", expanded=False):
                    st.write(f"### Details of {cleaned_category_name}")

                    # Count the total number of images and unique Weapon Categories using the DataFrame
                    filtered_data = data[data["Type"] == category]  # Filter for the specific category
                    total_images = len(filtered_data)  # Count total number of images
                    unique_categories = filtered_data["Weapon_Category"].nunique()  # Count unique weapon categories

                    st.write(f"**Total Weapons:** {total_images}")
                    st.write(f"**Unique Weapon Categories:** {unique_categories}")

                    # Create a ZIP file of all images and details
                    combined_zip_buffer = BytesIO()
                    with zipfile.ZipFile(combined_zip_buffer, "w") as zip_file:
                        for file_name in images_in_category:
                            image_path = os.path.join(category_dir, file_name)
                            try:
                                zip_file.write(image_path, arcname=file_name)

                                # Fetch details for the image from the DataFrame
                                image_details = filtered_data[filtered_data["Downloaded_Image_Name"] == file_name].to_dict(orient="records")
                                if image_details:
                                    details = image_details[0]

                                    # Create a PDF with details for each image
                                    pdf = FPDF()
                                    pdf.set_auto_page_break(auto=True, margin=15)
                                    pdf.add_page()
                                    pdf.set_font("Arial", size=12)
                                    pdf.cell(0, 10, f"Details of {file_name.replace('_', ' ')}", ln=True)
                                    pdf.ln(10)

                                    for key, value in details.items():
                                        if pd.notna(value):
                                            # Safely encode values
                                            safe_value = str(value).replace("’", "'")
                                            key_cleaned = key.replace("_", " ").title()  # Clean key names for better readability
                                            try:
                                                pdf.cell(0, 10, f"{key_cleaned}: {safe_value}", ln=True)
                                            except UnicodeEncodeError:
                                                # Skip problematic characters
                                                continue

                                    # Save the PDF
                                    pdf_file_path = f"{file_name.replace('_', ' ')}_details.pdf"
                                    pdf.output(pdf_file_path)
                                    zip_file.write(pdf_file_path, arcname=pdf_file_path)
                                    os.remove(pdf_file_path)  # Clean up temporary PDF files

                            except Exception:
                                # Skip problematic files silently
                                continue

                    combined_zip_buffer.seek(0)

                    # Download button for ZIP file
                    st.download_button(
                        label=f"Download All Images and Details of {cleaned_category_name} as ZIP",
                        data=combined_zip_buffer,
                        file_name=f"{cleaned_category_name.replace(' ', '_')}_images_and_details.zip",
                        mime="application/zip",
                    )

            elif os.path.exists(placeholder_image_path):
                col.image(placeholder_image_path, caption=f"{cleaned_category_name} (Placeholder)", use_container_width=True)
            else:
                col.error(f"No image available for {cleaned_category_name}")

# AI Prediction visualizations
elif st.session_state.current_page == "ai-prediction":
    st.write("### AI Prediction Analysis")
    # Load CSV file
    @st.cache_data
    def load_predictions():
        df = pd.read_csv("weapon_predictions_with_labels.csv")
        # Remove rows with "Unknown" values
        return df

    data = load_predictions()

    # Display data as table with download option
    st.write("#### Weapon Predictions Data")
    st.dataframe(data)

    # Download CSV button
    csv_data = data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Table as CSV",
        data=csv_data,
        file_name="Weapon_Predictions_Table.csv",
        mime="text/csv",
    )

   # 1. Bar Chart: Count of Actual vs Predicted Weapon Types
    st.write("### Actual vs Predicted Weapon Types")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.countplot(
        data=data.melt(id_vars=["Weapon_Name"], value_vars=["Type_Numerical_Label", "Predicted_Type"]),
        x="value",
        hue="variable",
        ax=ax1,
        palette="Set2"
    )
    ax1.set_title("Actual vs Predicted Weapon Types")
    ax1.set_ylabel("Count")
    st.pyplot(fig1)

    # Download bar chart as PDF
    def download_plot(fig, filename):
        buf = BytesIO()
        fig.savefig(buf, format="pdf")
        buf.seek(0)
        return buf

    pdf_data1 = download_plot(fig1, "Actual_vs_Predicted_Types.pdf")
    st.download_button("Download Bar Chart as PDF", pdf_data1, file_name="Actual_vs_Predicted_Types.pdf", mime="application/pdf")

    # 2. Accuracy Calculation: Percentage of Correct Predictions
    correct_predictions = data[data["Type"] == data["Predicted_Type_Label"]]
    accuracy = (len(correct_predictions) / len(data)) * 100
    st.write(f"### Prediction Accuracy: {accuracy:.2f}%")
    st.write(f"Total Predictions: {len(data)} | Correct Predictions: {len(correct_predictions)}")

    # 3. Pie Chart: Distribution of Prediction Outcomes (Correct vs Incorrect)
    st.write("### Distribution of Prediction Outcomes")
    outcome_data = pd.DataFrame({
        "Outcome": ["Correct", "Incorrect"],
        "Count": [len(correct_predictions), len(data) - len(correct_predictions)]
    })
    fig2, ax2 = plt.subplots()
    ax2.pie(outcome_data["Count"], labels=outcome_data["Outcome"], autopct="%1.1f%%", startangle=90)
    ax2.set_title("Prediction Outcomes")
    st.pyplot(fig2)

    pdf_data2 = download_plot(fig2, "Prediction_Outcomes_Distribution.pdf")
    st.download_button("Download Pie Chart as PDF", pdf_data2, file_name="Prediction_Outcomes_Distribution.pdf", mime="application/pdf")
    
    # 4. Table of Recognized vs Unrecognized Types
    st.write("### Model Prediction Performance by Weapon Type")
    performance_table = data.groupby("Type").apply(
        lambda x: pd.Series({
            "Total Predictions": len(x),
            "Correct Predictions": len(x[x["Type"] == x["Predicted_Type_Label"]]),
            "Incorrect Predictions": len(x[x["Type"] != x["Predicted_Type_Label"]]),
            "Accuracy (%)": 100 * len(x[x["Type"] == x["Predicted_Type_Label"]]) / len(x)
        })
    ).reset_index()

    st.dataframe(performance_table)

    # Download Performance Table as CSV
    performance_csv = performance_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Performance Table as CSV",
        data=performance_csv,
        file_name="Weapon_Type_Prediction_Performance.csv",
        mime="text/csv"
    )

    
    # 5. Display Confusion Matrix Image
    st.write("### Confusion Matrix of AI Model")
    confusion_matrix_path = "confusion_matrix_model1.png"  # Replace with actual file path
    st.image(confusion_matrix_path, caption="Confusion Matrix - Model 1", use_container_width=True)

    # Download Confusion Matrix Image
    with open(confusion_matrix_path, "rb") as img_file:
        st.download_button(
            label="Download Confusion Matrix as PNG",
            data=img_file,
            file_name="confusion_matrix_model1.png",
            mime="image/png"
        )

    # 6. Display ROC Curve Image
    st.write("### ROC Curve of AI Model")
    roc_curve_path = "roc_curve_model1.png"  # Replace with actual file path
    st.image(roc_curve_path, caption="ROC Curve - Model 1", use_container_width=True)

    # Download ROC Curve Image
    with open(roc_curve_path, "rb") as img_file:
        st.download_button(
            label="Download ROC Curve as PNG",
            data=img_file,
            file_name="roc_curve_model1.png",
            mime="image/png"
        )

# News Section
elif st.session_state.current_page == "News-Section":
    st.write("### News Section")
    st.write(f"Debug: Current Page - {st.session_state.current_page}")  # Confirm correct page in debug logs
   
    # Prepare the data for the news
    news_data = data[[
        "Weapon_Name", "Type", "Weapon_Category", "Origin", "Development", "Caliber",
        "Length", "Barrel_Length", "Weight", "Width", "Height", "Action", "Complement",
        "Speed", "Downloaded_Image_Name"
    ]].fillna("Unknown").reset_index(drop=True)

    total_news_items = len(news_data)
    IMAGE_FOLDER = os.path.join(os.path.dirname(__file__), "normalized_images")

    # Debugging log: Check if news data is loaded correctly
    st.write(f"Debug: Total News Items - {total_news_items}")

    # State to keep track of the current news index
    if "news_index" not in st.session_state:
        st.session_state.news_index = 0

    # Functions to update the news index without reloading the page
    def update_news_index(offset):
        new_index = (st.session_state.news_index + offset) % total_news_items
        st.session_state.news_index = new_index
        # Debugging log: Check the updated index
        st.write(f"Debug: Updated News Index - {st.session_state.news_index}")

    # Function to generate PDF for a single news item
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

        # Add the details (skip "Unknown" values)
        pdf.set_font("Arial", size=12)
        for key, value in news_item.items():
            if key != "Downloaded_Image_Name" and value != "Unknown":
                pdf.cell(0, 10, f"{key.replace('_', ' ')}: {value}", ln=True)

        # Save the PDF
        pdf_output_path = f"{news_item['Weapon_Name'].replace(' ', '_')}_news.pdf"
        pdf.output(pdf_output_path)
        return pdf_output_path

    # Display the current news item
    current_news = news_data.iloc[st.session_state.news_index]

    # Debugging log: Displaying current news item
    st.write(f"Debug: Displaying News for - {current_news['Weapon_Name']}")

    # Get the image for the current news item
    image_path = None
    if pd.notnull(current_news["Downloaded_Image_Name"]):
        image_name = current_news["Downloaded_Image_Name"]
        weapon_type = current_news["Type"].replace(" ", "_")
        category_folder = os.path.join(IMAGE_FOLDER, weapon_type)

        # Normalize filenames for better matching
        def normalize_name(name):
            return name.lower().strip().replace("_", " ").replace(".jpg", "").replace(".jpeg", "")

        normalized_image_name = normalize_name(image_name)
        if os.path.exists(category_folder) and os.path.isdir(category_folder):
            matching_file = next((f for f in os.listdir(category_folder) if normalize_name(f) == normalized_image_name), None)
            if matching_file:
                image_path = os.path.join(category_folder, matching_file)

    if not image_path or not os.path.exists(image_path):
        image_path = "placeholder_image_path_here.jpeg"  # Ensure this exists

    # Display the news image
    st.image(image_path, caption=f"Image for {current_news['Weapon_Name']}", use_container_width=True)
    st.write(f"**Here is {current_news['Weapon_Name']}**, developed in **{current_news['Development']}**.")
    
    # Display the news details (excluding "Unknown" values)
    for key, value in current_news.items():
        if key != "Downloaded_Image_Name" and value != "Unknown":
            st.write(f"**{key.replace('_', ' ')}:** {value}")

    # Debugging: News navigation buttons with log messages
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("⬅️ Previous", key="prev_button"):
            update_news_index(-1)
            st.write("Debug: Previous Button Clicked")  # Log button click

    with col3:
        if st.button("➡️ Next", key="next_button"):
            update_news_index(1)
            st.write("Debug: Next Button Clicked")  # Log button click

    # Download news as PDF button
    pdf_path = generate_single_news_pdf(current_news, image_path)
    with col2:
        if st.download_button(
            label="Download Current News as PDF",
            data=open(pdf_path, "rb").read(),
            file_name=os.path.basename(pdf_path),
            mime="application/pdf",
            key="news_pdf_download"
        ):
            st.write("Debug: PDF Download Button Clicked")  # Log button click
    os.remove(pdf_path)  # Clean up temporary file

    # Final log to confirm the current section
    


else:
    import os
    import pandas as pd
    import streamlit as st
    from fpdf import FPDF
    import zipfile
    from io import BytesIO

    # Dynamically get the current page
    current_page = st.session_state.current_page

    # Normalize current page name for folder matching
    normalized_current_page = current_page.replace(" ", "_").lower()

    # Cleaned display name (replace "_" with " " and title case)
    display_name = current_page.replace("_", " ").title()

    # Display the cleaned Page Title
    st.title(f"{display_name}")

    # Display the cleaned Category Heading
    st.header(f"Category: {display_name}")

    # Add description or further dynamic content
    st.write(f"This is the dynamically created page for **{display_name}**.")


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

    # Normalize image names for display after all processing is done
    def normalize_filename(file_name):
        """Replace underscores with spaces and capitalize each word in the filename."""
        normalized_name = file_name.replace("_", " ").title()  # Title-case and replace underscores
        return normalized_name

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
                # Filter by Year
                available_years = get_filter_options(filtered_image_details, "Development Era")
                selected_year = st.selectbox("Filter by Year", options=available_years, key="year_filter")
    
            # Apply the year filter
            if selected_year != "All":
                filtered_image_details = [
                   (image_path, file_name, details)
                   for image_path, file_name, details in image_details
                   if details.get("Development Era") == selected_year
                    ]
            with col2:
                available_origins = get_filter_options(filtered_image_details, "Origin")
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
        if filtered_image_details:
            st.write("### Weapon Images")
            cols_per_row = 3
            rows = [
                filtered_image_details[i: i + cols_per_row] for i in range(0, len(filtered_image_details), cols_per_row)
            ]

            # Combined Download: Create a ZIP file for all filtered images and their PDFs
            combined_zip_buffer = BytesIO()
            with zipfile.ZipFile(combined_zip_buffer, "w") as zip_file:
                for image_path, file_name, details in filtered_image_details:
                    normalized_file_name = normalize_filename(file_name)  # Normalize filename before using
                    if os.path.exists(image_path):
                        zip_file.write(image_path, arcname=normalized_file_name)

                    # Generate a PDF for the current image and its details
                    pdf = FPDF()
                    pdf.set_auto_page_break(auto=True, margin=15)
                    pdf.add_page()

                    # Add image to the PDF with consistent height
                    if os.path.exists(image_path):
                        pdf.image(image_path, x=10, y=10, w=100, h=75)  # Ensure consistent height

                    # Add details to the PDF
                    pdf.set_font("Arial", size=10)
                    pdf.ln(85)  # Adjust to keep consistent space below the image
                    for key, value in details.items():
                        safe_value = str(value).encode('latin-1', 'ignore').decode('latin-1')  # Handle unsupported characters
                        pdf.cell(0, 10, f"{key}: {safe_value}", ln=True)

                    # Save the PDF to the ZIP
                    pdf_file_path = f"{normalized_file_name}_details.pdf"
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
                    normalized_file_name = normalize_filename(file_name)  # Normalize the filename here
                    with col:
                        # Display image with consistent height
                        if os.path.exists(image_path):
                            st.image(image_path, caption="", use_container_width=True, output_format="JPEG")
                        else:
                            st.image(placeholder_image_path, caption="Image Not Available", use_container_width=True)

                        # Display the normalized file name
                        st.markdown(
                            f"<div style='text-align: center; font-size: 14px; background-color: #28a745; font-weight: bold; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;'>{normalized_file_name}</div>",
                            unsafe_allow_html=True,
                        )

                        # Add a single green details button with functionality
                        if st.button(f"See Details", key=f"details_button_{file_name}"):
                            st.markdown("<br>", unsafe_allow_html=True)  # Add space after the button
                            with st.expander(f"Details of {normalized_file_name}", expanded=True):
                                for key, value in details.items():
                                    st.write(f"**{key}:** {value}")

                                # Individual Downloads
                                # Create a PDF for the selected image and its details
                                pdf = FPDF()
                                pdf.set_auto_page_break(auto=True, margin=15)
                                pdf.add_page()

                                # Add the image to the PDF
                                if os.path.exists(image_path):
                                    pdf.image(image_path, x=10, y=10, w=100, h=75)  # Ensure consistent height

                                # Add the details to the PDF
                                pdf.set_font("Arial", size=12)
                                pdf.ln(85)  # Adjust to keep consistent space below the image
                                for key, value in details.items():
                                    safe_value = str(value).encode('latin-1', 'ignore').decode('latin-1')
                                    pdf.cell(0, 10, f"{key}: {safe_value}", ln=True)

                                # Save the PDF
                                pdf_file_path = os.path.join(BASE_DIR, f"{normalized_file_name}_details.pdf")
                                pdf.output(pdf_file_path)

                                # Provide a download button for the PDF
                                with open(pdf_file_path, "rb") as f:
                                    st.download_button(
                                        label="Download PDF with Details",
                                        data=f,
                                        file_name=f"{normalized_file_name}_details.pdf",
                                        mime="application/pdf",
                                    )

                                # Provide a download button for the image itself
                                with open(image_path, "rb") as img_file:
                                    st.download_button(
                                        label="Download Image",
                                        data=img_file,
                                        file_name=f"{normalized_file_name}.jpg",
                                        mime="image/jpeg",
                                    )
    else:
        st.warning("No images match the selected filters.")
