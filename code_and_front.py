import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import base64



# Load the historical dataset and train the model
def train_model():
    # Load the historical dataset
    dataset = pd.read_csv(r"D:\Naresh i Class\Sept 2024\24 Sep 24\logit classification.csv")
    
    # Extract features and target variable
    X = dataset.iloc[:, [2, 3]].values  # Assuming Age and Estimated Salary columns
    y = dataset.iloc[:, -1].values      # Assuming the last column is the target variable

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the Logistic Regression model
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    # Evaluate model accuracy
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return classifier, scaler, accuracy

# Train the model
classifier, scaler, model_accuracy = train_model()  #unpacked the variables becoz
# When you call a function that returns multiple values, such as:return classifier, scaler, accuracy. the function returns a 
# tuple containing these values. However, to use these values outside of the function, you need to assign them to variables. 
#This is why you "unpack" them: classifier, scaler, model_accuracy = train_model()
# Without unpacking, you wouldn't be able to directly use these returned values.

#If you didn't unpack the values, you would have to handle them as a tuple, like this:
#result = train_model()
#classifier = result[0]
#scaler = result[1]
#model_accuracy = result[2]


# Title and description
st.title("Vehicle Purchase Prediction App")
st.subheader(f"Our Trained Model Accuracy: {model_accuracy * 100:.2f}%")
st.write("""
This app uses a logistic regression model to predict whether a person will buy a vehicle based on their age and estimated salary.
Your CSV file should contain the following columns:
- **User ID**: A unique identifier for each record (will not be used in prediction).
- **Gender**: The gender of the person (this column will be ignored for prediction).
- **Age**: Numerical value representing the person's age.
- **EstimatedSalary**: Numerical value representing the person's estimated salary.

Example of the CSV file format:
| User ID | Gender | Age | EstimatedSalary |
|---------|--------|-----|-----------------|
| 15724611| Male   | 45  | 60000           |
| 15725621| Female | 79  | 64000           |
| 15725622| Male   | 23  | 78000           |
| 15720611| Female | 34  | 45000           |

Upload a CSV file with these fields for prediction.
""")

# Sidebar for the video
st.sidebar.header("Vehicle Overview")
video_file_path = r"D:\Naresh i Class\Sept 2024\25 Sep 24\Future Prediction Project\CARR.mp4"

# Open the video file and encode it in base64
with open(video_file_path, 'rb') as video_file:
    video_bytes = video_file.read()
    video_base64 = base64.b64encode(video_bytes).decode('utf-8')

# Embed the video in the sidebar with custom height and width using HTML, autoplay, and muted
st.sidebar.markdown(
    f"""
    <video width="250" height="200" autoplay muted loop>
        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    """,
    unsafe_allow_html=True
)




# Main area for uploading the CSV file
st.header("Upload Your CSV File")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])


# Prediction on uploaded file
if uploaded_file is not None:
    # Read the uploaded file
    input_data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Dataset")
    st.write(input_data.head())

    # Extract the features for prediction (assuming Age and Estimated Salary columns)
    X_new = input_data.iloc[:, [2, 3]].values

    # Apply feature scaling using the trained scaler
    X_new_scaled = scaler.transform(X_new)

    # Make predictions
    predictions = classifier.predict(X_new_scaled)

    # Add predictions to the DataFrame
    input_data['Prediction'] = predictions
    input_data['Prediction'] = input_data['Prediction'].map({0: 'Not Buy', 1: 'Buy'})  # Map to more meaningful labels

    # Display the predictions
    st.write("### Predictions")
    st.write(input_data)

    # Provide a download button for the results
    csv = input_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name='vehicle_purchase_predictions.csv',
        mime='text/csv',
    )

# Footer
st.write("""
*Developed by Hina Parashar*  
""")
