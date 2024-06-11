from transformers import pipeline
import pandas as pd
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt


# Ignore all runtime warnings
warnings.filterwarnings("ignore")

# Path to the Excel file  # Excel file name
excel_file_path = f'C:/Users/sdany/Desktop/pilot project/pilot50.xlsx'

def change_labels(labels):
    label_map = {
        'Propulsion, motor, thruster, propplant': 'propulsion', 
        'Satellite payload': 'Payload', 
        'Guidance, navigation, control, Attitude Determination': 'ACDS & GNC', 
        'System & Integration': 'System & Integration', 
        'Software': 'Software', 
        'Telecommunications': 'Communications', 
        'Sensors': 'Sensors',
        'Structure, Material & Mechanics': 'Mechanics',  # Updated label name
        'Command and Data Handling': 'C & DH',  # Updated label name
        'Power, energy': 'Power', 
        'Telemetry, Tracking and Command': 'TT & C'  # Updated label name
    }
    return label_map.get(labels, labels)



# Function to preprocess text
def preprocess_text(text):
    if not isinstance(text, str):  # Ensure the input is a string
        text = str(text)
    text = text.lower()  # Convert text to lowercase
    # Add more preprocessing steps as needed
    return text

# Define the task and model ID
task = "zero-shot-classification"
modelId = "facebook/bart-large-mnli"

# Read the Excel file into a DataFrame
df = pd.read_excel(excel_file_path)
# Preprocess the text in the 'Abstract' column
df['Abstract'] = df['Abstract'].apply(preprocess_text)

# Create the classifier pipeline
classifier = pipeline(task=task, model=modelId)
# Create the summarizer pipeline
summarizer = pipeline(task="summarization", model="sshleifer/distilbart-cnn-12-6")

# Define the candidate labels for classification
candidate_labels = [
    'Propulsion, motor, thruster, propplant', 
    'Satellite payload', 
    'Guidance, navigation, control, Attitude Determination',
    'System & Integration',
    'Software', 
    'Telecommunications', 
    'Sensors',
    'Structure, Material & Mechanics',
    'Command and Data Handling', 
    'Power, energy', 
    'Telemetry, Tracking and Command', 
]


# Summarize the text in the 'Abstract' column with a progress bar
results_summarization = []
for abstract in tqdm(df['Abstract'], desc="Summarizing"):
    result = summarizer(abstract, max_length=200, min_length=80, do_sample=True)
    results_summarization.append(result)

# Add the summarization results to the DataFrame
df['Summarization'] = [result[0]['summary_text'] for result in results_summarization]


# Classify the text in the 'Abstract' column with a progress bar
results = []
predicted_labels = []
process_column='Abstract'
for abstract in tqdm(df[process_column], desc="Classifying"):
    result = classifier(abstract, candidate_labels)
    results.append(result)
    # Find the label with the highest score
    max_score_index = result['scores'].index(max(result['scores']))
    predicted_labels.append(result['labels'][max_score_index])

# Add the classification results to the DataFrame
for label in candidate_labels:
    df[change_labels(label)] = [result['scores'][result['labels'].index(label)] if label in result['labels'] else 0 for result in results]


# Add the predicted label to the DataFrame
df['Predicted Label'] = [change_labels(label) for label in predicted_labels]



# Path to save the new Excel file
output_file_path = f'C:/Users/sdany/Desktop/pilot project/output_{process_column}.xlsx'

# Save the DataFrame with the classification results to a new Excel file
df.to_excel(output_file_path, index=False)

print(f"Output file path: {output_file_path}")