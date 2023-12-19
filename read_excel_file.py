import pandas as pd
def read_excel_dataset(file_path):
    try:
        # Reading the Excel dataset using Pandas
        dataset = pd.read_excel(file_path)
        print("Dataset loaded successfully!")
        return dataset
    except FileNotFoundError:
        print("File not found. Please make sure the file path is correct.")
    except Exception as e:
        print(f"An error occurred: {e}")

