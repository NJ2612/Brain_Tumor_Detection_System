
# README

## Dataset Installation
The dataset used in this project is the BrATS2021 dataset. This dataset can be downloaded using the following code:

```python
import kagglehub
path = kagglehub.dataset_download("dschettler8845/brats-2021-task1")
print("Path to dataset files:", path)
```

## Overview
This project allows you to train a machine learning model using the `model.py` script and then deploy it using a Flask application through the `app.py` file. Below are detailed instructions to help you get started.

## Prerequisites

1. **Python**: Ensure you have Python (>= 3.7) installed on your system.
2. **Virtual Environment**: Install `virtualenv` if not already installed.

   ```bash
   pip install virtualenv
   ```

3. **Dependencies**: The required Python libraries are listed in `requirements.txt`.

   Ensure you have the `requirements.txt` file downloaded from the repository. If not, you can fetch it manually using:

   ```bash
   curl -O <repository_url>/requirements.txt
   ```

## Setup Instructions

### Step 1: Clone the Repository

```bash
git clone <repository_url>
cd <repository_name>
```

### Step 2: Set Up a Virtual Environment

1. Create a virtual environment:

   ```bash
   virtualenv venv
   ```

2. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Step 3: Train the Model

Run the `model.py` script to train the model:

```bash
python model.py
```

This will process the dataset, train the model, and save it to the specified location (as defined in the script).

### Step 4: Run the Flask Application

Start the Flask application to serve the model:

```bash
python app.py
```

By default, the app will run on `http://127.0.0.1:5000/`.

### Step 5: Access the Application

Open your web browser and navigate to:

```
http://127.0.0.1:5000/
```

From here, you can interact with the application to make predictions or perform other functionalities provided by the Flask app.

## Notes

- Ensure the dataset required for training is available and correctly specified in `model.py`.
- Modify the configurations (e.g., model save path, dataset location, app settings) in the scripts as needed.
- To deactivate the virtual environment, use:

  ```bash
  deactivate
  ```

## Troubleshooting

- If you encounter missing module errors, ensure all dependencies are correctly installed using `pip install -r requirements.txt`.
- For any Flask-related issues, verify that the `FLASK_APP` environment variable is set to `app.py` if necessary.

```bash
export FLASK_APP=app.py
```

Enjoy using the project!
