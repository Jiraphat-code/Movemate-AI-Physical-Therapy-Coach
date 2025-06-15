# MoveMate — Your AI Rehab Companion at Home ✨

## 🚀 Welcome to MoveMate!

MoveMate is an AI-powered physical therapy assistant designed to help you perform rehabilitation exercises effectively and safely from the comfort of your home. By leveraging real-time pose estimation and machine learning, MoveMate provides instant feedback, counts repetitions, and tracks your progress to ensure proper form and optimal recovery.

## ✨ Key Features

1.  **Home Page (`1_Home.py`)**
    * An introductory section to the MoveMate project and its objectives.
    * A form for users to input personal details: Full Name, Age, and Rehabilitation Purpose. This information is saved in Streamlit's session state for personalized reports.

2.  **Processing Selection Page (`2_FeatureSelect.py`)**
    * Allows users to choose their video input source: Live Webcam feed or upload a video file (.mp4, .mov, .avi).
    * Specify the primary side of the body being exercised (Left, Right, or Both) for focused evaluation.
    * Select the Machine Learning model to be used for real-time pose classification.
    * Provides example videos of correct exercise forms to guide users.

3.  **Processing & Results Page (`3_Processing.py`)**
    * Displays real-time video feed with MediaPipe landmark visualizations and AI model predictions.
    * Features a repetition counter and a dynamic "stage" indicator for the exercise.
    * **Comprehensive Reports:** Upon completion of processing, a detailed report is generated, including:
        * User details (Name, Age, Purpose).
        * Date and time of the training session.
        * Total repetitions performed.
        * Detailed breakdown of each repetition, including frame ranges and predicted class for that rep (e.g., correct form, incorrect form type).
    * **(Planned) Incorrect Form Capture:** Future enhancements aim to automatically capture and display images of specific frames where incorrect form is detected, providing clear visual feedback for improvement.

## 🔗 Project Links

* **GitHub Repository:** [Jiraphat-code/Movemate-AI-Physical-Therapy-Coach](https://github.com/Jiraphat-code/Movemate-AI-Physical-Therapy-Coach)
* **Medium Article:** [MoveMate: Your AI-Powered Rehab Companion at Home](https://medium.com/@jiraphatpunthsang/movemate-your-ai-powered-rehab-companion-at-home-c5d39855642d)
* **Live Deployment (Streamlit Cloud):** [MoveMate AI Physical Therapy Coach](https://movemate-ai-physical-therapy-coach.streamlit.app/)

## 🛠️ Technologies Used

* **Python 3.x**
* **Streamlit:** For building the interactive web application.
* **MediaPipe:** For robust human pose estimation and landmark detection.
* **Scikit-learn:** For machine learning model implementation (RandomForestClassifier, LogisticRegression) and data preprocessing.
* **Pandas & NumPy:** For efficient data manipulation and numerical operations.
* **OpenCV (`cv2`):** For video processing, frame handling, and UI rendering.
* **Imbalanced-learn:** For handling class imbalance in the training data (e.g., SMOTE).

## 🚀 Getting Started

Follow these steps to set up and run the MoveMate application on your local machine.

### Prerequisites

* Python 3.8+ (Recommended)
* Git

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Jiraphat-code/Movemate-AI-Physical-Therapy-Coach.git](https://github.com/Jiraphat-code/Movemate-AI-Physical-Therapy-Coach.git)
    cd Movemate-AI-Physical-Therapy-Coach
    ```

2.  **Create and Activate a Virtual Environment (Highly Recommended):**
    ```bash
    python -m venv venv
    # For Windows:
    .\venv\Scripts\activate
    # For macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    All necessary Python packages are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Model & Label Encoder Files:**
    * Ensure your trained machine learning model (`.pkl` file, e.g., `movemate_no_encoding_rf.pkl`) and the `label_encoder.pkl` file are present in the `models/` directory. These files are essential for the Streamlit app to function.
    * If these files are missing or too large for direct GitHub hosting, you might need to download them separately (e.g., from a cloud storage link provided in the `training_scripts` documentation, if applicable) or use Git LFS.

### Usage

Once all installations are complete, you can launch the Streamlit application:

1.  **Ensure your virtual environment is activated.**
2.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```
    The application will automatically open in your default web browser (usually at `http://localhost:8501`).

3.  **Navigate and Interact within the App:**
    * **Home Page:** Fill in your personal details.
    * **Processing Selection Page:** Choose your video source, exercise side, and the AI model.
    * **Processing & Results Page:** Click "Start Processing" to begin the real-time analysis or video playback. A detailed report will be generated upon completion.

## 🧠 Model Training & Data Aspects

The core of MoveMate relies on a Machine Learning model trained to classify various physical therapy poses.

* **Data Collection:** Keypoint data is extracted from rehabilitation exercise videos using MediaPipe.
* **Data Preprocessing & Feature Engineering:** The raw keypoint data undergoes several preprocessing steps, including:
    * Filtering out noisy landmarks (e.g., based on visibility scores).
    * **Symmetry Augmentation:** Augmenting the training data by mirroring poses to enhance the model's generalization across different body sides.
    * **Feature Selection:** Focusing on essential X, Y coordinates of key anatomical landmarks and engineering biomechanically relevant features such as joint angles (e.g., elbow, shoulder, knee angles) and distances.
* **Model Training:** Various classification models (e.g., RandomForestClassifier, LogisticRegression) are trained on the prepared data.
* **Validation & Error Analysis:** Model performance is rigorously validated against separate datasets (including data from different users) to assess generalization capabilities and analyze specific error patterns.

Detailed scripts for data preparation and model training can be found in the `training_scripts/` directory.

## 🤝 Contributing

We welcome contributions to the MoveMate project! If you have ideas for improvements or new features, please feel free to:

1.  Fork this repository.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## ✉️ Contact

If you have any questions or feedback, please feel free to reach out:

**Jiraphat Punthsang Ultra** - jirapatulta2550@gmail.com

Project Link: [https://github.com/Jiraphat-code/Movemate-AI-Physical-Therapy-Coach](https://github.com/Jiraphat-code/Movemate-AI-Physical-Therapy-Coach)

---
Made with ❤️ by Jiraphat Punthsang.
