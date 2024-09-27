# Movie_Review_Sentiment_Analyzer

### Project Description
<p align='justify'>
This project involves the development of a Recurrent Neural Network (RNN) based deep learning model for sentiment analysis using the IMDB dataset. The model classifies movie reviews as positive or negative based on their textual content. Leveraging TensorFlow for model building and Streamlit for deployment, this application allows users to input movie reviews and receive real-time sentiment analysis.
</p>

### ScreenShot
<img width="800" height="400" align="center" src="/screenshots/screenshot 1.png">

### Technologies Used in This Project
* **Python** : Core programming language used for developing the model and application.
* **TensorFlow** : For building and training the RNN-based deep learning model.
* **Streamlit** : For deploying the model as a web application.
* **Jupyter-notebook** : For exploratory data analysis (EDA) and model development.
* **Git/Github** : For version control and hosting the project repository.

### Installation and Setup

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/Sharan-vj/Movie_Review_Sentiment_Analyzer.git
    cd Movie_Review_Sentiment_Analyzer
    ```

2. **Create a Virtual Environment**:
    ```bash
    conda create --name <env name> python=3.10 -y
    ```
3. **Activate the Virtual Enviroment**
    ```bash
    conda activate <env name>
    ```
4. **Install the Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

5. **Run the Streamlit Application**:
    ```bash
    streamlit run app.py
    ```
    The application will be available at `http://127.0.0.1:8501/`.

### Usage
1. **Enter a Movie Review**: Input a movie review into the text box in the web interface.
2. **Get Prediction**: The model will predict and display whether the review is positive or negative.

### Dataset
The dataset used for training and testing the model is the IMDB dataset, which contains 50,000 movie reviews labeled as either positive or negative. The dataset is split into training and test sets.

### Model
The model is an RNN-based deep learning model built using TensorFlow and trained to classify the sentiment of movie reviews. The trained model is saved in the model/ directory as a TensorFlow model file (IMDB_model.h5).

### Contributing
Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or suggestions.

### License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

### Acknowledgments
* Thanks to the TensorFlow team for providing the IMDB dataset and documentation.
* Special thanks to the open-source community for the libraries used in this project.