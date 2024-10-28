# Agricultural Chatbot using BERT and TensorFlow

A specialized chatbot for agricultural consultation using BERT embeddings and deep learning to provide accurate responses about crop diseases and farming practices.

## Overview

This chatbot combines modern NLP techniques with deep learning to understand and respond to farming-related queries. It uses BERT embeddings for enhanced natural language understanding and a neural network for intent classification.

## Features

- BERT-based text embeddings for improved language understanding
- Custom neural network for intent classification
- Agricultural domain-specific responses
- Interactive interface using Gradio
- Trained on specialized crop disease and farming practice dataset

## Requirements

```text
tensorflow-intel==2.16.2
tensorboard==2.16.2
torch==2.2.2
nltk
transformers
gradio
scikit-learn
pandas
numpy
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd chatbot
```

2. Create and activate virtual environment:
```bash
python -m venv env
source env/Scripts/activate  # For Windows
source env/bin/activate      # For Unix/MacOS
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

The chatbot is trained on a custom agricultural dataset which I made myself from looking at various questions only (`crop_diseases.csv`) containing:
- Questions about crop diseases
- Agricultural practices
- Pest management
- Soil health
- Organic farming

## Model Architecture

- Input Layer: BERT embeddings (768 dimensions)
- Hidden Layer 1: 128 neurons with ReLU activation
- Dropout Layer 1: 0.2
- Hidden Layer 2: 64 neurons with ReLU activation
- Dropout Layer 2: 0.2
- Output Layer: Softmax activation

## Performance Metrics

- Training Accuracy: ~95%
- Precision: ~92%
- Recall: ~90%

## Usage

Run the chatbot interface:
```bash
python chatbot.py
```

## Example Conversations

```text
User: What is crop rotation?
Bot: Crop rotation is the practice of growing different types of crops in the same area in sequential seasons to improve soil health and reduce pests.

User: How can I prevent powdery mildew?
Bot: Powdery mildew can be prevented by ensuring good air circulation, avoiding overhead watering, maintaining proper plant spacing, and using resistant varieties when available.
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- BERT model from Hugging Face Transformers
- TensorFlow team for the deep learning framework
- Agricultural domain experts for dataset validation
