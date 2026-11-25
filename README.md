# Text Emotion Detection

A reproducible pipeline for classifying emotional states in natural language text. 

## Overview
Standard sentiment analysis often reduces text to binary polarities. This project implements a multi-class emotion detection model that maps unstructured text to discrete emotional categories (e.g., joy, sadness, anger, fear, surprise), providing a more granular understanding of user intent and state.

## Features
- **End-to-End Pipeline**: Handles tokenization, preprocessing, and inference.
- **Lightweight**: Built for straightforward empirical testing and easy integration.
- **Extensible**: Designed so you can easily swap out the base model, tokenizer, or dataset.

## Quickstart

### Prerequisites
- Python 3.9+

```bash
git clone https://github.com/MahdiAlikhah/Text-Emotion-Detection.git
cd Text-Emotion-Detection
pip install -r requirements.txt
```

### Usage
Basic inference example:

```python
from emotion_detector import EmotionModel # (Update with your actual import path)

model = EmotionModel()
text = "I'm genuinely surprised by how well this performs out of the box."

prediction = model.predict(text)
print(prediction) 
# Expected Output: {'emotion': 'surprise', 'confidence': 0.94}
```

## Dataset & Training
*(Add a brief note here about how you trained it. For example: "Fine-tuned on the GoEmotions dataset" or "Trained using a custom Random Forest classifier on labeled conversational data.")*

## Roadmap
- [ ] Add batch inference support
- [ ] Export to ONNX for faster CPU execution
- [ ] Expand language support (potential integration with [rishe](https://github.com/MahdiAlikhah/rishe) for Persian text)

## License
MIT
