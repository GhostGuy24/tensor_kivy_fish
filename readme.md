# 📷 Kivy Image Classifier with User Feedback

This project is a **Kivy-based image classification application** that uses a **pre-trained Keras model** to predict image classes. Users can provide feedback on predictions using **Yes/No buttons**, helping to improve the model over time.

---

## 🚀 Features

✅ **Image Upload** - Select an image from your file system using a graphical file chooser.
✅ **ML Prediction** - The app uses a Keras model (`keras_Model.h5`) to classify images.
✅ **User Feedback** - Users can confirm or correct predictions using **Yes/No** buttons.
✅ **Data Logging for Retraining** - If the prediction is incorrect, the user can enter the correct class, which is stored in `corrections.txt` for future model retraining.

---

## 🛠️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/GhostGuy24/tensor_kivy_fish
cd tensor_kivy_fish
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the App
```bash
python app.py
```

---

## 🖼️ How It Works

1️⃣ **Run the app** - Opens a graphical interface.
2️⃣ **Click 'Browse Image'** - Select an image file.
3️⃣ **View Prediction** - The app displays the predicted class and confidence score.
4️⃣ **Provide Feedback**:
   - ✅ Click **Yes** if the prediction is correct.
   - ❌ Click **No** if incorrect, enter the correct class, and press **Enter**.
5️⃣ **Corrections Saved** - Incorrect predictions are logged in `corrections.txt`.

---

## 📂 File Structure
```
📁 kivy-image-classifier
│-- app.py                # Main application script
│-- keras_Model.h5        # Pre-trained Keras model
│-- labels.txt            # Class labels
│-- corrections.txt       # Logs incorrect predictions for retraining
│-- requirements.txt      # Required Python packages
```

---

## 🔄 Retraining the Model

To improve the model, use the data from `corrections.txt` to retrain your Keras model. Here’s an example approach:
```python
import pandas as pd
from tensorflow.keras.models import load_model

# Load corrections
corrections = pd.read_csv("corrections.txt", header=None, names=["image_path", "correct_label"])

# Load and fine-tune your model
model = load_model("keras_Model.h5")
# Add retraining logic here...
```

Would you like help implementing automated retraining? Let me know! 🚀

