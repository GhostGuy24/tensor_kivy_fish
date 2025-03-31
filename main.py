from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.textinput import TextInput
from keras.models import load_model  
from PIL import Image as PILImage, ImageOps  
import numpy as np
import os

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

class FileBrowserApp(App):
    def build(self):
        self.layout = BoxLayout(orientation="vertical")

        # Browse button
        self.browse_button = Button(text="Browse Image", size_hint_y=0.1)
        self.browse_button.bind(on_press=self.open_file_chooser)
        
        # Image display
        self.img_display = Image(size_hint_y=0.6)

        # Prediction label
        self.prediction_label = Button(text="Prediction will appear here", size_hint_y=0.1)

        # Yes/No Buttons
        self.yes_button = Button(text="Yes", size_hint_y=0.1, disabled=True)
        self.no_button = Button(text="No", size_hint_y=0.1, disabled=True)

        self.yes_button.bind(on_press=self.confirm_prediction)
        self.no_button.bind(on_press=self.correct_prediction)

        # Input for correction (hidden initially)
        self.correct_input = TextInput(hint_text="Enter correct class", size_hint_y=0.1, multiline=False, opacity=0)
        self.correct_input.bind(on_text_validate=self.save_correction)

        # Add widgets to layout
        self.layout.add_widget(self.browse_button)
        self.layout.add_widget(self.img_display)
        self.layout.add_widget(self.prediction_label)
        self.layout.add_widget(self.yes_button)
        self.layout.add_widget(self.no_button)
        self.layout.add_widget(self.correct_input)

        return self.layout

    def open_file_chooser(self, instance):
        """Open file chooser to select an image"""
        self.file_chooser = FileChooserIconView(filters=["*.png", "*.jpg", "*.jpeg"])
        self.file_chooser.bind(on_submit=self.load_image)

        # Replace layout with file chooser
        self.layout.clear_widgets()
        self.layout.add_widget(self.file_chooser)

    def load_image(self, chooser, selection, touch):
        """Load and display selected image"""
        if selection:
            self.image_path = selection[0]
            self.img_display.source = self.image_path  # Set image source

            # Predict the image class
            self.prediction, self.confidence_score = self.predict_image(self.image_path)

            # Update the prediction label
            self.prediction_label.text = f"Class: {self.prediction}\nConfidence: {self.confidence_score:.4f}"

            # Enable Yes/No buttons
            self.yes_button.disabled = False
            self.no_button.disabled = False

        # Restore the layout
        self.layout.clear_widgets()
        self.layout.add_widget(self.browse_button)
        self.layout.add_widget(self.img_display)
        self.layout.add_widget(self.prediction_label)
        self.layout.add_widget(self.yes_button)
        self.layout.add_widget(self.no_button)
        self.layout.add_widget(self.correct_input)

    def predict_image(self, image_path):
        """Predict the image using the Keras model"""
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # Load and preprocess the image
        image = PILImage.open(image_path).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, PILImage.Resampling.LANCZOS)
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Load into array
        data[0] = normalized_image_array

        # Predict
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]

        return class_name, confidence_score

    def confirm_prediction(self, instance):
        """User confirms that the prediction is correct"""
        self.prediction_label.text = "✅ Prediction confirmed!"
        self.yes_button.disabled = True
        self.no_button.disabled = True

    def correct_prediction(self, instance):
        """User corrects the prediction"""
        self.prediction_label.text = "Please enter the correct class:"
        self.correct_input.opacity = 1  # Show input field
        self.correct_input.focus = True

    def save_correction(self, instance):
        """Save the corrected label for retraining"""
        correct_class = self.correct_input.text.strip()

        if correct_class:
            with open("corrections.txt", "a") as f:
                f.write(f"{self.image_path},{correct_class}\n")

            self.prediction_label.text = f"Correction saved: {correct_class}"
        
        # Hide correction input again
        self.correct_input.opacity = 0
        self.correct_input.text = ""

        # Disable Yes/No buttons
        self.yes_button.disabled = True
        self.no_button.disabled = True

if __name__ == "__main__":
    FileBrowserApp().run()
