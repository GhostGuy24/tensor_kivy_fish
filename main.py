from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.filechooser import FileChooserIconView
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image as PILImage, ImageOps  # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

class FileBrowserApp(App):
    def build(self):
        self.layout = BoxLayout(orientation="vertical")

        # Button to open file chooser
        self.browse_button = Button(text="Browse Image", size_hint_y=0.1)
        self.browse_button.bind(on_press=self.open_file_chooser)
        
        # Image widget to display selected image
        self.img_display = Image(size_hint_y=0.8)

        # Label to display prediction results
        self.prediction_label = Button(text="Prediction will appear here", size_hint_y=0.1)

        self.layout.add_widget(self.browse_button)
        self.layout.add_widget(self.img_display)
        self.layout.add_widget(self.prediction_label)

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
            image_path = selection[0]
            self.img_display.source = image_path  # Set image source

            # Predict the image class
            prediction, confidence_score = self.predict_image(image_path)

            # Update the prediction label with class name and confidence score
            self.prediction_label.text = f"Class: {prediction}\nConfidence Score: {confidence_score:.4f}"

        # Restore the main layout with the image and prediction
        self.layout.clear_widgets()
        self.layout.add_widget(self.browse_button)
        self.layout.add_widget(self.img_display)
        self.layout.add_widget(self.prediction_label)

    def predict_image(self, image_path):
        """Predict the image using the Keras model"""
        # Create the array of the right shape to feed into the keras model
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # Load and preprocess the image
        image = PILImage.open(image_path).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, PILImage.Resampling.LANCZOS)
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # Predict the model
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()  # Strip newline or extra spaces
        confidence_score = prediction[0][index]

        return class_name, confidence_score

if __name__ == "__main__":
    FileBrowserApp().run()
