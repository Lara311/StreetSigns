from fastapi import FastAPI, UploadFile, File
from PIL import Image
import numpy as np
import onnxruntime as ort
import io

app = FastAPI()

# Load the ONNX model
onnx_model_path = 'yolov5s.onnx'
session = ort.InferenceSession(onnx_model_path)

def preprocess_image(image: Image.Image):
    # Resize the image to 640x640 (expected input size for YOLOv5 model)
    image = image.resize((640, 640))
    
    # Convert image to numpy array and scale pixel values to [0, 1]
    image_np = np.array(image).astype(np.float32) / 255.0
    
    # Transpose image to match model input shape (C, H, W)
    image_np = np.transpose(image_np, [2, 0, 1])
    
    # Add batch dimension (now the shape is 1, 3, 640, 640 for one image)
    image_np = np.expand_dims(image_np, axis=0)
    
    return image_np

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        
        # Preprocess the image
        input_tensor = preprocess_image(image)
        
        # Get the model's input and output names
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Run inference
        outputs = session.run([output_name], {input_name: input_tensor})
        
        # Return the outputs (You can customize this based on what YOLOv5 returns)
        return {"outputs": outputs}
    
    except Exception as e:
        return {"error": str(e)}

@app.get("/packages")
async def list_packages():
    result = subprocess.run(['pip', 'freeze'], stdout=subprocess.PIPE)
    return {"installed_packages": result.stdout.decode('utf-8').split('\n')}
