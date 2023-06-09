import cv2
from PIL import Image
import numpy as np
from microdot import Microdot, Response
from scipy import signal
import base64
from io import BytesIO

app = Microdot()
Response.default_content_type = 'text/html'

kernels = {
    "Identity": np.array([[0,0,0],[0,1,0],[0,0,0]]),
    "Outline": np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]),
    "Blur": np.array([[0.0625,0.125,0.0625],[0.125,0.25,0.125],[0.0625,0.125,0.0625]]),
    "Emboss": np.array([[-2,-1,0],[-1,1,1],[0,1,2]]),
    "Sharpen": np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]),
    "Random": np.random.randn(3,3)
}

def apply_kernel(img, kernel):
    convolved = signal.convolve2d(img, kernel, mode='same', boundary='fill', fillvalue=0)
    convolved = np.clip(convolved, 0, 255)  # ensure values are within the valid range for uint8
    convolved_img = Image.fromarray(convolved.astype(np.uint8))
    return convolved_img

def image_to_data_url(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return "data:image/jpeg;base64," + img_str.decode()

@app.route("/")
def index(request):
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return "Cannot open webcam"

    # Capture a single frame
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "Can't receive frame (stream end?). Exiting ..."

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert to PIL image
    gray_img = Image.fromarray(gray_frame)

    kernel_images = {k: apply_kernel(gray_img, v) for k, v in kernels.items()}
    img_elements = [f'<div><h3>{k}</h3><img src="{image_to_data_url(v)}" alt="{k}"></div>' for k, v in kernel_images.items()]

    html_content = f"""
            <!DOCTYPE html>
            <html>
                <head>
                <style>
                    body {{
                    font-family: Arial, sans-serif;
                    }}
                    .image-grid {{
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 20px;
                    justify-items: center;
                    align-items: center;
                    padding: 20px;
                    }}
                    .image-grid div {{
                    box-shadow: 0px 0px 10px rgba(0, 0, 0,
                                        }}
                    .image-grid img {{
                        max-width: 100%;
                        height: auto;
                    }}
                </style>
                </head>
                <body>
                    <h2>Image Filters</h2>
                    <div class="image-grid">
                        {''.join(img_elements)}
                    </div>
                </body>
            </html>
            """
    return html_content

app.run(debug=True, port=8008)