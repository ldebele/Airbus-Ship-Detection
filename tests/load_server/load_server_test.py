import io
from PIL import Image
from locust import HttpUser, task, constant


class LoadTest(HttpUser):
    wait_time = constant(0)
    host = "http://localhost:8080"

    @task
    def home(self):
        # sent a GET request to home page endpoints
        headers = {"Content-Type": "application/"}
        self.client.get("/home", headers=headers)


    @task
    def predict(self):

        # read the image
        img = Image.open('./test.jpg')

        # Convert the image to bytes and create a BytesIO stream
        img_byte_array = io.BytesIO()
        img.save(img_byte_array, format="PNG")
        img_byte_array.seek(0)

        payload = {"file": ("test_img.png", img_byte_array)}

        # send a POST request to image prediction endpoint
        self.client.post("/predict", files=payload)