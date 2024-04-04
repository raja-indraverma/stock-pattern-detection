from ultralyticsplus import YOLO
import cv2
from selenium import webdriver
import numpy as np
import io
from PIL import Image
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import winsound  # For Windows systems, to play sound notifications
from google.auth import exceptions
from google.oauth2 import service_account
from twilio.rest import Client

# Load the YOLOv8 model
model = YOLO('foduucom/stockmarket-pattern-detection-yolov8')

# Set model parameters
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45   # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # Maximum number of detections per image

# Initialize the web browser (Chrome)
driver = webdriver.Chrome()

# Navigate to the desired URL
driver.get("https://www.tradingview.com/chart/GNabfRZZ/?symbol=FX%3AEURUSD")  # Replace "https://example.com" with the URL of the desired webpage

# Twilio credentials
twilio_account_sid = 'your sid'
twilio_auth_token = 'your token'
twilio_whatsapp_number = '+14155238886.'  # Your Twilio WhatsApp number
recipient_whatsapp_number = '+918076935422'  # Recipient's WhatsApp number

# Initialize Twilio client
client = Client(twilio_account_sid, twilio_auth_token)
# Flag to track whether notification has been sent
notification_sent = False

# Function to send WhatsApp notification with image
def send_whatsapp_notification(image_path):
    try:
        message = client.messages.create(
            body='Pattern detected! ',
            from_=twilio_whatsapp_number,
            to=recipient_whatsapp_number,
            # media_url=[image_path]  # Attach the image as media
        )
        print("WhatsApp notification sent successfully.")
    except Exception as e:
        print("Error sending WhatsApp notification:", str(e))

cv2.namedWindow("YOLOv8 Inference", cv2.WINDOW_NORMAL)

while True:
    # Capture a screenshot of the entire browser window as a byte array
    screenshot_bytearray = driver.get_screenshot_as_png()

    # Convert the byte array to a PIL Image object
    screenshot_pil = Image.open(io.BytesIO(screenshot_bytearray))

    # Convert the PIL Image to RGB format
    screenshot_rgb = screenshot_pil.convert('RGB')

    # Convert the RGB image to a numpy array
    screenshot_np = np.array(screenshot_rgb)

    # Run YOLOv8 inference on the screenshot
    # results = model(screenshot_np)

    results = model.predict(screenshot_np)

    # Visualize the results on the screenshot
    annotated_frame = results[0].plot()



    # for result in results:
    #             boxes = result.boxes
    #             if len(boxes) != 0 and not notification_sent:
    #                     print("detected")
    #                     # Send email notification with screenshot
    #                     send_whatsapp_notification(annotated_frame)
    #                     # Play sound notification (you may need to adjust the path)
    #                     winsound.PlaySound("SystemHand", winsound.SND_ALIAS)
    #                     # You can replace "SystemHand" with the path to your custom sound file
                       

    res = model.predict(annotated_frame)
    for result in res:
                boxes = result.boxes
                if len(boxes) != 0:
                    print("no detections")

    # Display the annotated frame
    cv2.imshow("YOLOv8 Inference", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Close the web browser
driver.quit()

# Close the display window
cv2.destroyAllWindows()
