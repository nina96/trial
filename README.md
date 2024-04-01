# trial

##DMS

# libraries
import argparse
import cv2
from pyzbar.pyzbar import decode
import PyPDF2
from PIL import Image
import io
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")


# MODULE 1 reading pdf and converting them into images for processing:  file_path, output_path -> images 

def extract_images_from_pdf(file_path, output_file): #taking input file path
    
        images = [] #making array of images made
        pdf_reader = PyPDF2.PdfReader(file_path, output_file) #reading pdf pages 
        for page_num in range(len(pdf_reader.pages)): #going through whole pdf pages and saving images in array
            page = pdf_reader.pages[page_num]
            xObject = page['/Resources']['/XObject'].get_object()
            for obj in xObject:
                if xObject[obj]['/Subtype'] == '/Image':
                    img_data = xObject[obj].get_object()
                    img_bytes = io.BytesIO(img_data.get_data())
                    image = Image.open(img_bytes) #converting byte stream to Pillow image format
                    images.append(image) #appending image info in an array

                    #image.save(f"{output_file}/image_page4_{page_num+1}.png", "PNG") to save images as png. experimentally
                    #print(image.mode) to check mode of image
        
                   
        return images








#MODULE 2: conveting image to cv2 src using np and returning appropriate grayscale image for further processing: Image-> np.ndarray (grayscale image)

def convert_from_image_to_cv2(img: Image) -> np.ndarray: 
    if img.mode=='1':                       #for binary image(0 and 1) we can just convert it by multiplying values by 255 and typecast it into array
        return np.array((img*255))

    elif img.mode=='L':               #for grayscale image(0 -255) just typecast
        return np.array(img)
    else:
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)     #for RGB image read image as array and convert to grayscale







#MODULE 3 : extracting barcode from given images and save the value and return it along with page number: image(array value) -> barcodes(list), pagenumber(list)

#extracting barcodes from given images
def read_barcodes_from_images(extracted_images): 
    barcodes = [] #to save barcode values
    page_number=[] # to save page number in which values are found

    
    for i, image in enumerate(extracted_images):     #looping through images 
            if image.format=='JPEG':                 #checking image.format for processing TIFF images not processed
                img= convert_from_image_to_cv2(image)
                _, thresh= cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #thresholding grayscale image for more clear image


                for barcode in decode(thresh):   #using pyzbar for decode purpose
                    
                    if barcode:
                        barcode_data = barcode.data.decode('utf-8')  #getting only data of barcode
                        if barcode_data:
                            barcodes.append(barcode_data)
                            page_number.append(i+1)
                            # Store barcode data and page number
                #print(f"{barcodes} found in page number: {i}")      #experimental
            else:
                print(f"TIFF file detected skipped page {i+1}")
                continue

    return barcodes, page_number









# MODULE 4: for splitting the pdf according to the barcode and saving the splits with barcode value as name.pdf: pdf_file(path), barcodes(list), page_numbers(list), output_file(path)-> generated_file(str)

#demerger for pdf files 

def split_pdf_by_barcodes(pdf_file,barcodes, page_nums,output_file):

    pdf_reader = PyPDF2.PdfReader(pdf_file)
    num_pages = len(pdf_reader.pages)
    page_ranges = []
    generated_filename=[]
    
    for i, (barcode, page_num) in enumerate(zip(barcodes, page_nums)): #looping through barcodes,page_nums along with index
        start_page = page_num-1
        if i < len(barcodes) - 1:
            end_page = page_nums[i+1] - 1
        else:
            end_page = num_pages
        page_ranges.append((start_page, end_page))
        generated_filename.append(f"{barcode}")

    #print(f"{generated_filename} page range{page_ranges}") experimental checking of generated files along with the range of pages
    output_folder = output_file   # Specify the folder name
    ##print(output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

    for i, (start_page, end_page) in enumerate(page_ranges):
        output_pdf = os.path.join(output_folder,f"{barcodes[i]}.pdf")  # Use barcode data as the file name
        writer = PyPDF2.PdfWriter()
        for page_num in range(start_page, end_page):
            writer.add_page(pdf_reader.pages[page_num])
        with open(output_pdf, "wb") as output_file:
            writer.write(output_file)
   
    return generated_filename



    



#MAIN PROGRAM
def main():
    parser = argparse.ArgumentParser(description='Split PDF based on barcode data.')
    parser.add_argument('input_pdf', type=str, help='The path to the input PDF file.')
    parser.add_argument('output_folder', type=str, help='The path to the output folder.')

    args = parser.parse_args()
    pdf_dir = args.input_pdf
    output_file1 = args.output_folder

    print(os.listdir(pdf_dir))
    for path_file in os.listdir(pdf_dir):
        p_file= os.path.join(pdf_dir,path_file)
        print(p_file)
        extracted_images= extract_images_from_pdf(p_file,output_file1)

        barcodes, pages= read_barcodes_from_images(extracted_images)

        if barcodes:
            generated_files= split_pdf_by_barcodes(p_file,barcodes,pages,output_file1)
            print("PDF split successfully generated")
            for i in generated_files:
                print(f"{i} successfully generated")

        else:
            print("No barcode in file")

        os.remove(p_file)

    


if __name__ == '__main__':
    main()





### interview


import cv2 as cv
import argparse
import numpy as np

from utils.login import login_user
from utils.register import register_user
from utils.authenticate import periodic_authentication
from utils.common import load_known_faces
from utils import logger
import threading



def get_args():
     parser = argparse.ArgumentParser()
     parser.add_argument("--device", type=int, default=0)
     parser.add_argument("--width", help='cap width', type=int, default=960)
     parser.add_argument("--height", help='cap height', type=int, default=540)

     args = parser.parse_args()

     return args


def main():
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    
  
 
    cap = cv.VideoCapture(cap_device)
    known_faces= load_known_faces()

    if not cap.isOpened():
         print("Video capture failed camera not open")
         return
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    user_action = input("Do you want to [login] or [register]? ")

    cv2.putText()

    if user_action.lower() == "register":
        register_user(cap)

    elif user_action.lower() == "login":
        login_user(cap)
        threading.Thread(target=periodic_authentication, args=(known_faces,), daemon=True).start()
    else:
        print("Invalid action.")

    cap.release()
    cv.destroyAllWindows()
                    
                
if __name__ == '__main__':
    main()


import time
import random
import face_recognition
import numpy as np
#from .utils  import logger
import  mediapipe as mp
from mediapipe .tasks import python

mpPose= mp.solutions.pose
mpDraw= mp.solutions.drawing_utils
pose= mpPose.Pose()
PoseLandmarker = mp.tasks.vision.PoseLandmarker

model_path = 'pose_landmarker_lite.task'
VisionRunningMode = mp.tasks.vision.RunningMode
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult

def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('pose landmarker result: {}'.format(result))

options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='pose_landmarker_lite.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_poses= 3,
    result_callback=print_result)
detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)



def periodic_authentication(known_faces):
    try:
        while True:
            wait_time = random.randint(5, 15)  # Random time between 5 and 15 seconds
            time.sleep(wait_time)
            if not recognize_person(known_faces):
                print("Authentication failed. Please ensure the registered user is present and alone.")
            # Here you can add any additional actions to take if authentication fails
    except Exception as e:
        #logger.info(str(e))
        pass
    
def recognize_person(frame, known_data):
    timestamp = int(time.time() * 1e6)
    #print("recognizing person")
    try:
        face_encodings = face_recognition.face_encodings(frame)
        single= face_recognition.face_locations(frame)
        with PoseLandmarker.create_from_options(options) as landmarker:
            results= landmarker.detect_async(frame,timestamp)
            print(results)

            if results.pose_landmarks:
                mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            
            if len(single)==1:
                if face_encodings:
                    face_encoding = face_encodings[0]
                    for name, encoding in known_data.items():
                        matches = face_recognition.compare_faces([np.array(encoding)], face_encoding)
                        #print(matches)
                        if True in matches:
                            return name
                    
                    return "Unknown"
            elif len(single)>1:
                print("please stand alone otherwise process will invalidate") 
        
    except Exception as e:
        pass
        # logger.info(str(e))


###common

import cv2 as cv
import face_recognition
import numpy as np
import json
from flip.utils import logger
from skimage.exposure import is_low_contrast
import cv2



def detect_single_face(image):

    try:
        #print("inside detect_single_face")

        image_rgb=cv. cvtColor(image, cv.COLOR_BGR2RGB)
        is_low= is_low_contrast(image_rgb)
        print(f"is_low: {is_low}")
        if is_low:
            print("low light picture please take again")
            return False

        else:
            results= face_recognition.face_locations(image_rgb)
            upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
            full_body_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            upper_bodies = upper_body_cascade.detectMultiScale(gray_image, scaleFactor=1.5, minNeighbors=3)
            full_bodies= full_body_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=3)
            print(f"{len(results)} , upper body : {len(upper_bodies)}, full body : {len(full_bodies)}")


        if len(results)==1:
            return True, results
        elif len(results)>1 or len(upper_bodies)>1: 
            print("Please make sure you are alone while registration and try again")
            return False
        else :
            return False
    except Exception as e:
        logger.info(str(e))
        print(e)
def is_face_already_registered(new_encoding, known_data):
    try:
        if not known_data:
            return False, None  # No known data to compare against
        
        for name, encoding in known_data.items():
            known_encoding = np.array(encoding)
            distance = face_recognition.face_distance([known_encoding], new_encoding)
            if np.any(distance <= 0.6):  # assuming 0.6 as threshold for face similarity
                return True, name
        return False, None       
    except Exception as e:
        logger.info(str(e))
        print(e)

def load_known_faces():
    json_file_path = "candidates.json"
    try:
        with open(json_file_path, 'r') as file:
            known_data = json.load(file)
            return known_data
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
 ###login

 import json
from utils.common import detect_single_face, load_known_faces
import threading
from utils.authenticate import recognize_person
from utils import logger
import cv2 as cv
import time





def login_user(cap):
    try:
        known_data = load_known_faces()
        pTime=0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break
            cTime=time.time()
            fps=1/(cTime-pTime)
            pTime= cTime
            
            success= detect_single_face(frame)
            if success and known_data is not None:
                user_name = recognize_person(frame, known_data)
                #print(user_name)
                if user_name != "Unknown":
                    print(f"Welcome back, {user_name}!")
                    #break
                else:
                    print("Face not recognized. Please try again or register again.")
                    #break
            cv.putText(frame, str(int(fps)),(70,30), cv.FONT_HERSHEY_COMPLEX, 3,(255,0,0),1)
            cv.imshow('Login', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        logger.info(str(e))


        #### register


        from utils.common import load_known_faces, detect_single_face, is_face_already_registered
import face_recognition
import cv2 as cv
import json
from utils import logger





def register_user(cap):
    try:
        known_data = load_known_faces()
        print("please enter c to capture image or q to exit, keep caps off")
        while True:
            ret, frame= cap.read()
            if not ret:
                print("Failed to capture frame.")
                break
            cv.imshow('Register', frame)
            #print("please enter c to capture image or q to exit, keep caps off")
            if cv.waitKey(1) & 0xFF == ord('c'):
                
                success, face_locations= detect_single_face(frame)

                if success:
                    name = input("Enter your name: ")
                    new_user_encoding = face_recognition.face_encodings(frame, face_locations)[0]
                    is_registered, registered_name = is_face_already_registered(new_user_encoding, known_data)

                    if is_registered:
                        print(f"Face already registered under the name {registered_name}. Please log in.")
                        break

                    add_new_user(name, frame, known_data)
                    break
                else:
                    print("Failed to detect a single face.")
            elif cv.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        logger.info(str(e))

def add_new_user(name, frame, known_data):
    try:
        json_file_path = "candidates.json"
        new_user_encoding = face_recognition.face_encodings(frame)[0].tolist()
        
        known_data[name] = new_user_encoding

        with open(json_file_path, 'w') as file:
            json.dump(known_data, file)
        print(f"New user {name} added successfully.")

    except Exception as e:
        logger.info(str(e))
