from multiprocessing import Process, Queue
import numpy as np
import time, math, os, argparse, shutil, pathlib, re
from tensorflow.python.client import device_lib
from datetime import date

CLASSES = sorted(["Bear", "Coyote", "Deer", "Empty", "Other"])
MODEL_DIR = './checkpoints/imagenet.h5'
IMAGE_DIR = os.getcwd() + "/WTB_Images_03152021"
NUM_THREAD = 8
IMAGE_SIZE = (1920, 1080)
CURR_DATE = date.today().strftime("%m%d%Y")
FILENAME = "WTB_Images_Inf_report_{}.txt".format(CURR_DATE)

class Scheduler:
    def __init__(self, gpu_num):
        self._queue = Queue()
        self._gpu_num = gpu_num
        self.__init_workers()

    def __init_workers(self):
        self._workers = list()
        for gpuid in range (self._gpu_num):
            self._workers.append(Worker(gpuid, self._queue))

    def start(self, image_list):
        for img in image_list:
            self._queue.put(img)

        # Add a None to indicate the end of queue
        self._queue.put(None)

        for worker in self._workers:
            worker.start()

        for worker in self._workers:
            worker.join()
        print ("All image are done inferencing...")

class Worker(Process):
    def __init__(self, gpuid, queue):
        Process.__init__(self, name="ModelProcessor")
        self._gpuid = gpuid
        self._queue = queue
    
    def run(self):
        #set enviornment
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self._gpuid)

        from tensorflow.keras.applications.resnet50 import preprocess_input
        from tensorflow.keras.preprocessing import image
        from tensorflow.keras.models import load_model
        trained_model = load_model(MODEL_DIR)
        
        while True:
            img_path = self._queue.get()
            if img_path == None:
                self._queue.put(None)
                break
            with open(FILENAME, "a+") as f:
                img = image.load_img(path=img_path, target_size=IMAGE_SIZE)
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                y_prob = trained_model.predict(x)
                index = y_prob.argmax()
                f.write("image : {0}, class : {1}".format(img_path, CLASSES[index]))
        
        print("GPU {} has done inferencing...".format(self._gpuid))

# For the runtime with 0 GPU
def run_sequential(image_list):
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.models import load_model

    trained_model = load_model(MODEL_DIR)
    
    with open(FILENAME, "w+") as f:
        for img_path in image_list:
            img = image.load_img(path=img_path, target_size=IMAGE_SIZE)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            y_prob = trained_model.predict(x)
            index = y_prob.argmax()
            match = re.search("\/[^/]+\/[^/]+(\.jpg|\.png)", img_path)
            if match:
                img_name = match.group(0)
                f.write("image : {0}, class : {1} \n".format(img_name, CLASSES[index]))
            else:
                f.write("image : {0}, class : {1} \n".format(img_path, CLASSES[index]))

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    GPUs = [x.name for x in local_device_protos if x.device_type == 'GPU']
    return len(GPUs)

def handler(path): 
    # Get image number
    num_image = 0
    image_list = list()
    for root, dirs, files in os.walk(path):
        for img in files:
            if img.lower().endswith(".jpg") or img.lower().endswith(".png"):
                image_list.append(os.path.join(root, img))
                num_image += 1
        
    # Get GPU counts
    NUM_GPU = get_available_gpus()
    # print ("Current GPU num is {0}".format(NUM_GPU))
    
    start = time.time()

    if NUM_GPU == 0:
        run_sequential(image_list)
    else:
        # initialize Scheduler
        scheduler = Scheduler(NUM_GPU)
        # start multiprocessing
        scheduler.start(image_list)
        
    end = time.time()

    print ("Time with model loading {0} for {1} images.".format(end - start, num_image))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=IMAGE_DIR)
    args = parser.parse_args()
    handler(args.path)
