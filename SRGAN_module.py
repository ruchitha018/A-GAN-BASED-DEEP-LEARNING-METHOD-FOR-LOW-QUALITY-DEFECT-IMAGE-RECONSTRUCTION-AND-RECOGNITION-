# selecting a Data-Structure to handle images effeciently
import numpy as np

# For handling the Folder/Files Structure
import os

# Reading images and performing Image processing
import cv2
from PIL import Image
import tensorflow as tf

# visualizing the images
from matplotlib import pyplot as plt
plt.switch_backend('Agg')

# GAN model
import tensorflow_hub as hub

# miscellaneous purpose
import math
import time
from skimage.metrics import structural_similarity as ssim

# SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
SAVED_MODEL_PATH = "esrgan-tf2_1"
model = hub.load(SAVED_MODEL_PATH)

def psnr(target, ref):
    """define a function for peak signal-to-noise ratio (PSNR)"""
    # assume RGB image
    target_data = target.astype(float)
    ref_data = ref.astype(float)

    diff = ref_data - target_data
    diff = diff.flatten('C')

    rmse = math.sqrt(np.mean(diff ** 2.))

    return 20 * math.log10(255. / rmse)

def mse(target, ref):
    """define function for mean squared error (MSE)"""
    # the MSE between the two images is the sum of the squared difference between the two images
    err = np.sum((target.astype('float') - ref.astype('float')) ** 2)
    err /= float(target.shape[0] * target.shape[1])
    return err
 
def compare_images(target, ref):
    """define function that combines all three image quality metrics"""
    un_changed = "Shape: "+str(target.shape)+"\n"+"D-Type: "+str(target.dtype)
    # ref = cv2.resize(ref,(target.shape[1],target.shape[0]))
    target = cv2.resize(target,(ref.shape[1],ref.shape[0]))
    return  f'PSNR: {psnr(target,ref)}\nMSE: {mse(target,ref)}\nSSIM: {ssim(target, ref, multichannel =True)}\n{un_changed}'

def image_preparation_for_model(image=None, path=None):
    ''' Function to read image and prepare it for the model
     to predict the super resolution image as an output'''
    if path:
        hr_image = cv2.imread(path)
        hr_image = cv2.cvtColor(hr_image,cv2.COLOR_BGR2RGB)
    else:
        hr_image = image
        
    hr_size = (np.array(hr_image.shape[:-1]) // 4) * 4

    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
    hr_image = tf.cast(hr_image, tf.float32)
    hr_image = tf.expand_dims(hr_image, 0)

    return hr_image

def convert_to_viewable_image(nd_image):
    image_array = tf.squeeze(nd_image)
    image_array = np.clip(image_array, a_min = 0, a_max = 255)
    return image_array.astype("uint8")

def downscale_image(image):
    """
      Scales down images using bicubic downsampling.
      Args:
          image: 3D or 4D tensor of preprocessed image
    """
    image_size = []
    if len(image.shape) == 3:
        image_size = [image.shape[1], image.shape[0]]
    else:
        raise ValueError("Dimension mismatch. Can work only on single image.")

    image = tf.squeeze(tf.cast(tf.clip_by_value(image, 0, 255), tf.uint8))

    lr_image = np.asarray(Image.fromarray(image.numpy()).resize([image_size[0] // 4, image_size[1] // 4],Image.BICUBIC))
    lr_image = tf.expand_dims(lr_image, 0)
    lr_image = tf.cast(lr_image, tf.float32)
    return lr_image

def predict_normal(image_path,file_name):
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 10))

    plt.subplot(1,2,1)
    plt.imshow(original_image)
    plt.title("ORIGINAL IMAGE",size=15,weight="bold")
    text = "Shape: "+str(original_image.shape)+"\n"+"D-Type: "+str(original_image.dtype)
    plt.xlabel(text,size=15,weight="bold")

    plt.subplot(1,2,2)
    processed_image = image_preparation_for_model(path=image_path)
    GAN_output = model(processed_image)
    gan_out = convert_to_viewable_image(GAN_output)
    plt.imshow(gan_out)
    plt.title("SUPER RESOLUTION IMAGE",size=15,weight="bold")
    plt.xlabel(compare_images(gan_out,original_image),size=15,weight="bold")

    plt.tight_layout()
    plt.savefig(file_name, bbox_inches="tight")
    plt.show()


def predict_degrade(image_path,file_name):
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 10))

    plt.subplot(1,3,1)
    plt.imshow(original_image)
    plt.title("ORIGINAL IMAGE",size=15,weight="bold")
    text = "Shape: "+str(original_image.shape)+"\n"+"D-Type: "+str(original_image.dtype)
    plt.xlabel(text,size=15,weight="bold")
    plt.ioff()

    plt.subplot(1,3,2)
    degraded_image = downscale_image(original_image)
    d_img = convert_to_viewable_image(degraded_image)
    plt.imshow(d_img)
    plt.title("DRGRADED IMAGE",size=15,weight="bold")
    plt.xlabel(compare_images(d_img,original_image),size=15,weight="bold")
    plt.ioff()

    plt.subplot(1,3,3)
    GAN_output = model(degraded_image)
    gan_out = convert_to_viewable_image(GAN_output)
    plt.imshow(gan_out)
    plt.title("SUPER RESOLUTION IMAGE",size=15,weight="bold")
    plt.xlabel(compare_images(gan_out,original_image),size=15,weight="bold")
    plt.ioff()

    plt.tight_layout()
    plt.savefig(file_name, bbox_inches="tight")