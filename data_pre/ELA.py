import os
from PIL import Image, ImageChops, ImageEnhance
from tqdm import tqdm
import matplotlib.pyplot as plt

def perform_ela_show(image_path, ela_image_path, quality=75):
    """
    Perform Error Level Analysis on the provided image.

    :param image_path: Path to the input image.
    :param ela_image_path: Path to save the ELA image.
    :param quality: Quality level used for JPEG compression (lower value amplifies artifacts).
    """
    # Open the original image
    original = Image.open(image_path)

    # If the image is in RGBA or P mode (palette), convert it to RGB
    if original.mode in ['RGBA', 'P']:
        original = original.convert('RGB')

    # Save the image at a lower JPEG quality to amplify potential changes
    original.save("temp.jpg", 'JPEG', quality=quality)

    # Open the recompressed image
    recompressed = Image.open("temp.jpg")

    # Perform ELA by calculating the difference between the original and recompressed image
    diff = ImageChops.difference(original, recompressed)

    # Enhance the difference image to make the changes more visible
    extrema = diff.getextrema()

    # Handle extrema: it could be a tuple (min, max) or a simple integer for grayscale images
    if isinstance(extrema[0], tuple):  # If each channel returns a tuple
        max_diff = max([ex[1] for ex in extrema])
    else:  # For single channel or grayscale images
        max_diff = extrema[1] if isinstance(extrema, tuple) else extrema

    # Scale the difference image for better visibility
    scale = 255.0 / max_diff if max_diff != 0 else 1
    diff = ImageEnhance.Brightness(diff).enhance(scale)

    # Save the ELA image
    diff.save(ela_image_path)

    # Display the original and ELA image
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(original)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(diff)
    ax[1].set_title('ELA Image')
    ax[1].axis('off')

    plt.show()


def perform_ela(image_path, ela_image_path, quality=90):
    """
    Perform Error Level Analysis on the provided image and save the result.

    :param image_path: Path to the input image.
    :param ela_image_path: Path to save the ELA image.
    :param quality: Quality level used for JPEG compression (lower value amplifies artifacts).
    """
    # Open the original image
    original = Image.open(image_path)

    # If the image is in RGBA or P mode (palette), convert it to RGB
    if original.mode in ['RGBA', 'P']:
        original = original.convert('RGB')

    # Save the image at a lower JPEG quality to amplify potential changes
    original.save("temp.jpg", 'JPEG', quality=quality)

    # Open the recompressed image
    recompressed = Image.open("temp.jpg")

    # Perform ELA by calculating the difference between the original and recompressed image
    diff = ImageChops.difference(original, recompressed)

    # Enhance the difference image to make the changes more visible
    extrema = diff.getextrema()

    # Handle extrema: it could be a tuple (min, max) or a simple integer for grayscale images
    if isinstance(extrema[0], tuple):  # If each channel returns a tuple
        max_diff = max([ex[1] for ex in extrema])
    else:  # For single channel or grayscale images
        max_diff = extrema[1] if isinstance(extrema, tuple) else extrema

    # Scale the difference image for better visibility
    scale = 255.0 / max_diff if max_diff != 0 else 1
    diff = ImageEnhance.Brightness(diff).enhance(scale)

    # Save the ELA image
    diff.save(ela_image_path)


def process_directory(input_dir, output_dir, quality=90):
    """
    Process all images in the input directory, performing ELA on each image and saving the results in the output directory.

    :param input_dir: Path to the directory containing input images.
    :param output_dir: Path to the directory where ELA images will be saved.
    :param quality: Quality level used for JPEG compression (lower value amplifies artifacts).
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get list of image files
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Initialize progress bar
    with tqdm(total=len(image_files), desc="Processing images", unit="image") as pbar:
        # Iterate through all image files
        for filename in image_files:
            input_image_path = os.path.join(input_dir, filename)
            output_image_path = os.path.join(output_dir, f"ela_{filename}")

            # Perform ELA on the image
            perform_ela(input_image_path, output_image_path, quality=quality)

            # Update the progress bar
            pbar.update(1)


if __name__ == '__main__':
    # perform_ela_show("test.jpg", "ela_image.jpg", quality=95)

    # input_directory  = '../../data/Weibo/train/img'  # Replace with the path to your image
    # output_directory  = '../../data/Weibo/ELA/train/img'
    # process_directory(input_directory, output_directory, quality=95)

    perform_ela("2.png","ela_image.jpg")
