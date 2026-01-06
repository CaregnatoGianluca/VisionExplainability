import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_gaze_overlay(image_path, gaze_path, alpha=0.5, output_dir=None):
    """
    Overlays a gaze heatmap on an image and visualizes the result.

    Args:
        image_path (str): Path to the original image.
        gaze_path (str): Path to the gaze heatmap image (grayscale).
        alpha (float): Transparency factor for the heatmap overlay (0.0 to 1.0).
        output_dir (str, optional): Directory to save the result. If None, only displays.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    if not os.path.exists(gaze_path):
        print(f"Error: Gaze heatmap not found at {gaze_path}")
        return

    # Helper to load image with unicode path support
    def imread_unicode(path, flags=cv2.IMREAD_COLOR):
        return cv2.imdecode(np.fromfile(path, np.uint8), flags)

    # Load image
    img = imread_unicode(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Load gaze map (grayscale)
    gaze = imread_unicode(gaze_path, cv2.IMREAD_GRAYSCALE)
    if gaze is None:
        print(f"Error: Could not load gaze map from {gaze_path}")
        return

    # Resize gaze map to match image dimensions if necessary
    if gaze.shape != img.shape[:2]:
        gaze = cv2.resize(gaze, (img.shape[1], img.shape[0]))

    # Normalize gaze map to 0-255
    gaze_norm = cv2.normalize(gaze, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Apply colormap to gaze map
    heatmap = cv2.applyColorMap(gaze_norm, cv2.COLORMAP_JET)

    # Create overlay
    overlay = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)

    # Convert BGR to RGB for matplotlib display
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Visualization
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(gaze, cmap='gray')
    plt.title("Gaze Heatmap")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(overlay_rgb)
    plt.title("Overlay")
    plt.axis('off')
    
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        img_name = os.path.basename(image_path)
        name, ext = os.path.splitext(img_name)
        save_path = os.path.join(output_dir, f"overlay_{name}.png")
        # Save the overlay image (BGR format as expected by OpenCV)
        cv2.imwrite(save_path, overlay)
        print(f"Result saved to {save_path}")

    plt.show()

def test_gaze_overlay():

    #image_path = "CUB\\DATASET\\CUB_200_2011\\images\\001.Black_footed_Albatross\\Black_Footed_Albatross_0001_796111.jpg"
    #gaze_path = "CUB\\GAZE_DATASET\\CUB_GHA\\1.jpg"
    name = "a2277f33-7f93debb-91b440be-331ec88d-c603e031"
    #prima immagine 04f60e26-26d3412f-0061558d-41981e82-6aa918e3
    ##seconda immagine 6ff76a0c-49246bc5-a7c149d1-59dc38d9-c11b7ec7
    image_path = "CXR/test/Normal/" + name + ".jpg"
    gaze_path = "CXR/gaze/fixation_heatmaps/" + name + "/heatmap.png"

    print(f"Testing with image: {image_path}")
    print(f"Testing with gaze: {gaze_path}")
    
    visualize_gaze_overlay(image_path, gaze_path, output_dir="per_paper")

if __name__ == "__main__":
    test_gaze_overlay()
