from exif import Image
from datetime import datetime
import cv2
import math
import time
from orbit import ISS  # Simulating ISS height retrieval

# Constants
GSD = 12648  # Ground Sample Distance in cm/pixel, needs to be customizable
EARTH_RADIUS = 6371  # in kilometers, for curvature correction
TIME_TO_RUN = 10 * 60  # 10 minutes in seconds

# Switch between feature detection algorithms (SIFT, SURF, AKAZE)
def detect_features(image, algorithm='AKAZE'):
    if algorithm == 'SIFT':
        sift = cv2.SIFT_create()
        return sift.detectAndCompute(image, None)
    elif algorithm == 'SURF':
        surf = cv2.xfeatures2d.SURF_create()
        return surf.detectAndCompute(image, None)
    elif algorithm == 'AKAZE':
        akaze = cv2.AKAZE_create()
        return akaze.detectAndCompute(image, None)

# FLANN-based matcher (more efficient than brute-force)
def calculate_matches(descriptors_1, descriptors_2):
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    return good_matches

# Extract datetime from EXIF and simulate writing time to disk
def get_time(image_path):
    start_time = time.time()
    with open(image_path, 'rb') as image_file:
        img = Image(image_file)
        time_str = img.get("datetime_original")
        capture_time = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
    write_time = time.time() - start_time
    return capture_time, write_time

# Calculate time difference in seconds between consecutive images
def get_time_difference(image_1_path, image_2_path):
    time_1, write_time_1 = get_time(image_1_path)
    time_2, write_time_2 = get_time(image_2_path)
    return (time_2 - time_1).seconds + write_time_1 + write_time_2

# Calculate mean feature distance and adjust for curvature and height of identified features
def calculate_mean_distance(coordinates_1, coordinates_2, altitude=0, curvature_correction=False):
    total_distance = 0
    for (x1, y1), (x2, y2) in zip(coordinates_1, coordinates_2):
        x_diff = x1 - x2
        y_diff = y1 - y2
        distance = math.hypot(x_diff, y_diff)

        # Curvature of Earth correction (if enabled)
        if curvature_correction:
            distance = correct_for_curvature(distance, altitude)

        total_distance += distance
    return total_distance / len(coordinates_1)

# Correct for Earth's curvature using simple geometric correction
def correct_for_curvature(distance, altitude):
    curvature_correction = 2 * math.pi * EARTH_RADIUS * (distance / 360)
    return distance + curvature_correction

# Calculate the speed in km/s
def calculate_speed_in_kmps(feature_distance, time_difference, GSD):
    distance_km = (feature_distance * GSD) / 100000  # Convert to km
    return distance_km / time_difference

# Get the height of ISS from the orbit module
def get_iss_height():
    location = ISS.coordinates()
    return location.altitude  # Height in km

# Main program
if __name__ == "__main__":
    images = ['photo_07464.jpg', 'photo_07465.jpg', 'photo_07466.jpg']  # Multiple photos
    algorithm = 'SIFT'  # Example: use SIFT for feature detection
    use_colored_images = True  # Option to use colored images
    curvature_correction = True  # Apply curvature correction
    ignore_clouds = True  # Placeholder for cloud ignoring logic

    start_program_time = time.time()

    for i in range(len(images) - 1):
        image_1 = images[i]
        image_2 = images[i + 1]
        
        # Get time difference including writing time
        time_difference = get_time_difference(image_1, image_2)
        
        # Load images in color or grayscale
        image_1_cv = cv2.imread(image_1) if use_colored_images else cv2.imread(image_1, cv2.IMREAD_GRAYSCALE)
        image_2_cv = cv2.imread(image_2) if use_colored_images else cv2.imread(image_2, cv2.IMREAD_GRAYSCALE)

        # Detect keypoints and descriptors
        keypoints_1, descriptors_1 = detect_features(image_1_cv, algorithm)
        keypoints_2, descriptors_2 = detect_features(image_2_cv, algorithm)

        # Match descriptors
        matches = calculate_matches(descriptors_1, descriptors_2)

        # Get matching coordinates
        coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_1, keypoints_2, matches)

        # Calculate mean feature distance
        altitude = get_iss_height()
        average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2, altitude, curvature_correction)

        # Calculate speed in km/s
        speed = calculate_speed_in_kmps(average_feature_distance, time_difference, GSD)

        # Write speed to a file
        with open('results.txt', 'a') as result_file:
            result_file.write(f"Speed (km/s): {round(speed, 5)}\n")
        
        # Stop the program after 10 minutes
        if time.time() - start_program_time >= TIME_TO_RUN:
            break
