import streamlit as st
import requests
import json
import folium
from streamlit_folium import st_folium
import os
from streamlit_option_menu import option_menu
from datetime import datetime
import geocoder  
import cv2 
import numpy as np
from ultralytics import YOLO
import tempfile
from io import BytesIO

model = YOLO("trained_model.pt")
APIKEY = "YOUR-API-KEY"
ROUTE_DATA_FILE = 'route_data.json'
USER_DATA_FILE = 'user_data.json'

def save_route_data(source, destination, coordinates, traffic_delay, eta):
    route_data = {
        'source': source,
        'destination': destination,
        'coordinates': coordinates,
        'traffic_delay': traffic_delay,
        'eta': eta
    }
    with open(ROUTE_DATA_FILE, 'w') as f:
        json.dump(route_data, f)

def load_route_data():
    if os.path.exists(ROUTE_DATA_FILE):
        with open(ROUTE_DATA_FILE, 'r') as f:
            return json.load(f)
    return None

def save_user_data(points):
    user_data = {'points': points}
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(user_data, f)

def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'r') as f:
            return json.load(f)
    return {'points': 0}

def geocode_location(api_key, address):
    url = f"https://api.tomtom.com/search/2/geocode/{address}.json?key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            coordinates = data['results'][0]['position']
            return coordinates['lat'], coordinates['lon']
    else:
        print("Error in geocoding:", response.status_code)
    return None

def get_static_map(api_key, lat, lon):
    url = f"https://api.tomtom.com/map/1/staticimage?key={api_key}&center={lon},{lat}&zoom=15&width=800&height=600&format=jpg"
    response = requests.get(url)
    if response.status_code == 200:
        with open("traffic_map.jpg", "wb") as f:
            f.write(response.content)
        print("Map image saved as traffic_map.jpg")
    else:
        print("Error fetching map:", response.status_code)

def detect_vehicles(image_path):
    net = cv2.dnn.readNet("Traffic Control System/yolov4.weights", "Traffic Control System/yolov4.cfg")
    with open(r"C:\Users\AARON\Desktop\CyberCup 4.0\Traffic Control System\coco.names", "r") as f:
        classes = f.read().strip().split("\n")

    image = cv2.imread(image_path)
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames()
    detections = net.forward(output_layers)

    vehicle_count = {
        "north": 0,
        "south": 0,
        "east": 0,
        "west": 0
    }

    # Define lane boundaries (these values depend on your specific image)
    lane_boundaries = {
        "north": (width // 4, width // 2),
        "south": (width // 2, (3 * width) // 4),
        "east": ((3 * width) // 4, width),
        "west": (0, width // 4)
    }

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 2:  # Class ID for cars
                center_x = int(obj[0] * width)

                # Determine which direction the vehicle is in based on its position
                if lane_boundaries["north"][0] < center_x < lane_boundaries["north"][1]:
                    vehicle_count["north"] += 1
                elif lane_boundaries["south"][0] < center_x < lane_boundaries["south"][1]:
                    vehicle_count["south"] += 1
                elif lane_boundaries["east"][0] < center_x < lane_boundaries["east"][1]:
                    vehicle_count["east"] += 1
                elif lane_boundaries["west"][0] < center_x < lane_boundaries["west"][1]:
                    vehicle_count["west"] += 1

    return vehicle_count

def adjust_traffic_signals(vehicle_count):
    base_green_time = 30  # Base green time in seconds
    vehicle_multiplier = 2  # Additional seconds per vehicle
    
    signals = {}
    
    for direction in vehicle_count.keys():
        green_time = base_green_time + (vehicle_count[direction] * vehicle_multiplier)
        signals[direction] = green_time

    # Print signal timings and set the appropriate signals
    for direction, green_time in signals.items():
        print(f"{direction.capitalize()}: Set signal to GREEN for {green_time:.2f} seconds")

def save_video_for_download(video_path):
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    return video_bytes

with st.sidebar:
    selected = option_menu("Traffic Saarthi",
                           ["Home", "Route Optimization", "Yolo Real Time Congestion"],
                           default_index=0)

if selected == "Home":
    st.title("Welcome to Traffic Saarthi")
    st.write("This webapp allows you to optimize routes and get real-time congestion predictions using YOLO V8.")
    st.write("Use the sidebar to navigate to the Route Optimization page.")
    st.image("image0_0.jpg", width=600)

elif selected == "Route Optimization":
    st.title("Route Optimization")
    saved_route = load_route_data()
    user_data = load_user_data()

    use_current_location = st.checkbox("Use current location as source")
    source = ""

    if use_current_location:
        g = geocoder.ip('me')  # Attempt to get the user's current location using their IP
        if g.ok and g.latlng:
            source_lat, source_lon = g.latlng
            # Reverse geocode to get a readable address
            reverse_geocode_url = f"https://api.tomtom.com/search/2/reverseGeocode/{source_lat},{source_lon}.json?key={APIKEY}"
            reverse_response = requests.get(reverse_geocode_url)
            reverse_data = reverse_response.json()

            if 'addresses' in reverse_data and len(reverse_data['addresses']) > 0:
                source = reverse_data['addresses'][0]['address']['freeformAddress']
            else:
                st.error("Could not find a human-readable address for the current location.")
                source = "Unknown Location"
        else:
            st.error("Could not determine current location. Please check your internet connection or try again later.")
            source_lat, source_lon = None, None
    else:
        source = st.text_input("Enter source location:", value=saved_route['source'] if saved_route else "")

    # Input for destination
    destination = st.text_input("Enter destination location:", value=saved_route['destination'] if saved_route else "")

    if st.button("Find Route"):
        if not destination:
            st.error("Please enter a destination.")
        else:
            def geocode_address(address):
                geocode_url = f"https://api.tomtom.com/search/2/geocode/{address}.json?key={APIKEY}"
                response = requests.get(geocode_url)
                data = response.json()
                if 'results' in data and len(data['results']) > 0:
                    position = data['results'][0]['position']
                    return position['lat'], position['lon']
                else:
                    st.error(f"Location not found: {address}")
                    return None, None

            # Check if using current location or user-entered source
            if use_current_location and source_lat is not None and source_lon is not None:
                pass  # If using current location, the lat and lon are already obtained
            else:
                source_lat, source_lon = geocode_address(source)

            destination_lat, destination_lon = geocode_address(destination)

            # Ensure both lat/lon pairs are valid before requesting a route
            if source_lat is not None and destination_lat is not None:
                route_url = f"https://api.tomtom.com/routing/1/calculateRoute/{source_lat},{source_lon}:{destination_lat},{destination_lon}/json?key={APIKEY}"
                route_response = requests.get(route_url)
                route_data = route_response.json()

                if 'routes' in route_data and len(route_data['routes']) > 0:
                    route = route_data['routes'][0]['legs'][0]
                    traffic_delay = route['summary']['trafficDelayInSeconds']
                    arrival_time = route['summary']['arrivalTime']

                    try:
                        eta = datetime.fromisoformat(arrival_time[:-1]).strftime("%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        st.error("Error parsing arrival time.")
                        eta = "Unavailable"

                    coordinates = [[point['latitude'], point['longitude']] for point in route['points']]
                    m = folium.Map(location=[source_lat, source_lon], zoom_start=6)
                    folium.Marker(location=[source_lat, source_lon], popup='Source').add_to(m)
                    folium.Marker(location=[destination_lat, destination_lon], popup='Destination').add_to(m)
                    folium.PolyLine(coordinates, color='blue').add_to(m)

                    st_folium(m, width=700, height=500)
                    st.write(f"Traffic Delay: {traffic_delay} seconds")
                    st.write(f"Estimated Time of Arrival (ETA): {eta}")

                    user_data['points'] += 10
                    save_user_data(user_data['points'])
                    save_route_data(source, destination, coordinates, traffic_delay, eta)
                else:
                    st.error("No route found between the specified locations.")

    # Display previously saved route if it exists
    if saved_route:
        st.subheader("Previously Saved Route")
        st.write(f"Source: {saved_route['source']}")
        st.write(f"Destination: {saved_route['destination']}")
        st.write(f"Traffic Delay: {saved_route['traffic_delay']} seconds")

        m_saved = folium.Map(location=saved_route['coordinates'][0], zoom_start=6)
        folium.Marker(location=saved_route['coordinates'][0], popup='Source').add_to(m_saved)
        folium.Marker(location=saved_route['coordinates'][-1], popup='Destination').add_to(m_saved)
        folium.PolyLine(saved_route['coordinates'], color='blue').add_to(m_saved)

        st_folium(m_saved, width=700, height=500)

elif selected == "Yolo Real Time Congestion":
    st.title("YOLO Real-Time Congestion Prediction")

    uploaded_file = st.file_uploader("Upload a video or image for congestion prediction", type=["jpg", "jpeg", "png", "mp4"], key="congestion_uploader")
    
    if uploaded_file is not None:
        if uploaded_file.type == "video/mp4":
            # Save uploaded video to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name

            st.video(video_path)  # Display original video
            st.write("Processing the video for congestion prediction...")

            if st.button("Predict Congestion", key="predict_video"):
                # Open the video file
                cap = cv2.VideoCapture(video_path)
                
                # Check if video opened successfully
                if not cap.isOpened():
                    st.error("Error opening video file.")

                # Get video properties
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))

                # Create temporary output file
                output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

                # Define video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Run YOLOv8 model on the frame
                    results = model.predict(frame)

                    # Visualize results on the frame
                    annotated_frame = results[0].plot()

                    # Write the annotated frame to the output video
                    out.write(annotated_frame)

                cap.release()
                out.release()

                st.success("Video processing complete.")

                # Provide a download link for the processed video
                video_bytes = save_video_for_download(output_path)
                st.download_button(
                    label="Download Processed Video",
                    data=video_bytes,
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )
