# marcel the coolest
# campus map 2.0
# i was inspired to make this for my final because i was really excited about the map project,
# and knew that it could be done better.
# all data is stored in one clean json file, no need for exporting data from caltopo,
# taking full advantage of gis software for a real mapping program!

# also, I recall you mentioned you considered including Streamlit to be used in your future assignments.abs
# I recomend NiceGUI over streamlit. Very easy to use, with a lot more control. Can even add css and html code if desired, and able to host as a real webapp unlike Streamit.

# "We at Zauberzeug like Streamlit but find it does too much magic when it comes to state handling. In search for an alternative nice library to write simple graphical user interfaces in Python we discovered JustPy."
# "Although we liked the approach, it is too "low-level HTML" for our daily usage. But it inspired us to use Vue and Quasar for the frontend."
# "We have built on top of FastAPI, which itself is based on the ASGI framework Starlette and the ASGI webserver Uvicorn because of their great performance and ease of use."

# p.s: the farthest node is 107, if you'd like to try a really long path as the starting location

import json
from pathlib import Path # Object oriented file system paths
from typing import Dict, List, Tuple, Any, Optional # Import type hints to indicate expected data types
from nicegui import ui # UI interface
import heapq # Used for the priority queue in Dijkstra's algorithm

# --- Configuration ---

# Assuming the geojson files are in the same directory as the script
# Primary GeoJSON file with line features (roads/paths) and building polygons
GEOJSON_FILE_PATH = Path(__file__).parent / 'paths.geojson'
BUILDINGS_FILE_PATH = Path(__file__).parent / 'buildings.geojson'

# Initial map view settings
initial_zoom = 16 # Zoom level

center_coords = (38.03059348134153, -120.38742722828174)

# --- Type Hinting for Graph Structure ---

# Represents the data associated with an edge
EdgeData = Dict[str, Any] # Could include 'coordinates', 'weight', 'type', etc.
# Represents the outgoing edges from a node: {end_node: edge_data}
OutgoingEdges = Dict[int, EdgeData]

# --- Name to Node ID Mapping ---
# This is so we can search for building names
# The ID assigned is the nearest intersection node to that building.
# Preferably, they would have their own custom nodes, but I could not resolve errors in my mapping software when trying to implement this.
name_to_node_id = {
    "observatory": 4,
    "willow": 49,
    "toyon": 10,
    "tamarack": 31,
    "sugar pine": 7,
    "sequoia": 23,
    "redbud": 25,
    "ponderosa": 19,
    "poison oak": 102,
    "pinyon": 8,
    "oak pavilion": 104,
    "maple": 19,
    "manzanita": 70,
    "mahogany": 6,
    "madrone": 8,
    "child care": 8,
    "juniper": 38,
    "fir": 41,
    "dogwood": 48,
    "cedar": 66,
    "buckeye": 77,
    "aspen": 57,
    "alder": 65
}

# Represents the entire graph: {start_node: outgoing_edges}
Graph = Dict[int, OutgoingEdges]

# --- Load GeoJSON Data and Build Initial Graph ---

def load_geojson_data_and_build_graph(filepath: Path) -> Tuple[Graph, List[Dict]]:
    """
    Loads GeoJSON, builds a directed graph with bidirectional edges
    weighted by 'time' and 'time_backward', and stores line features.

    Graph structure: {start_node: {end_node: {'coordinates': [(lat, lon), ...], 'weight': float, 'type': str}, ...}, ...}
    Returns:
        A tuple containing:
        - The graph.
        - A list of line features with their properties and coordinates.
    """
    graph: Graph = {}
    line_features: List[Dict] = []
    print(f"Attempting to load GeoJSON from: {filepath}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if data.get("type") != "FeatureCollection" or "features" not in data:
            print("Error: Invalid GeoJSON structure (expected FeatureCollection with features)")
            return {}, []

        print(f"Loaded {len(data.get('features', []))} features.")

        for feature in data.get("features", []):
            geometry = feature.get("geometry")
            properties = feature.get("properties", {})

            if geometry and geometry.get("type") == "LineString":
                coordinates_lon_lat = geometry.get("coordinates", [])

                try:
                    start_node_str = properties.get("start_NorthingOrder")
                    end_node_str = properties.get("end_NorthingOrder")
                    line_id = properties.get("fid") # Get the line ID

                    if start_node_str is None or end_node_str is None or line_id is None:
                         print(f"Warning: Skipping feature with missing start/end node or fid properties: {properties}")
                         continue

                    start_node = int(start_node_str)
                    end_node = int(end_node_str)

                    coordinates_lat_lon = [(coord[1], coord[0]) for coord in coordinates_lon_lat if len(coord) == 2]

                    if not coordinates_lat_lon:
                        print(f"Warning: Skipping feature with no valid coordinates: {properties}")
                        continue

                    # Store the line feature data
                    line_features.append({
                        'fid': line_id,
                        'properties': properties,
                        'coordinates': coordinates_lat_lon,
                        'start_node': start_node,
                        'end_node': end_node
                    })

                    # Add forward edge (start -> end) with 'time' weight
                    time_prop = properties.get("time")
                    if time_prop is not None:
                        try:
                            time_weight = float(time_prop)
                            if start_node not in graph:
                                graph[start_node] = {}
                            graph[start_node][end_node] = {
                                'coordinates': coordinates_lat_lon,
                                'weight': time_weight,
                                'type': 'road_segment'
                            }

                        except (ValueError, TypeError):
                            print(f"Warning: Skipping forward edge from {start_node} to {end_node} due to invalid 'time' property: {time_prop}")

                    # Add backward edge (end -> start) with 'time_backward' weight
                    time_backward_prop = properties.get("time_backward")
                    if time_backward_prop is not None:
                         try:
                            time_backward_weight = float(time_backward_prop)
                            if end_node not in graph:
                                graph[end_node] = {}
                            graph[end_node][start_node] = {
                                # Reverse coords for backward
                                'coordinates': list(reversed(coordinates_lat_lon)),
                                'weight': time_backward_weight,
                                'type': 'road_segment'
                            }
                         except (ValueError, TypeError):
                            print(f"Warning: Skipping backward edge from {end_node} to {start_node} due to invalid 'time_backward' property: {time_backward_prop}")

                except (ValueError, TypeError, KeyError) as e:
                    print(f"Warning: Skipping feature due to invalid/missing property or conversion error ({e}): {properties}")
                except IndexError:
                    print(f"Warning: Skipping feature due to invalid coordinate format: {properties}")

    except FileNotFoundError:
        print(f"Error: GeoJSON file not found at {filepath}")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
    except Exception as e:
        print(f"An unexpected error occurred loading GeoJSON: {e}")

    print(f"Graph loading complete. Loaded {len(graph)} nodes with outgoing edges.")
    print(f"Loaded {len(line_features)} line features.")
    return graph, line_features

# --- Load Building Polygon Data ---

def load_building_data(filepath: Path) -> Dict[str, List[Tuple[float, float]]]:
    """
    Loads building polygon data from a GeoJSON file.

    Args:
        filepath: The path to the GeoJSON file containing building polygons.

    Returns:
        A dictionary mapping building names to their polygon coordinates.
    """
    buildings: Dict[str, List[Tuple[float, float]]] = {}
    print(f"Attempting to load building data from: {filepath}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if data.get("type") != "FeatureCollection" or "features" not in data:
            print("Error: Invalid GeoJSON structure for buildings (expected FeatureCollection with features)")
            return {}

        print(f"Loaded {len(data.get('features', []))} building features.")

        for feature in data.get("features", []):
            geometry = feature.get("geometry")
            properties = feature.get("properties", {})

            if geometry and geometry.get("type") == "Polygon":
                building_name = properties.get("name")
                coordinates_lon_lat = geometry.get("coordinates", [])

                if building_name and coordinates_lon_lat and len(coordinates_lon_lat) > 0:
                    # Because GeoJSON Polygon coordinates are list of rings,
                    # Each ring being a list of [lon, lat] points forming a closed loop (ring or hole).
                    # I assume the first ring is the exterior boundary and only process that.
                    exterior_ring = coordinates_lon_lat[0]
                    coordinates_lat_lon = [(coord[1], coord[0]) for coord in exterior_ring if len(coord) == 2]

                    if coordinates_lat_lon:
                        buildings[building_name.lower()] = coordinates_lat_lon
                        # print(f"Loaded building: {building_name}")

        print(f"Building data loading complete. Loaded {len(buildings)} buildings.")
        return buildings

    except FileNotFoundError:
        print(f"Error: Buildings GeoJSON file not found at {filepath}")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath} for buildings")
    except Exception as e:
        print(f"An unexpected error occurred loading building data: {e}")

    return buildings


# Load the data into the graph and get line features
edge_graph, all_line_features = load_geojson_data_and_build_graph(GEOJSON_FILE_PATH)
building_polygons = load_building_data(BUILDINGS_FILE_PATH)


# --- Dijkstra's Algorithm Implementation ---

def dijkstra(graph: Graph, start_node: int, end_node: int) -> Tuple[Optional[List[int]], Optional[float]]:
    """
    Finds the shortest path between a start and end node in a graph
    using Dijkstra's algorithm.

    Args:
        graph: The graph represented as {start_node: {end_node: {'weight': float, ...}, ...}, ...}
        start_node: The starting node ID (can be intersection or building node).
        end_node: The destination node ID (can be intersection or building node).

    Returns:
        A tuple containing:
        - The shortest path as a list of node IDs (e.g., [1, 5, 10]), or None if no path exists.
        - The total weight (time) of the shortest path, or None if no path exists.
    """
    if start_node not in graph or end_node not in graph:
        print(f"Error: Start node {start_node} or end node {end_node} not found in the graph.")
        return None, None

    priority_queue = [(0, start_node)]
    distances: Dict[int, float] = {node: float('inf') for node in graph}
    distances[start_node] = 0
    predecessors: Dict[int, Optional[int]] = {node: None for node in graph}

    print(f"Starting Dijkstra from {start_node} to {end_node}")

# Process nodes from the priority queue
    while priority_queue:
        # Get the node with the smallest distance
        current_distance, current_node = heapq.heappop(priority_queue)

        # If the destination node is reached, reconstruct and return the path
        if current_node == end_node:
            path = []
            # Backtrack from the end node using predecessors
            while current_node is not None:
                path.append(current_node)
                current_node = predecessors[current_node]
            path.reverse() # Path was built backward, so reverse it
            print(f"Path found: {path} with total distance {current_distance}")
            return path, current_distance # Return the final path and its total time

        if current_distance > distances[current_node]:
            continue

        # Explore neighbors
        if current_node in graph:
            for neighbor, edge_data in graph[current_node].items():
                weight = edge_data.get('weight')
                if weight is not None:
                    distance = current_distance + weight

                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        predecessors[neighbor] = current_node
                        heapq.heappush(priority_queue, (distance, neighbor))

    print(f"No path found from {start_node} to {end_node}")
    return None, None

# --- Create the Leaflet Map and NiceGUI Elements ---

with ui.column().classes('w-full items-center'):
    ui.label('Shortest Path Finder').classes('text-xl font-bold') # Updated title

    map_instance = ui.leaflet(center=center_coords, zoom=initial_zoom)
    map_instance.style('width: 600px; height: 600px;')
    map_instance.classes('mx-auto rounded-lg shadow-lg')

    # Input fields for start and destination nodes/buildings
    with ui.row().classes('w-full justify-center gap-4'):
        start_node_input = ui.input('Start', placeholder='Enter Name or ID', autocomplete=list(name_to_node_id.keys())).classes('w-1/8')
        end_node_input = ui.input('Destination', placeholder='Enter Name or ID', autocomplete=list(name_to_node_id.keys())).classes('w-1/8')

    # Label to display the total time
    total_time_label = ui.label('Shortest Path Time: --').classes('text-lg mt-4')

# --- Define Line Styling ---

line_options = {
    'color': 'purple', # Color of the drawn path
    'weight': 10,       # Thickness of the line
    'opacity': 0.9     # Transparency
}

# --- Function to Draw Selected Lines/Segments from Graph ---

def draw_path_from_sequence(sequence: List[int]):
    """
    Draws lines/segments on the map corresponding to the node sequence using the loaded graph data.
    Handles both road segments and access edges to/from buildings.
    """
    # Path line drawing is handled in the main function after clearing layers.
    # This function specifically draws the path polyline.

    if not sequence or len(sequence) < 2:
        return

    print(f"Attempting to draw path sequence: {sequence}")

    # Dictionary to store drawn edges to avoid drawing duplicates in bidirectional graphs
    drawn_edges = set()

    path_coordinates = []
    for i in range(len(sequence) - 1):
        start_node = sequence[i]
        end_node = sequence[i+1]
        # print(f"Checking segment for drawing: {start_node} -> {end_node}")

        # Check if the edge exists in the graph
        coordinates = None
        edge_type = None

        if start_node in edge_graph and end_node in edge_graph[start_node]:
             edge_data = edge_graph[start_node][end_node]
             coordinates = edge_data.get('coordinates')
             edge_type = edge_data.get('type')


        if coordinates:
            # Add the coordinates for this segment to the overall path coordinates
            path_coordinates.extend(coordinates)
            # print(f"Added coordinates for segment {start_node} -> {end_node}")

    if path_coordinates:
         # Draw the complete path as a single polyline
         map_instance.generic_layer(name='polyline', args=[path_coordinates, line_options])
         print("Successfully drew complete path polyline.")

# --- Function to handle button click and trigger pathfinding/drawing ---

def find_and_draw_shortest_path():
    """
    Finds the shortest path between a user-specified start and destination and visualizes it on the map.

    Retrieves the start and destination points from the 'start_node_input' and 'end_node_input'
    NiceGUI elements. These inputs can be either a building name or an integer graph node ID.
    The function validates the inputs, maps building names to node IDs, runs Dijkstra's algorithm
    on the 'edge_graph', draws the resulting path, and displays the estimated time.

    Input Sources:
        start_node_input (NiceGUI Input): The UI element providing the start point value.
        end_node_input (NiceGUI Input): The UI element providing the destination value.

    Actions:
        - Clears existing path drawings and building highlights from the map.
        - Highlights the start and destination buildings on the map if entered by name.
        - Draws a polyline representing the shortest path on the map instance.
        - Updates the 'total_time_label' text with the calculated path duration.
        - Sends notifications to the user via 'ui.notify' regarding the process outcome
          (success, warning, error).

    Assumes:
        - Global or accessible variables: map_instance, building_polygons, name_to_node_id,
          edge_graph, total_time_label, ui.
        - Availability of dijkstra(graph, start, end) and draw_path_from_sequence(sequence) functions.
    """
    try:
        start_input_value = start_node_input.value.strip()
        end_input_value = end_node_input.value.strip()

        if not start_input_value or not end_input_value:
            ui.notify("Please enter both a start and a destination.", type='warning')
            return

        # Clear previous drawings (path and buildings)
        map_instance.clear_layers()
        # Re-add base map tiles after clearing other layers
        map_instance.tile_layer(url_template='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png')

        # Determine the start node/building and highlight
        start_node_id = None
        start_building_name = None
        start_coords = None

        # Try to find the start as a building name first
        if start_input_value.lower() in building_polygons:
            start_building_name = start_input_value.lower()
            start_coords = building_polygons[start_building_name]
            # Find the nearest graph node to this building for pathfinding
            # For simplicity, we'll use the predefined mapping.
            start_node_id = name_to_node_id.get(start_building_name)
            if start_node_id is None:
                 ui.notify(f"Start building '{start_input_value}' found, but corresponding graph node is not mapped.", type='negative')
                 total_time_label.set_text('Shortest Path Time: --')
                 return
            print(f"Start is building: {start_building_name}, mapped to node ID: {start_node_id}")
            # Highlight start building
            map_instance.generic_layer(name='polygon', args=[start_coords, {'color': 'yellow', 'fillColor': 'yellow', 'fillOpacity': 0.6, 'weight': 2}])
        else:
            # If not a building name, try to interpret as an integer node ID
            try:
                start_node_id = int(start_input_value)
                print(f"Interpreted start as integer node ID: {start_node_id}")
            except ValueError:
                ui.notify("Please enter a valid Building Name or integer Node ID for the Start.", type='negative')
                total_time_label.set_text('Shortest Path Time: --')
                return

        if start_node_id is not None and start_node_id not in edge_graph:
             ui.notify(f"Start node ID {start_node_id} not found in the graph.", type='negative')
             total_time_label.set_text('Shortest Path Time: --')
             return

        # Determine the destination node/building and highlight
        destination_node_id = None
        destination_building_name = None
        destination_coords = None
        destination_identifier = end_input_value # Use original input for notifications

        # Try to find the destination as a building name
        if end_input_value.lower() in building_polygons:
            destination_building_name = end_input_value.lower()
            destination_coords = building_polygons[destination_building_name]
            # Find the nearest graph node to this building for pathfinding
            # For simplicity, we'll use the predefined mapping.
            destination_node_id = name_to_node_id.get(destination_building_name)
            if destination_node_id is None:
                 ui.notify(f"Destination building '{end_input_value}' found, but corresponding graph node is not mapped.", type='negative')
                 total_time_label.set_text('Shortest Path Time: --')
                 return
            print(f"Destination is building: {destination_building_name}, mapped to node ID: {destination_node_id}")
            # Highlight destination building
            map_instance.generic_layer(name='polygon', args=[destination_coords, {'color': 'blue', 'fillColor': 'blue', 'fillOpacity': .6, 'weight': 2}])
        else:
            # If not a building name, try to interpret as an integer node ID
            try:
                destination_node_id = int(end_input_value)
                print(f"Interpreted destination as integer node ID: {destination_node_id}")
            except ValueError:
                 ui.notify("Please enter a valid Building Name or integer Node ID for the Destination.", type='negative')
                 total_time_label.set_text('Shortest Path Time: --')
                 return

        # Final check that a destination was found and exists in graph (if not a building)
        if destination_node_id is None or destination_node_id not in edge_graph:
             if destination_building_name is None: # Avoid double notifying if building not mapped
                ui.notify(f"Destination '{destination_identifier}' not found as a valid Building Name or integer Node ID in the graph.", type='negative')
                total_time_label.set_text('Shortest Path Time: --')
                return
             elif destination_node_id is not None and destination_node_id not in edge_graph:
                  ui.notify(f"Destination building '{destination_building_name}' mapped to node ID {destination_node_id}, which is not found in the graph.", type='negative')
                  total_time_label.set_text('Shortest Path Time: --')
                  return

        if start_node_id is None or destination_node_id is None:
             ui.notify("Could not determine valid start and destination for pathfinding.", type='negative')
             total_time_label.set_text('Shortest Path Time: --')
             return

        if start_node_id == destination_node_id:
            ui.notify("Start and destination cannot be the same.", type='warning')
            total_time_label.set_text('Shortest Path Time: --')
            # Still highlight buildings if they were entered by name
            if start_building_name and start_coords:
                 map_instance.generic_layer(name='polygon', args=[start_coords, {'color': 'grey', 'fillColor': 'grey', 'fillOpacity': 0.6, 'weight': 2}])
            if destination_building_name and destination_coords:
                 map_instance.generic_layer(name='polygon', args=[destination_coords, {'color': 'blue', 'fillColor': 'blue', 'fillOpacity': 0.6, 'weight': 2}])
            return

        # Run Dijkstra's algorithm
        path_sequence, total_weight = dijkstra(edge_graph, start_node_id, destination_node_id)

        if path_sequence:
            # Draw the found path
            draw_path_from_sequence(path_sequence)

            # Time Conversion ( 'Time Flies' :D )
            if total_weight is not None:
                total_seconds = round(total_weight) # Round to nearest whole second
                if total_seconds < 60:
                    time_display = f'{total_seconds} seconds'
                else:
                    minutes = total_seconds // 60
                    seconds = total_seconds % 60
                    time_display = f'{minutes} minutes {seconds} seconds'

                total_time_label.set_text(f'Shortest Path Time: {time_display}')
                ui.notify(f"Shortest path found and drawn. Total time: {time_display}")
            else:
                 total_time_label.set_text('Shortest Path Time: --')
                 ui.notify(f"Shortest path found and drawn. Total time: --")
            # --- End time conversion ---

        # Edge case when no path is found
        else:
            ui.notify(f"No path found from '{start_input_value}' to '{destination_identifier}'.", type='negative')
            total_time_label.set_text('Shortest Path Time: --')
            # Still highlight buildings if they were entered by name
            if start_building_name and start_coords:
                 map_instance.generic_layer(name='polygon', args=[start_coords, {'color': 'grey', 'fillColor': 'grey', 'fillOpacity': 0.6, 'weight': 2}])
            if destination_building_name and destination_coords:
                 map_instance.generic_layer(name='polygon', args=[destination_coords, {'color': 'blue', 'fillColor': 'blue', 'fillOpacity': 0.6, 'weight': 2}])

    # Handle any unexpected errors
    except Exception as e:
        ui.notify(f"An error occurred: {e}", type='negative')
        print(f"An error occurred during pathfinding/drawing: {e}")
        total_time_label.set_text('Shortest Path Time: --')
        map_instance.clear_layers()
        map_instance.tile_layer(url_template='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png')

# --- Create the Button to Find and Draw the Path ---
with ui.row().classes('w-full justify-center mt-4'):
    ui.button(
        'Find and Draw Shortest Path',
        on_click=find_and_draw_shortest_path
    ).classes('px-6 py-3 text-lg rounded-lg shadow-md bg-green-600 text-white hover:bg-green-700')

# --- Run the App ---
# Set the title of the web page
ui.run(title='Columbia Pathfinder 2.0')


# P.S.
# Thank you for class, Joe!
# I had a great time. :)
