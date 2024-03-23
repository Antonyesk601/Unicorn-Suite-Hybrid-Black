import dash
from dash import dcc, html
import plotly.graph_objs as go
import pandas as pd
from dash.dependencies import Input, Output, State
import numpy as np
from time import sleep
import matplotlib.pyplot as plt
import os
import sys

def find_latest_csv(directory):
    try:
        # List all files in the given directory
        files = os.listdir(directory)
        # Filter out files with a .csv extension
        csv_files = [file for file in files if file.endswith('.csv')]
        # Check if there are no CSV files in the directory
        if not csv_files:
            print("No CSV files found in the directory.")
            return

        # Find the latest modified CSV file
        latest_csv = max(csv_files, key=lambda x: os.path.getmtime(os.path.join(directory, x)))

        print(f"The latest CSV file is: {latest_csv}")
        return latest_csv
    except Exception as e:
        print(f"Error: {e}")

# Assuming df is your DataFrame loaded from the CSV

def identify_segments(df, class_column='State'):
    """
    Identify segments for each class.

    :param df: DataFrame with data
    :param class_column: Name of the column with class labels
    :return: Dictionary with class labels as keys and lists of segments (DataFrames) as values
    """
    segments = {}
    current_class = None
    segment_start = 0

    for i, (index, row) in enumerate(df.iterrows()):
        if row[class_column] != current_class:
            if current_class is not None:
                segment_end = i
                if current_class not in segments:
                    segments[current_class] = []
                segments[current_class].append(df.iloc[segment_start:segment_end])
            segment_start = i
            current_class = row[class_column]

    # Add the last segment
    if current_class not in segments:
        segments[current_class] = []
    segments[current_class].append(df.iloc[segment_start:])

    return segments

def plot_segment(segment, title):
    """
    Plots a given segment as a line plot.

    :param segment: DataFrame segment to plot
    :param title: Title for the plot
    """
    plt.figure(figsize=(10, 6))
    for col in [c for c in segment.columns if c.startswith('EEG')]:
        plt.plot(segment.index, segment[col], label=col)
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

def simulate_segment_display_with_plots(segments):
    """
    Simulates displaying each segment as a line plot for 3 seconds.

    :param segments: Dictionary with class labels as keys and lists of segments as values
    """
    for class_label, segment_list in segments.items():
        for segment in segment_list:
            plot_title = f"Class: {class_label}, Segment starting at index {segment.index[0]}"
            plot_segment(segment, plot_title)
            sleep(3)  # Wait for 3 seconds before moving to the next segment

# csv_file_name = 'Antony-2024-03-16-12-33-17.csv'

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <directory_path>")
        sys.exit(1)

    directory_path = sys.argv[1]
    csv_file_name = find_latest_csv(directory_path)
    # Example usage
    df = pd.read_csv(os.path.join(directory_path,csv_file_name))  # Load your CSV file
    segments = identify_segments(df)

    app = dash.Dash(__name__)

    # Calculate stats for the right column header
    total_rows = len(df)
    session_length_seconds = total_rows / 250
    segments_count_text = "\n".join([f"{cls}: {len(segs)} segments" for cls, segs in segments.items()])

    app.layout = html.Div([
        html.H3(csv_file_name, style={'textAlign': 'center', 'marginBottom': '20px'}),
        html.Div(style={'display': 'flex', 'flexDirection': 'row'}, children=[
            html.Div([
                *[dcc.Graph(id=f'eeg-{i}', figure={}) for i in range(1, 9)]
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'height': '95vh', 'overflowY': 'scroll'}),
            
            html.Div([
                html.H4(f"Total Rows: {total_rows}"),
                html.H4(f"Session Length (seconds): {session_length_seconds:.2f}"),
                html.H4("Segments per Class:"),
                html.Pre(segments_count_text),
                *[dcc.Graph(id=f'segment-graph-{i}', figure={}) for i in range(4)]
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'height': '95vh', 'overflowY': 'scroll'}),
        ]),
        
        dcc.Interval(
            id='interval-component',
            interval=3*1000,  # in milliseconds
            n_intervals=0
        )
    ])

    @app.callback(
        [Output(f'eeg-{i}', 'figure') for i in range(1, 9)] +
        [Output(f'segment-graph-{i}', 'figure') for i in range(4)],
        [Input('interval-component', 'n_intervals')]
    )
    def update_graph(n):
        # Update EEG graphs independently
        eeg_figures = [
            go.Figure(data=[go.Scatter(y=df[f'EEG {i}'], mode='lines')]).update_layout(
                title=f'EEG {i}',
                margin=dict(l=40, r=40, t=40, b=40),  # Adjusted margins
                plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
            ) for i in range(1, 9)
        ]
        # for fig in eeg_figures:
        #     fig.update_yaxes(range=[230_000,270_000])
        
        # Update segment graphs independently
        segment_figures = []
        classes = list(segments.keys())[:4]  # Adjust as necessary for the number of classes you wish to display
        for idx, cls in enumerate(classes):
            segments_list = segments[cls]
            segment_index = n % len(segments_list)  # Ensures cycling through each segment independently
            segment = segments_list[segment_index]
            fig = go.Figure()
            for i in range(1, 9):
                # Ensure each segment is plotted independently
                fig.add_trace(go.Scatter(x=np.arange(len(segment)), y=segment[f'EEG {i}'], mode='lines', name=f'EEG {i}'))
            fig.update_layout(
                title=f'Class: {cls}, Segment: {segment_index + 1}',
                margin=dict(l=40, r=40, t=40, b=40),
                plot_bgcolor='rgba(0,0,0,0)',
            )
            segment_figures.append(fig)

        return eeg_figures + segment_figures

    app.run_server(debug=True)