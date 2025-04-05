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

# Defining a function to remove outliers using the IQR method
def remove_outliers(df, column_names):
    Q1 = df[column_names].quantile(0.25)
    Q3 = df[column_names].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_filtered = df[((df[column_names] >= lower_bound) & (df[column_names] <= upper_bound)).all(axis=1)]
    return df_filtered

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
    csv_file_name = 'Antony-2024-03-16-12-33-17.csv'
    csv_file_name = find_latest_csv(directory_path)
    if csv_file_name is None:
        sys.exit("No CSV file found. Exiting application.")

    full_csv_path = os.path.join(directory_path, csv_file_name)
    df = pd.read_csv(full_csv_path)  # Load the found CSV file

    eeg_columns = [col for col in df.columns if 'EEG' in col]

    # df = remove_outliers(df, eeg_columns)
    #remove rows if Validation Indicator is 0
    df = df[df['Validation Indicator'] == 1]
    # print(df.head())
    segments = identify_segments(df)  # Assuming you have this function defined

    app = dash.Dash(__name__)

    # Dynamically create segment graph components based on the number of classes
    segment_graph_components = [dcc.Graph(id=f'segment-graph-{cls}', figure={}) for cls in segments.keys()]

    app.layout = html.Div([
        html.H3(csv_file_name, style={'textAlign': 'center', 'marginBottom': '20px'}),
        html.Div(style={'display': 'flex', 'flexDirection': 'row'}, children=[
            html.Div([
                *[dcc.Graph(id=f'eeg-{i}', figure={}) for i in range(1, 9)]
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'height': '95vh', 'overflowY': 'scroll'}),
            
            html.Div([
                html.H4(f"Total Rows: {len(df)}"),
                html.H4(f"Session Length (seconds): {len(df) / 250:.2f}"),
                html.H4("Segments per Class:"),
                html.Pre("\n".join([f"{cls}: {len(segs)} segments" for cls, segs in segments.items()])),
                *segment_graph_components
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
        [Output(f'segment-graph-{cls}', 'figure') for cls in segments.keys()],
        [Input('interval-component', 'n_intervals')]
    )
    def update_graph(n):
        # Update EEG graphs to skip the first 100 points
        eeg_figures = [
            go.Figure(data=[go.Scatter(y=df[f'EEG {i}'][100:], mode='lines')]).update_layout(  # Adjusted here
                title=f'EEG {i}',
                margin=dict(l=40, r=40, t=40, b=40),
                plot_bgcolor='rgba(0,0,0,0)',
            ) for i in range(1, 9)
        ]
        
        # Update segment graphs dynamically based on the segments dictionary
        segment_figures = []
        for cls, segments_list in segments.items():
            segment_index = n % len(segments_list)
            segment = segments_list[segment_index]
            fig = go.Figure()
            for i in range(1, 9):
                # Ensure each segment graph reflects independent data
                # and adjust the x-axis if necessary to reflect skipping the first 100 points
                segment_length = len(segment)
                x_values = np.arange(segment_length)[20000:] if segment_length > 20000 else np.arange(segment_length)
                y_values = segment[f'EEG {i}'][20000:] if segment_length > 20000 else segment[f'EEG {i}']
                fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name=f'EEG {i}'))
            fig.update_layout(
                title=f'Class: {cls}, Segment: {segment_index + 1}',
                margin=dict(l=40, r=40, t=40, b=40),
                plot_bgcolor='rgba(0,0,0,0)',
            )
            segment_figures.append(fig)

        return eeg_figures + segment_figures

    app.run_server(debug=True)