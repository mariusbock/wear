from utils import merge_sensor_csv
import plotly.graph_objects as go

input_dir = 'data_creation/recordings/sbj_2_session_1'
sensors = ['1a3e', '2b9f', 'a1cf', 'ba11']

# plot each sensor and look for the sync points using the plots and final cut
for sensor in sensors:
    sensor_data = merge_sensor_csv(input_dir, sensor)
    fig = go.Figure(layout_title_text=sensor)
    fig.add_trace(go.Scatter(y=sensor_data['acc_x'], x=list(range(len(sensor_data))),
                             mode='lines',
                             name='lines'))
    fig.add_trace(go.Scatter(y=sensor_data['acc_y'], x=list(range(len(sensor_data))),
                             mode='lines',
                             name='lines'))
    fig.add_trace(go.Scatter(y=sensor_data['acc_z'], x=list(range(len(sensor_data))),
                             mode='lines',
                             name='lines'))
    fig.show()

# use the plots and final cut in order to:
#   - write down the sync_points and data_points belonging to the synchronisation steps of each sensor
#   - define a final_time which is consistent with all sensors
#   - write your results into the meta file
