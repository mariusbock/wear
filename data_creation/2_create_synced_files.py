from utils import create_synced_files

input_dir = 'data_creation/recordings/sbj_12_session_2'
# please provide as right arm, right leg, left leg, left arm
sensors = ['1a3e', '2b9f', 'a1cf', 'ba11']


# paste the data_points, sync_points and final_time of the current session from the meta file
data_points = [[2976, 82385],
               [2553, 81826],
               [2916, 82478],
               [3395, 82883]]
sync_points = [['00:28:41.850', '00:56:17.950'],
               ['00:28:41.950', '00:56:18.000'],
               ['00:28:42.100', '00:56:18.100'],
               ['00:28:41.850', '00:56:17.950']]
final_time = ['00:28:49.000', '00:56:11.000']

# create synced files (wave files and concatenated csv file)
create_synced_files(input_dir, sensors, data_points, sync_points, final_time)

# use the wave files to annotate the video data in Final Cut!
