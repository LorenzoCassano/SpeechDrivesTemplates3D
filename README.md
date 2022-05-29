# Speech Drives Templates 3D
This repository allows you to train a model from a video/videos and to predict the gesture of a person from an audio track.<br>
This repository is based on SpeechDrivesTeamplates, it does this work on 2 coordinates (x,y) and we've expanded it to 3 coordinates (x,y,z).

## Organization
The main directory is splitted into 2 sub directories:
  <ul>
   <li>  Preprocessing3D: It expands the SpeechDrivesTeamplates's preprocessing but managing a 3rd coordinate (z), in order to create a 3d dataset.
   <li> SpeechDrivesTeamplates: It's the core of the project and it contains all the files to train a 3D model and test it.
  </ul>

The file changed in SpeechDrivesTeamplates are:
<ul>
<li><code>  core/networks/poses_recostruction/autoencoder.py</code>
<li><code> core/networks/keypoints_generation/generator.py </code>
<li><code>core/datasets/gesture_dataset.py</code>
<li><code>core/utils/keypoint_visualization.py </code>
<li><code> core/datasets/speaker_stat.py </code>
<li><code>  core/pipelines/voice2pose.py  </code>
<li><code> data_process/4_1_calculate_mean_std.py  </code>
</ul>

## Execute Preprocessing3D

## Execute Model
