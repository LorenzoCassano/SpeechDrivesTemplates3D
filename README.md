# Speech Drives Templates 3D
This repository allows you to train a model from a video/videos and to predict the gesture of a person from an audio track.<br>
This repository is based on [SpeechDrivesTeamplates](https://github.com/ShenhanQian/SpeechDrivesTemplates), it does this work on 2 coordinates (x,y) and we've expanded it to 3 coordinates (x,y,z).

## Organization
The main directory is splitted into 2 sub directories:
  <ul>
   <li>  <b>Preprocessing3D</b>: It expands the SpeechDrivesTeamplates's preprocessing but managing a 3rd coordinate (z), in order to create a 3d dataset.
   <li><b> SpeechDrivesTeamplates</b>: It's the core of the project and it contains all the files to train a 3D model and test it.
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

## Dataset
We created a small 3D dataset based on an university teacher, but it would be better creating a grater dataset to have a better model, you can use the speakers of [Speech2Gesture](https://people.eecs.berkeley.edu/~shiry/projects/speech2gesture/index.html).

## Execute Preprocessing3D
To build a 3D dataset, we provide scripts in <code>Preprocessing3D</code>.<br>
It's necessary to run the scripts in the following order:
<ul>
<li> <code>1_1_change_fps.py</code>, it needs 2 arguments: videos directory and target directory where fps videos will be saved.
<li> <code> 1_2_video2frames.py</code> it needs 2 arguments: fps videos directory and directory where the frames will be saved (we suggest to use the same name of path in the code).
<li><code> preprocessing.py</code>, it creates the keypoints.
<li><code> fixing.py</code>, it fixes the keypoints.
<li><code>3_2_split_train_val_test.py</code>, it creates a csv file.
<li><code>4_1_calculate_mean_std.py</code>, it calculates the mean and std of each keypoint.<br> After that, to insert the mean and std in speaker_stat.py
</ul>

## Execute Model
To run the model, we suggest to use "Execute.ypnb" file in [Google Colab](https://colab.research.google.com).<br>
To run the code in local, you need to install on your device [Cuda](https://developer.nvidia.com/cuda-toolkit).<br>
You also need to create dataset and output directory and set it in the configuration files (<code>voice2pose_sdt_bp.yaml</code>).<br>

### Training
<code>python main.py --config_file configs/voice2pose_sdt_bp.yaml \
    --tag speaker_name \
    DATASET.SPEAKER speaker_name \
    SYS.NUM_WORKERS 32
</code>
