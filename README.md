<!-- [![GitHub Stars](https://img.shields.io/github/stars/bwittmann/gav?style=social)](https://github.com/bwittmann/gav) -->
![visitors](https://visitor-badge.glitch.me/badge?page_id=bwittmann.gav)
# Graph Attentive Vectors (GAV)

<img src="gav_overview.png">

To run our proposed GAV framework on the ogbl-vessel benchmark, please follow the instructions below. The ogbl-vessel benchmark's data will be automatically downloaded and stored under `./dataset`.

## Installation

Please create a new virtual environment using, e.g., anaconda:

    conda create --name gav python=3.8.15

Subsequently, install the required packages:

    pip install torch==1.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-1.13.0+cu116.html

    pip install ogb tensorboard tqdm networkit

The installation was tested using Ubuntu 16.04 and CUDA 11.6. 
<!-- torch 1.13.1, torch-geometric 2.2.0, and ogb 1.3.5 -->


## Training

To train GAV, please run:

    python gav_link_pred.py --save_appendix <appendix> --gpu_id <gpu_id> --dataset ogbl-vessel

Checkpoints and tensorboard log files will be stored under `./results`.

## Testing

To test GAV's performance on an individual checkpoint, please run:

    python gav_link_pred.py --save_appendix <appendix_of_run_of_ckpt> --gpu_id <gpu_id> --dataset ogbl-vessel --only_test --continue_from <ckpt_epoch_nr>

To test GAV's performance on our provided checkpoint, please run:

    python gav_link_pred.py --save_appendix _gav --gpu_id <gpu_id> --dataset ogbl-vessel --only_test --continue_from 34


## Road Networks
### Preprocessing

To preprocess the road network datasets, please download the graph and coordinates from [here](https://www.cc.gatech.edu/dimacs10/archive/streets.shtml). The downloaded file from **Graph** should be called `edges.graph`, while the downloaded file from **Coordinates** should be called `nodes.graph`. Both files should be stored in an individual directory located at `<path_to_downloaded_files>`. Finally, run:

    python create_dataset.py --path <path_to_downloaded_files> --gpu_id <gpu_id> --dataset_name <e.g., ogbl-luxembourg_road>

Please note that in the preprocessing step, the `--dataset_name` has to start with `ogbl-` and should not include additional hyphens.

### Training and Testing

Follow the instructions above and simply state the processed dataset's name after the `--dataset` flag, omitting `ogbl-`. E.g., `--dataset luxembourg_road`.


## More Whole-Brain Vessel Graphs
### Preprocessing

To preprocess additional whole-brain vessel graphs, please download the **raw** data from [here](https://github.com/jocpae/VesselGraph).
The downloaded files should be stored in an individual directory located at `<path_to_downloaded_files>` Finally, run:

    python create_dataset.py --path <path_to_downloaded_files> --gpu_id <gpu_id> --dataset_name <e.g., ogbl-c57_tc_vessel>

Please note that in the preprocessing step, the `--dataset_name` has to start with `ogbl-` and should not include additional hyphens.

### Training and Testing

Follow the instructions above and simply state the processed dataset's name after the `--dataset` flag, omitting `ogbl-`. E.g., `--dataset c57_tc_vessel`.
