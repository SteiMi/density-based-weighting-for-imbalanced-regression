# DeepSD with DenseLoss

 __Everything here except for DenseLoss and the Kubernetes integration was built by the DeepSD authors! We were not involved in DeepSD's development!__

This code contains the Stacked Super-resolution CNN proposed in the KDD paper [DeepSD](http://www.kdd.org/kdd2017/papers/view/deepsd-generating-high-resolution-climate-change-projections-through-single). We obtained it from [GitHub](https://github.com/tjvandal/deepsd) and adapted it for use with DenseLoss and Kubernetes.

## Dependencies

The current codebase has only been tested with Python2, not Python3. Major dependencies include Tensorflow and Gdal libraries.
The user must install these independently of this package.  I will try to add more support regarding dependencies at a later time.

## Usage

### Quick Look

To run the example code execute the following on the terminal: <br>
`python prism.py` <br>
`bash run_job.sh` <br>
`python inference.py` <br>

### Configuration File

`config.ini` provides and example configuration file allowing one to selection prism data options, file directories, architecture selection, and how many models to stack. All of the following scripts which download and process the data, train the models, and do inference, rely directly on this configuration file.  One can make their own config file and include it as an argument to the scripts.


### Download and Process Data 
 
`prism.py` -- To simpilify the example, I download year 2014 for training and 2015 for testing (as set in `config.ini`), but given the high resolution, the data size is still a couple gbs. Training data is processed by selecting sub-images of size 38x38 and saved to a tfrecords file. The test set contains full prism precipitation maps and saved to corresponding tfrecord files.
 
### Train Model

`train.py` -- This file handles the heavy lifting for reading in the data, building the graph, and learning the parameters. The `--model_number` parameter allows one to select which configured model to train, ie. what resolutions and data. Tensorboard summary files will be saved inside the configured scratch directory. The checkpoints are saved in the scratch directory which contain all then necessary information for inference (../scratch/DEEPSD_MODEL_NAME/models/NAME_OF_MODELX/).

`train.py` relies solely on the tfrecord files written by `prism.py`.
Note: I mostly work on ../scratch_mini/ which contains only images from lowest to second-lowest resolution.  

`run_job.sh` -- Edit this file to use a single script to train multiple models.

### Kubernetes workflow

* Build image with `./build_docker.sh`
* Use `start_k8s_job.sh` or `start_batch_k8s_job.sh` from the `k8s` folder

### Inference

`inference.py` -- To downscale we need to join all the trained models to a single graph. This script loops through all the models in `config.py` to freeze, join, and apply inference. This script can easily be edited to include GCM outputs.

`inference.py` relies solely on netcdf data and not on the tfrecords files.