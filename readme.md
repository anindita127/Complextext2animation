# Synthesis of Compositional Animations from Textual Descriptions

Our code is tested on the following OS: 
* Ubuntu 18.04 LTS
* Windows 10

## Pre-requisites
### Code
* Python >= 3.6.10
* Pytorch >= 1.5.1
* conda >= 4.9.2 (optional but recommended)

All other pre-requisites are in the file `environment.yml`. Steps to install:

1. Create the conda environment:
```
conda env create -f environment.yml
conda activate text2motion
```
2. Install pytorch following the [official instructions](https://pytorch.org/get-started/locally/).

3. Install all other requirements:
```
pip install -r requirements.txt
```


We denote the base directory of our project as `$BASE`.
### Data
Download the [KIT Motion-Language dataset](https://motion-annotation.humanoids.kit.edu/dataset/) and unzip it in the `$BASE/dataset` folder.

## Running the code
1. First, run:
```cd $BASE/src```

2. Preprocessing the data: 
```python data.py```
This will create `quat.csv`, `fke.csv` and `rifke.csv` files for each input data.  

3. Calculate the mean and variance for Z-Normalization:
```python dataProcessing/meanVariance.py -mask '[0]' -feats_kind rifke -dataset KITMocap -path2data ../dataset/kit-mocap -f_new 8```
The outputs will be saved in `$BASE/src/dataProcessing/meanVar`.

4. [Optional] Train the model:
```python train_model_GD.py -batch_size 32 -curriculum 0 -dataset KITMocap -early_stopping 1 -exp 1 -f_new 8 -feats_kind rifke -lr 0.001 -mask "[0]" -model hierarchical_twostream -cpk t2m -num_epochs 300 -path2data ../dataset/kit-mocap -render_list subsets/render_list -s2v 1 -save_dir save/model -tb 1 -time 32 -transforms "['zNorm']"``` 

We also provide a pre-trained model [here](https://drive.google.com/file/d/1qt4mjtbPUYILJyjFFapA38-O9T_5hkcP/view?usp=sharing). To use it, extract the contents of the downloaded zip file `pretrained_model.zip` inside the folder `$BASE/src/save/model`.

5. Testing the trained model: 
```python sample_wordConditioned.py -load save/model/$weights```
where `$weights` is the pre-trained network parameters, saved as a `.p` file. The outputs are generated inside a folder `$BASE/src/save/model/$OUTPUT` created automatically.


6. Calculating the error metrics:
```
python eval_APE.py -load save/model/$weights
python eval_AVE.py -load save/model/$weights
python eval_CEE_SEE.py -load save/model/$weights
```

7. Rendering the output files:
```python render.py -dataset KITMocap -path2data ../dataset/kit-mocap -feats_kind rifke -clean_render 0 -save_dir save/model/$OUTPUT```
