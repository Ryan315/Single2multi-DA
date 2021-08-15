# Instructions to generate the results.
## 1. Install required dependencies.
The proposed method is written in python 3.6. Please use the requirements file to install the required dependencies.
> pip install -r requirements.txt



## 2. Run the testing code.
> python3 SCIDA.py -e

1. The checkpoint(about 180M in total) will be downloaded to 'checkpoint/' folder automatically.
2. The source and target images used for demo are placed in 'data/' folder.
3. The results will be saved to 'log/log.txt'.
4. Because of the limitation of the file size, only 50 target images are chosen randomly for demo. The results could be slightly different as the results in the paper.
