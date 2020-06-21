Handwritten Text Recognition (HTR) system implemented using Pytorch and trained on the Bentham/IAM/Rimes/Saint Gall/Washington offline HTR datasets. This Neural Network model recognizes the text contained in the images of segmented texts lines.

Data pre-processing is totally based on this awesome repository of [handwritten text recognition](https://github.com/arthurflor23/handwritten-text-recognition).
Data partitioning (train, validation, test) was performed following the methodology of each dataset. 

Model building was done using the the transformer architecture. 
Recentely facebook research realeased a [paper](https://github.com/facebookresearch/detr), where they used transformer for object detection. I made few changes to their model so that it can run on text recognition, 

## Datasets supported

a. [Bentham](http://transcriptorium.eu/datasets/bentham-collection/)

b. [IAM](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)

c. [Rimes](http://www.a2ialab.com/doku.php?id=rimes_database:start)

d. [Saint Gall](http://www.fki.inf.unibe.ch/databases/iam-historical-document-database/saint-gall-database)

e. [Washington](http://www.fki.inf.unibe.ch/databases/iam-historical-document-database/washington-database)

## Requirements

- Python 3.x
- OpenCV 4.x
- editdistance
- Pytorch 1.x

## Command line arguments

- `--source`: dataset/model name (bentham, iam, rimes, saintgall, washington)
- `--transform`: transform dataset to the HDF5 file
- `--image`: predict a single image with the source parameter
- `--train`: train model using the source argument
- `--test`: evaluate and predict model using the source argument
- `--norm_accentuation`: discard accentuation marks in the evaluation
- `--norm_punctuation`: discard punctuation marks in the evaluation
- `--epochs`: number of epochs
- `--batch_size`: number of the size of each batch
- `--lr`: Learning rate

## Tutorial (Google Colab/Drive)

A Jupyter Notebook is available to demo run, check out the **[tutorial](https://colab.research.google.com/drive/1rCPaksWk7SAH4crOVYVzUaWsKbz2i3jE?authuser=1#scrollTo=rQew0_CkacDU)** on Google Colab/Drive.

**Notes**:

* Model used is from DETR(facebook research) notebook but in there paper they perfromed few more steps.
* For improving the results few more things can be done:
    * Using the warmup steps
    * Using sine positional encodings for image vector.
    * Trying more FC layers before output.
    * Trying different parameters of Transformer.
    * Trying different backbone model for getting feature vector of image.
* Training took ~20 hrs on google colab. where as [arthurflor23](https://github.com/arthurflor23/handwritten-text-recognition) can be trained in ~8hrs.
* Purpose of this project was to showcase the power of Transformer ie: You can use them anywhere.
