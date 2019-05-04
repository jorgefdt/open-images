"""Helper library for downloading open images categorically."""
import sys
import os

# import t4

import pandas as pd
import requests

from tqdm import tqdm
import ratelim
from checkpoints import checkpoints
checkpoints.enable()


##
## -- UTILITIES
##

def getTemp():
    return "temp-jorge"


def _startPrintDataFrame(title):
    print(f"{title}", end='')


def _endPrintDataFrame(df, verbose=False, index=True):
    print(f" ({df.shape[0]})")
    if verbose:
        print(f"{df if index else df.to_string(index=False)}")
        print()


def _printDataFrame(title, df, verbose=False, index=True):
    _startPrintDataFrame(title)
    _endPrintDataFrame(df, verbose, index)
        

##
## -- DATA LOADERS
##

def _getClassNamesData(class_names_fp=None, verbose=False):
    """Download or load the class names pandas DataFrame"""
    _startPrintDataFrame(":: Loading class_names")
    kwargs = {'header': None, 'names': ['LabelID', 'LabelName']}
    # orig_url = "https://storage.googleapis.com/openimages/2018_04/class-descriptions-boxable.csv"
    orig_url = "../../data/class-descriptions-boxable.csv"
    retValue = pd.read_csv(class_names_fp, **kwargs) if class_names_fp else pd.read_csv(orig_url, **kwargs)
    _endPrintDataFrame(retValue, verbose)
    return retValue


def _getTrainedBoxedData(train_boxed_fp=None, verbose=False):
    """Download or load the boxed image metadata pandas DataFrame"""
    _startPrintDataFrame(":: Loading train_boxed")
    # TODO: setting index_col should not be necessary in this and the next section, update the save-to-disk code
    # orig_url = "https://storage.googleapis.com/openimages/2018_04/train/train-annotations-bbox.csv"
    # orig_url = "../../data/train-annotations-bbox.csv"
    orig_url = "../../data/train-annotations-bbox-1000.csv"
    retValue = pd.read_csv(train_boxed_fp, index_col=0) if train_boxed_fp else pd.read_csv(orig_url)
    _endPrintDataFrame(retValue, verbose)
    return retValue


def _getImageIdsData(image_ids_fp=None, verbose=False):
    """Download or load the image ids metadata pandas DataFrame"""
    _startPrintDataFrame(":: Loading image_ids")
    # orig_url = "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv"
    orig_url = "../../data/train-images-boxable-with-rotation.csv"
    retValue = pd.read_csv(image_ids_fp, index_col=0) if image_ids_fp else pd.read_csv(orig_url)
    _endPrintDataFrame(retValue, verbose)
    return retValue


##
## -- IO UTILS
##

@ratelim.patient(5, 5)
def _download_image(url, pbar):
    """Download a single image from a URL, rate-limited to once per second"""
    r = requests.get(url)
    r.raise_for_status()
    pbar.update(1)
    return r


@ratelim.patient(5, 5)
def _download_image_safely(url, pbar):
    """Download a single image from a URL, rate-limited to once per second"""
    try:
        return _download_image(url, pbar)
    except requests.exceptions.HTTPError:
        msg = "ERROR: Can't download image {0}.".format(url)
        print(msg)



def _write_image_file(r, image_name):
    """Write an image to a file"""
    filename = f"{getTemp()}/{image_name}"
    with open(filename, "wb") as f:
        f.write(r.content)


def _write_image_file_safely(r, image_name):
    """Write an image to a file"""
    try:
        if r != None: _write_image_file(r, image_name)
    except AttributeError:
        msg = "ERROR: Can't write image {0}.".format(image_name)
        print(msg)

##
## -- HELPERS
##

def _find_labels_used_in_train_boxed():
    """Calculates the labelNames used by the train_boxed data file."""
    class_names = _getClassNamesData()
    train_boxed = _getTrainedBoxedData()

    # Get the labelIDs used in train_boxed
    print(f":: Calculating usedLabelIDs", end='')
    usedLabelIDs = train_boxed[['LabelName']]
    usedLabelIDs = usedLabelIDs.drop_duplicates() \
                        .sort_values("LabelName") \
                        .reset_index() \
                        .drop(columns=['index']) \
                        .rename(columns={'LabelName':'LabelID'})
    _endPrintDataFrame(usedLabelIDs, verbose=False)

    # Find labels 
    print(f":: Mapping used label IDs to label names", end='')
    usedLabels = class_names.merge(usedLabelIDs).LabelName.sort_values()
    _endPrintDataFrame(usedLabels, verbose=True, index=False)



##
## -- MAIN
##

def downloadTrainingImagesForCategories(categories,  # packagename, registry,
             class_names_fp=None, train_boxed_fp=None, image_ids_fp=None):
    """Download images in categories from flickr"""
    print(f":: downloadTrainingImagesForCategories({categories})")

    class_names = _getClassNamesData(class_names_fp)
    train_boxed = _getTrainedBoxedData(train_boxed_fp)
    image_ids = _getImageIdsData(image_ids_fp)

    # Get category IDs for the given categories and sub-select train_boxed with them.
    print(f":: Mapping labels")

    # i = 2
    # f = class_names.set_index('LabelName');     print(f":: {i}: {f}\n"); i = i+1
    # f = f.loc[categories, 'LabelID'];           print(f":: {i}: {f}\n"); i = i+1
    # f = f.to_frame();                           print(f":: {i}: {f}\n"); i = i+1
    # f = f.reset_index();                        print(f":: {i}: {f}\n"); i = i+1
    # f = f.set_index('LabelID');                 print(f":: {i}: {f}\n"); i = i+1
    # f = f['LabelName'];                         print(f":: {i}: {f}\n"); i = i+1
    # f = dict(f);                                print(f":: {i}: {f}\n"); i = i+1

    label_map = dict(class_names.set_index('LabelName').loc[categories, 'LabelID']
                     .to_frame().reset_index().set_index('LabelID')['LabelName'])
    print(f":: label_map: {label_map}")
    
    labelIDs = set(label_map.keys())
    print(f":: labelIDs: {labelIDs}")

    relevant_training_images = train_boxed[train_boxed.LabelName.isin(labelIDs)]
    print(f":: relevant_training_images:\n{relevant_training_images}\n")

    # Start from prior results if they exist and are specified, otherwise start from scratch.
    relevant_flickr_urls = (relevant_training_images.set_index('ImageID')
                            .join(image_ids.set_index('ImageID'))
                            .loc[:, 'OriginalURL'])
   
    # Eliminate duplicate image urls
    relevant_flickr_urls = relevant_flickr_urls.drop_duplicates()
    print(f":: relevant_flickr_urls2:\n{relevant_flickr_urls}\n")

    relevant_flickr_img_metadata = (relevant_training_images.set_index('ImageID').loc[relevant_flickr_urls.index]
                                    .pipe(lambda df: df.assign(LabelValue=df.LabelName.map(lambda v: label_map[v]))))
    print(f":: relevant_flickr_img_metadata:\n{relevant_flickr_img_metadata}\n")

    remaining_todo = len(relevant_flickr_urls) if checkpoints.results is None else\
        len(relevant_flickr_urls) - len(checkpoints.results)
    print(f"Parsing {remaining_todo} images "
          f"({len(relevant_flickr_urls) - remaining_todo} have already been downloaded)")

    # Download the images
    with tqdm(total=remaining_todo) as progress_bar:
        # relevant_image_requests = relevant_flickr_urls.safe_map(lambda url: _download_image(url, progress_bar))
        relevant_image_requests = relevant_flickr_urls.safe_map(lambda url: _download_image_safely(url, progress_bar))
        progress_bar.close()

    # Initialize a new data package or update an existing one
    # p = t4.Package.browse(packagename, registry) if packagename in t4.list_packages(registry) else t4.Package()

    # Write the images to files, adding them to the package as we go along.
    if not os.path.isdir(f"{getTemp()}/"):
        os.mkdir(f"{getTemp()}/")

    for ((_, r), (_, url), (_, meta)) in zip(relevant_image_requests.iteritems(), relevant_flickr_urls.iteritems(),
                                             relevant_flickr_img_metadata.iterrows()):
        image_name = url.split("/")[-1]
        image_label = meta['LabelValue']
        _write_image_file_safely(r, image_name)



def myTest():
    df = pd.DataFrame({'month': [1, 4, 7, 10],
                        'year': [2012, 2014, 2013, 2014],
                        'sale': [55, 40, 84, 31]})
    df = df.set_index(['year', 'month'])                       
    # _printDataFrame(":: sample 1", df, verbose=True, index=True)
    print(f"{df}")

    

if __name__ == '__main__':
    categories = sys.argv[1:]
    downloadTrainingImagesForCategories(categories)
    # _find_labels_used_in_train_boxed()
    # myTest()
    print(f":: Done")
