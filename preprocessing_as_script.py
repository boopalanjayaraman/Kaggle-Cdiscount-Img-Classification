import numpy as np
import pandas as pd
import io
import bson
import matplotlib.pyplot as plt
from skimage.data import imread
import multiprocessing as mp
import pickle
import time
import os.path

NCORE = 8
all_categories = mp.Manager().list()

all_categories_array = np.array([])

#categories to int dictionary
categ_to_int = {}
int_to_categ = {}

#total number of items in the list
n_train = 7069896 #from kaggle page
n_test = 1768182 #from kaggle page
n_example = 100 #from kaggle page

all_categories_filename_format = 'allcategoriesdata_{0}.p'
train_data_batch_file_format = 'training_batches/train_data_batch_{0}_{1}.p'
test_data_batch_file_format = 'testing_batches/test_data_batch_{0}_{1}.p'

show_every = 10000

def process_all_categories(filepath):
    """
    processes all categories and forms the list
    : filepath: file path
    """
    process_filename = filepath[filepath.rfind('/')+1:]
    filename_suffix = process_filename.replace('.bson','')
    categories_filename = all_categories_filename_format.format(filename_suffix)
    if os.path.isfile(categories_filename):
        print('File already exists. Seems already it is processed.')
        return

    def process_record_multicore_category(queue, iolock):
        while True:
            record = queue.get()
            if record is None:
                break
            
            all_categories.append(record['category_id'])
    
    queue = mp.Queue(maxsize=NCORE)
    iolock = mp.Lock()
    pool = mp.Pool(NCORE, initializer=process_record_multicore_category, initargs=(queue, iolock))
    
    #loading data from file
    data = bson.decode_file_iter(open(filepath, 'rb'))
    
    print('Starting to go through the file. Time: {0}'.format(time.ctime()))
    for c, record in enumerate(data):
        queue.put(record)
        if c % 100000 ==0:
            print ('records processed: {0}, time: {1}'.format(c, time.ctime()))
    
    # tell workers we're done and join the stuff
    for _ in range(NCORE):
        queue.put(None)
    pool.close()
    pool.join()
    print('File is processed. Time: {0}'.format(time.ctime()))
    
    all_categories_array = np.array(list(set(all_categories)))

    #process the categories and save them
    process_all_categories_array(all_categories_array, categories_filename)
    print('all categories processed.')
	
#data record preprocess sub-function 
def process_record_train(record, final_data):
    """
    processes each record from the training / test file during preprocessing function execution for training dataset
    : record: record to be processed
    : return: void
    """ 
    product_id = record['_id']
    category_id = record['category_id']
    for e, pic in enumerate(record['imgs']):
        picture = imread(io.BytesIO(pic['picture']))    
        #adding a record for each image with same product id and category id (ungrouping the images)
        flattened_picture = np.reshape(picture, -1)
        input_row = np.array(flattened_picture, dtype='int16')
        input_row = np.insert(input_row, 0, categ_to_int[category_id]) #will be pushed to second column 
        #pushed int value of category to reduce memory size so we can stay with int16
        input_row = np.insert(input_row, 0, product_id)
        final_data.append(input_row)
		
#data record preprocess sub-function for test data set
def process_record_test(record, final_test_data):
    """
    processes each record from the training / test file during preprocessing function execution for test data set
    : record: record to be processed
    : return: void
    """
    product_id = record['_id']  
    for e, pic in enumerate(record['imgs']):
        picture = imread(io.BytesIO(pic['picture']))
        #adding a record for each image with same product id and category id (ungrouping the images)
        flattened_picture = np.reshape(picture, -1)
        input_row = np.array(flattened_picture, dtype='int16')
        input_row = np.insert(input_row, 0, product_id)
        final_test_data.append(input_row)
		
#data preprocess function 
def process_training_file(data, enum_start=None, limit = None, file_suffix=''):
    """
    processes the training file and saves them to batch files for loading them later
    : filepath: path of the training file
    : return: void
    """
    final_data = list()
    final_data_array = np.array([])
    
    #loading data from file
    print('Starting to go through the Set. Time: {0}'.format(time.ctime()))
    
    init =  0 if enum_start == None else enum_start
    for c, record in enumerate(data, start=init):
        if(c % show_every ==0):
            print('processed records: {0}'.format(c))
        if(c > limit):
            break
        process_record_train(record, final_data)
        
    print('File is processed. Time: {0}'.format(time.ctime()))
    
    final_data_array = np.array(final_data, dtype='int16')
    
    print(final_data_array.shape)
    print(final_data[0])
    
    #save preprocessed data to batch files after one hot encoding them
    save_preprocessed_data(final_data_array, file_suffix)
    print('Preprocessing is done and saved. Time: {0}'.format(time.ctime()))
	
#test data preprocess function 
def process_test_file(data, enum_start=None, limit=None, file_suffix=''):
    """
    processes test file and saves the output to batch files for loading them later
    : filepath: path of the test file on disk
    : return: void
    """
    final_test_data = list()
    final_test_data_array = np.array([])
    
    #loading data from file
    print('TestFile: Starting to go through the Set. Time: {0}'.format(time.ctime()))
    
    init =  0 if enum_start == None else enum_start
    
    for c, record in enumerate(data, start=init):
        if(c % show_every==0):
            print('processed records: {0}'.format(c))
        if(c > limit):
            break
        process_record_test(record, final_test_data)
    
    print('TestFile: File is processed. Time: {0}'.format(time.ctime()))
    
    final_test_data_array = np.array(final_test_data, dtype='int16')
    
    #save preprocessed data to batch files after one hot encoding them
    save_preprocessed_test_data(final_test_data_array, batch, file_suffix)
    print('Preprocessing is done and saved. Time: {0}'.format(time.ctime()))
	
def process_all_categories_array(all_categories_array, processed_filename):
    """
    processes all categories found in training data and creates dictionaries for faster reference
    : all_categories_array: array that contains all categories to form one hot encoding
    : return: void
    """
    global categ_to_int, int_to_categ
    categories_length = len(all_categories_array)
    categ_to_int = { categ:idx for idx, categ in enumerate(all_categories_array) }
    int_to_categ = { idx:categ for idx, categ in enumerate(all_categories_array) }
    
    pickle.dump((categ_to_int, int_to_categ), open(processed_filename, 'wb'))
	
def load_categ_to_int_dicts(data_file_path):
    """
    restores categ_to_int and int_to_categ object dictionaries from saved state files if exist
    : data_file_path: actual data file path - to represent the mode (train or train example)
    """
    process_filename = data_file_path[data_file_path.rfind('/')+1:]
    filename_suffix = process_filename.replace('.bson','')
    categories_filename = all_categories_filename_format.format(filename_suffix)
    
    with open(categories_filename, 'rb') as f:
        
        global categ_to_int, int_to_categ
        
        categ_to_int, int_to_categ = pickle.load(f)

def create_one_hot_label(original_label, label_length, one_hot_labels):
    """
    creates one hot label for a given original label value. A sub function for multi core processing of one hot encode function
    : label_length: length of label to initialize the array
    : one_hot_labels: the array that contains all one hot label
    : return: void
    """
    one_hot_label = np.zeros(label_length, dtype='int16')
    
    #commenting below line since now conversion to index happens while preparing the matrix.
    #one_hot_label[categ_to_int[original_label]] = 1
    #so changing it to
    one_hot_label[original_label] = 1
    
    one_hot_labels.append(one_hot_label)

def one_hot_encode(data_batch):
    """
    creates one hot encoded label for the given data batch using multi-core processing
    : data_batch: the sub-section of original final training data
    : return: array of one hot encoded label
    """
    one_hot_labels = list()
    label_length = len(categ_to_int)
    
    #print(data_batch)
    
    for i in range(len(data_batch)):
        original_label = data_batch[i][1] # category column
        create_one_hot_label(original_label, label_length, one_hot_labels)

    one_hot_labels = np.array(list(one_hot_labels))
    
    return one_hot_labels
	
def save_preprocessed_data(final_data_array, file_suffix=''):
    """
    saves preprocessed data array into batch files using pickle
    : final_data_array: data array formed from file
    : batches: number of files to be created for saving
    : return: void
    """
    final_data_length = len(final_data_array)
    # one hot encode all the categories / labels
    labels_array = one_hot_encode(final_data_array)
        
    features = np.delete(final_data_array, 1, axis=1)
    labels =  labels_array

    filename = train_data_batch_file_format.format(0, file_suffix)
    pickle.dump((features, labels), open(filename, 'wb'))
		
def save_preprocessed_test_data(final_test_data_array, file_suffix = ''):
    """
    saves preprocessed data array into batch files using pickle
    : final_test_data_array: test data array formed from file
    : batches: number of files to be created for saving
    : return: void
    """
    final_data_length = len(final_test_data_array)
    
    features = final_test_data_array
    filename = test_data_batch_file_format.format(0, file_suffix)
    pickle.dump((features), open(filename, 'wb'))
		

def preprocess_test_batches(filepath, override_batch=None):
    """
    preprocesses batches and saves them in batches to end up losing data due to long running processes
    : filepath: path of file to be processed
    """
    input_data = bson.decode_file_iter(open(filepath, 'rb'))
    
    limit = 10000
    batches_count = int(n_train / limit)
    batch_range = batches_count if override_batch is None else override_batch
    for batch_idx in range(batch_range):
        print('starting with batch: {0}'.format(batch_idx))
        result_filename = test_data_batch_file_format.format(0, batch_idx)
        if os.path.isfile(result_filename):
            print('Batch File: {0} already exists. Seems already it is processed. Moving on..'.format(result_filename))
            continue
        try:
            process_test_file(input_data, enum_start=batch_idx*limit, limit=(batch_idx+1)*limit, file_suffix=batch_idx)
        except:
            print("Error in processing training batch:{0}, limit:{2}, error:{1}".format(batch_idx, sys.exc_info()[0], limit))
        
    print('all test files are preprocessed. cool!')
	

def preprocess_training_batches(filepath, override_batch=None):
    """
    preprocesses batches and saves them in batches to end up losing data due to long running processes
    : filepath: path of file to be processed
    """
    input_data = bson.decode_file_iter(open(filepath, 'rb'))
    
    limit = 10000
    batches_count = int(n_train / limit)
    batch_range = batches_count if override_batch is None else override_batch
    for batch_idx in range(batch_range):
        print('starting with batch: {0}'.format(batch_idx))
        result_filename = train_data_batch_file_format.format(0, batch_idx)
        if os.path.isfile(result_filename):
            print('Batch File: {0} already exists. Seems already it is processed. Moving on..'.format(result_filename))
            continue
        try:
            process_training_file(input_data, enum_start=batch_idx*limit, limit=(batch_idx+1)*limit, file_suffix=batch_idx)
        except:
            print("Error in processing training batch:{0}, limit:{2}, error:{1}".format(batch_idx, sys.exc_info()[0], limit))
        
    print('all training files are preprocessed. cool!')


if __name__ =='__main__':
	print('loading categ_to_int dictionaries from saved file.')
	load_categ_to_int_dicts('data/train.bson')
	
	print('starting to preprocess the training set.')
	preprocess_training_batches('data/train.bson')
	
