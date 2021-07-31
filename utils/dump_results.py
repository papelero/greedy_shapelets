import pickle
import os

def dump_object(file_name, obj):
    file = open(os.path.join('results',file_name), 'wb') 
    pickle.dump(obj,file)
    file.close()

def load_object(file_name):
    file = open(os.path.join('results',file_name), 'rb') 
    return pickle.load(file)