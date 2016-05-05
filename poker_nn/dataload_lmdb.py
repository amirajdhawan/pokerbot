import numpy as np
import lmdb
import caffe

def load_data_into_lmdb(path, features, labels=None):
    env = lmdb.Environment(path, map_size = features.nbytes*2, subdir=True)
    #env = lmdb.open(lmdb_name, )
    
    features = features[:,:,None,None]
    with env.begin(write=True) as txn:
        for i in range(features.shape[0]):
            datum = caffe.proto.caffe_pb2.Datum()
            
            datum.channels = features.shape[1]
            datum.height = 1
            datum.width = 1
            
            if features.dtype == np.int:
                datum.data = features[i].tostring()
            elif features.dtype == np.float: 
                datum.float_data.extend(features[i].flat)
            else:
                raise Exception("features.dtype unknown.")
            
            if labels is not None:
                datum.label = int(labels[i])
            
            str_id = '{:08}'.format(i)
            txn.put(str_id, datum.SerializeToString())

def get_data_for_case_from_lmdb(path, id):
    lmdb_env = lmdb.Environment(path, subdir=True, readonly=True)
    lmdb_txn = lmdb_env.begin()

    raw_datum = lmdb_txn.get(id)
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(raw_datum)

    feature = ""

    if len(datum.data):
        feature = np.fromstring(datum.data, dtype=np.int).reshape(
            datum.channels, datum.height, datum.width)
    else:
        feature = np.array(datum.float_data).astype(float).reshape(
            datum.channels, datum.height, datum.width)

    label = datum.label

    return (label, feature)