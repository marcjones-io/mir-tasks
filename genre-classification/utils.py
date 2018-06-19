# functions to help us prepare the data for the classifier
def flatten_json_dicts(b):
    val = {}
    for key in b.keys():
        if isinstance( b[key], dict ):
            get = flatten_json_dicts(b[key])
            for subkey in get.keys():
                val[ key + '_' + subkey ] = get[subkey]
        else:
            val[key] = b[key]
    return val

def flatten_json_lists(b):
    val = {}
    for key in sorted(b.keys()):
        if(type(b[key])) is list:
            for i,item in enumerate(b[key]):
                newkey = key[:key.rfind('_')]+'_'+str(i)+key[key.rfind('_'):]
                val[newkey] = item
            b.pop(key,None)
        else:
            val[key] = b[key]
    return val

def remove_non_numerical_data(b):
    val = {}
    for key in sorted(b.keys()):
        if type(b[key]) is str and key != 'genre':
            b.pop(key,None)
        else:
            val[key] = b[key]
    return val