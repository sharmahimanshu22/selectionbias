import copy

def safeUpdate(d1, d2, *str):
    # Creates a copy of the dictionary d1 and overwrites the matching key values available in d2.
    # d2 is also copied. Both d1 and d2 stays intact
    # if str is a list of strings, d2 is interpreted to be a nested dictionary.
    # In that case, it is searched iteratively for the strings to get the right dictionary to be matched to d1
    d1Copy = d1.copy( )
    d2Copy = d2.copy( )
    for s in str:
        if s in d2Copy:
            if isinstance(d2Copy[s], dict):
                d2Copy = d2Copy[s].copy()
            else:
                d2Copy = {}
        else:
            d2Copy = {}

    common_keys = set(d1Copy.keys( )) & set(d2Copy.keys( ))
    d1Copy.update({k: d2Copy[k] for k in common_keys})
    return d1Copy


def safeRemove(d, *str):
    # removes the keys in str from a copy of dictionary d and returns it. d remains intact
    dCopy = copy.deepcopy(d)
    [dCopy.pop(s, None) for s in str]
    return dCopy



