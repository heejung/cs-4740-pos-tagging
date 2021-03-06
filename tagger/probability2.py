from collections import Mapping

class SmoothedDistribution(Mapping):
    def __init__(self, conditioned_count):
        self._conditioned_count = conditioned_count
    
    def __iter__(self):
        return self._conditioned_count.__iter__()

    def __len__(self):
        return self._conditioned_count.__len__()
    
    def __getitem__(self, key):
        #observation table
        if self._conditioned_count.has_key("UNK"):
            w_t = key.split()
            if self._conditioned_count.has_key(w_t[0]):
                if self._conditioned_count[w_t[0]].has_key(w_t[1]):
                    return self._conditioned_count[w_t[0]][w_t[1]]
                return self._conditioned_count[w_t[0]]["SMOOTH"]
            else:
                if self._conditioned_count["UNK"].has_key(w_t[1]):
                    return self._conditioned_count["UNK"][w_t[1]]
                return self._conditioned_count["UNK"]["SMOOTH"]
        #transition matrices
        if self._conditioned_count.has_key(key):
            return self._conditioned_count[key]
        """        
        tags = key.split()
        if len(tags) > 2:
            tagtag = tag[0] + " " + tag[2]
            if self._conditioned_count.has_key(tagtag):
                return self._conditioned_count[tagtag]
            return self._conditioned_count[tag[0]]
        """
        return self._conditioned_count["SMOOTH"]
