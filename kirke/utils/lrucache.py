#!/usr/bin/env python

# source: https://www.kunxi.org/blog/2014/05/lru-cache-in-python/
import collections

class LRUCache:

    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = collections.OrderedDict()

    def get(self, key):
        try:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        except KeyError:
            return None
            
    def set(self, key, value):
        try:
            self.cache.pop(key)
        except KeyError:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
        self.cache[key] = value

class LRUCache2:

    def __init__(self, capacity):
        self.deque = collections.deque(maxlen=capacity)
        self.cache = {}

    def get(self, key):
        try:
            value = self.cache[key]
            self.deque.remove(key)  # this could be very slow
            self.deque.append(key)
            return value
        except KeyError:
            return None
        
    def set(self, key, value):
        try:
            old_value = self.cache[key]
            self.deque.remove(key)  # this could be very slow
        except KeyError:
            if len(self.deque) >= self.deque.maxlen:
                old_key = self.deque.popleft()
                del self.cache[old_key]
        self.deque.append(key)
        self.cache[key] = value

    def __str__(self):
        alist = []
        for akey in self.deque:
            alist.append((akey, self.cache[akey]))
        return 'deque = {}\nlen(dict) = {}\t{}'.format(self.deque,
                                                       len(self.cache),
                                                       str(alist))
    
if __name__ == '__main__':
    cache = LRUCache(2)
    # cache = LRUCache2(2)

    cache.set(1, 1);
    cache.set(2, 2);
    print("cache: {}".format(cache))
    value = cache.get(1);       # returns 1
    print("should be 1, got {}".format(value))
    cache.set(3, 3);    # evicts key 2
    value = cache.get(2);       # returns -1 (not found)
    print("should be None, got {}".format(value))
    cache.set(4, 4);    # evicts key 1
    value = cache.get(1);       # returns -1 (not found)
    print("should be None, got {}".format(value))    
    value = cache.get(3);       # returns 3]
    print("should be 3, got {}".format(value))        
    value = cache.get(4);       # returns 4
    print("should be 4, got {}".format(value))
