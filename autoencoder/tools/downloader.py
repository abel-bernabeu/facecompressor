#
# Script to retrieve the dataset 
import os
import sys
import urllib3
import shutil
import tarfile
import psutil
#
# TODO --> is it going to be text plain here??
# username and password 
username = 'wolftau'
password = 'wtal997'
#
# check if the current directory have enough space for the dataset
hdd = psutil.disk_usage('/')
print('Total: %i GiB' % int(hdd.total / (2**30)))
print('Used: %i GiB' % int(hdd.used / (2**30)))
print('Free: %i GiB' % int(hdd.free / (2**30)))
#
# abort if not enough space (size of the dataset + decompress folder aprox 50gb)
if int(hdd.free / (2**30)) < 50:
    print('The current drive does not have enough space to alocate the dataset.')
    sys.exit(0)
# 
# This the absolute path of the dataset to be downloaded.
url = 'http://www.cslab.openu.ac.il/download/wolftau/YouTubeFaces.tar.gz'
filename='YouTubeFaces.tar.gz'
path=os.getcwd() + '/YouTubeFaces.tar.gz'
#
# create the objects for the http request and the basic authentication
http = urllib3.PoolManager()
#
basic_auth = username + ':' + password
headers = urllib3.util.make_headers(basic_auth=basic_auth)
r = http.request('GET', url, preload_content=False, headers=headers)
#
# execute the request to retrieve the dataset.
print('Retrieving the Dataset. Might take a while...')
with http.request('GET', url, preload_content=False, headers=headers) as r, open(path, 'wb') as out_file:       
    shutil.copyfileobj(r, out_file)

# open and extract the files
tar = tarfile.open(path)
files = tar.getmembers()
print('Starting the extraction. Files to be extracted --> %s' % len(files))
#
tar.extractall()
tar.close()
#
# rename the dataset dir after the decompression then delete tar.gz
os.rename('YouTubeFaces', 'data')
os.remove('YouTubeFaces.tar.gz')
