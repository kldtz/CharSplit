import csv
import sys
import re

#csv.field_size_limit(sys.maxsize)

# this reads an arff file using the python CSV reader.  Arffs are basically CSV files with type information anyway
def getArffReader( handle ):
    headers = []
    
    while 1:
      line = handle.readline()
      if not line:
        break    
      line = line.strip()
      parts = line.split()

      if(len(parts) == 0):
        continue
      if(parts[0] == '@attribute'):
        headers.append(parts[1])
      if(parts[0] == '@data'):
        curPos = handle.tell()
        line = handle.readline().strip()
        if len(line) > 0:
          # put it back
          handle.seek(curPos)
        break        

 
    filteredHandle = readConvertNewline(handle)
    #special CSV dialect for arff files
    reader = csv.reader(filteredHandle,quotechar="'",escapechar='\\',doublequote=False)
    return (reader,headers)

def readConvertNewline(handle):
  p_n = re.compile(r'\\n') 
  p_r = re.compile(r'\\r') 
  for line in handle:
    line = p_n.sub(r'\n',line)            
    line = p_r.sub(r'\r',line) 
    yield line
            
# make sure each row has the same number of cols as the first one
def checkCSVCols( reader ):
    numcols = 0;
    for row in reader:
        mycols = len(row)
        if(numcols == 0):
            numcols = mycols
        if(numcols != mycols):
            print ('inconsistent number of columns in this row')
            print ('| '.join(row))
    print ('There were ',numcols,' columns')

# write to a CSV that python's standard reader can handle
def writeToStandardCSV(reader,headers,name):
    with open(name,'w',newline='') as csvfile:
        w = csv.writer(csvfile)
        print (headers[0:82])
        w.writerow(headers)
        for row in reader:
            w.writerow(row)

if __name__ == "__main__":
  print ('converting '+ sys.argv[1])
  with open(sys.argv[1],newline='') as csvfile:    
    reader,headers = getArffReader(csvfile)
    #checkCSVCols(reader)
    writeToStandardCSV(reader,headers,"std.csv")

    
