import scipy
#import HoloClass


def radav(arr, x, y, le):
	m = -1
	n = -1
	sizeofrad = le
	radsum = scipy.zeros(sizeofrad)
	numsum = scipy.zeros(sizeofrad)
	xdim, ydim = arr.shape
	for i in range(x-le, x+le):
		#n += 1
		for j in range(y-le, y+le):
			#m += 1
			if (i < xdim) and (j < ydim) and (i > 0) and (j > 0):
				temp = scipy.sqrt((i-x)**2 + (j-y)**2)
				if (round(temp) < sizeofrad):
					radsum[int(temp)] += arr[i, j]
					numsum[int(temp)] += 1
	return radsum/numsum, numsum
	

def manyradavs(start, stop, step, xyzpos, le, bg, usebg=True):
	j = 0
	h = HoloClass.Hologram(1)
	radavs = scipy.zeros((range(start,stop,step).__len__(), le))
	for i in range(start, stop, step):
		h.hol_num = i
		h.Load12BitTiff512()
		if usebg:
                        h.array = bg - h.array
		radavs[j] = radav(h.array, round(xyzpos[j,0]), round(xyzpos[j,1]), le)
		j+=1
	return radavs
		
def get_deriv(arr):
        fr, le = arr.shape
        dervs = scipy.zeros((fr,le-2),dtype=float)
        for i in range(0,fr):
                for j in range(1,le-1):
                        dervs[i,j-1] = arr[i,j-1]-arr[i,j+1]
        return dervs
		

def findmins(vec, tempsize):
        #tempsize = 20
        le = len(vec)
        j=0
        mins = scipy.zeros((40),dtype=float)
        for i in range(0,le,tempsize/2):
                temp = scipy.ndimage.minimum_position(vec[i:i+tempsize])
                if (temp[0] > 0) and (temp[0] < tempsize-1):
                        if j==0:
                                mins[j] = temp[0] + i
                                j = j+1
                        else:
                                if (mins[j-1] != temp[0]+i):
                                        mins[j] = temp[0]+i
                                        j=j+1
        return mins

def getallmins(arr):
        fr = arr.shape[0]
        ms = scipy.zeros((fr, 16),dtype=float)
        for i in range(0,fr):
                ms[i,:] = findmins(arr[i]/arr[i,0])
        return ms

def findmins2(vec, start):
        tempsize = 20
        le = len(vec)
        j=0
        mins = scipy.zeros((16),dtype=float)
        for i in range(0,len(start)):
                temp = scipy.ndimage.minimum_position(vec[start[i]-(tempsize/2):start[i]+tempsize/2])
                mins[j] = start[i]-(tempsize/2)+temp
                j=j+1
        return mins

def getallmins2(arr, start):
        fr = arr.shape[0]
        ms = scipy.zeros((fr, 16),dtype=float)
        for i in range(0,fr):
                ms[i,:] = findmins2(arr[i]/arr[i,0], start)
        return ms

def getfittedmins(arr, start, ds):
        mins = scipy.zeros(len(start),dtype=float)
        for i in range(0, len(start)):
                #print len(arr[start[i]-ds:start[i]+ds+1])
                #print len(scipy.arange(0,2*ds+1))
                pfit = scipy.polyfit(scipy.arange(0,2*ds+1),arr[start[i]-ds:start[i]+ds+1],2)
                mins[i] = start[i] - ds - (pfit[1] / (2*pfit[0]))
        print(mins[0])
        print(mins[-1])
        return mins
                
