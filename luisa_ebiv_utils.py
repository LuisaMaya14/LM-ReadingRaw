import os
import numpy as np
from struct import unpack,pack


def LoadEventsMetativision(
    pathRAW,    #name of the RAW file
    duration=0  #in microsec
    ):
    """
    With this function is possible to read all the events in the 
    RAW file by using the metavision software.
    """
    from metavision_core.event_io import RawReader
    
    metavisionRAW = RawReader(pathRAW, max_events= 20000000)
    
    height, width = metavisionRAW.get_size()
    print("Image size: W" +str(width)+ "x H" +str(height)) #== (640, 480)
    
    #Calculate the duration of the file
    if duration<=0:         #This step is to veryfy if the "duration" has not been calculated
        count = 0
        n_events = 10000    #Number of events that we are going to see in the buffer
        while not metavisionRAW.is_done(): #Is going to enter into the loop until it finished examining the entire file
            evs = metavisionRAW.load_n_events(n_events)
            count +=len(evs)
            duration = evs ['t'][-1]
            print('File length: '+str(duration/1e6)+ ' s with '+str(count)+ ' events')
            
                
    
    metavisionRAW = RawReader(pathRAW, max_events= 20000000)
    ev = metavisionRAW.load_delta_t(duration) #Loads all events in the specified time range in this case 'duration'
    print('Loaded '+str(ev.size)+' events for '+str(duration/1e6)+ ' sec')
    
    return {'ev':ev, 'image_size':[height, width]}

def EBISaveEvents(
    pathOUT:str, 
    loadEvMv, 
    duration = 0
    ):
    """
    This function save the information of the RAW (current event set)
    as binary in 64-bit evt-format [xypt] 
    [t 31bit][p 1bit][x 16bit][y 16bit]
    """
    #Determine the duration of the file, with user data or by reading the t max of the file.
    
    ev = loadEvMv['ev']
    t1 = 0                      #If you have an offset it would be the offset
    t2 = t1 + duration
    t_max = ev['t'].max() + 1
    if duration <=0:            #If you do not know the duration put 0 in the definition and the code will calculate the t2 as the maximum,
        t2=t_max                #if you know the duration simply put the time and do not enter to the loop.
        
    # Save events in numpy arrays
    idx = np.where((ev['t']>=t1) & (ev['t'] <= t2))
    evTime_OUT = ev['t'][idx]
    evX_OUT = ev['x'][idx]
    evY_OUT = ev['y'][idx]
    evPol_OUT = ev['p'][idx]
    eventCount = evTime_OUT.size
    
    #Package them in 64 bits
    packedData = (evX_OUT.astype(np.int64)) | (evY_OUT.astype(np.int64) << 16) \
                 | (evPol_OUT.astype(np.int64) << 32)\
                    | (evTime_OUT.astype(np.int64) << 33)
    
    #Header
    h,w = loadEvMv['image_size']
    duration = evTime_OUT.max() - t1
    sig = b'EVT3'
    hdrLen = 64
    fileSize = hdrLen + (eventCount*8)
    fillVal = 0
    header = pack('4s3Q4L2Q', sig, fileSize, eventCount, fillVal, duration, 
                  hdrLen, w, h, fillVal,fillVal)
    with open(pathOUT, "wb") as binary_file:
        nBytesWritten = binary_file.write( header )
        nBytesWritten2 = binary_file.write( packedData )
        binary_file.close()
        print('Wrote '+str(eventCount)+' events of duration '+str(duration/1e6)+' sec into EVT file with '+str((nBytesWritten+nBytesWritten2))+' bytes')
        return True
    return False

def LoadEventsEVT(
    pathEVT:str, 
    dbg = False
    ):
    """
    This function load events of a binary file in 64-bit format
    and convert it into a numpy array
    
    data in the numpy array:
            event time 't' in [usec], 
            event position 'x' in [pixel], 
            event position 'y' in [pixel], 
            event polarity 'p' [0,1]
            'image_size' of sensor (height,width) in [pixel]
    """
    #Open the file as a binary
    with open(pathEVT, "rb") as binary_file:
        binary_file.seek(0, 2)  # Seek the end
        binary_file.seek(0)     # Seek the beggining 
        header = binary_file.read( 48 )
        sig, fileSize, eventCount, timeStamp, duration, hdrLen, w, h = unpack('4s3Q4L',header)
        if dbg:
            print("signature:      " + str(sig))
            print("file size:      " + str(fileSize))
            print("header lenth:   " + str(hdrLen))
            print("event count:    " + str(eventCount))
            print("duration [us]:  " + str(duration))
            print("timeStamp [us]: " + str(timeStamp))
            print("image size [h,w]" + str([h,w]))
        
        binary_file.seek(hdrLen)    #The seek function goes to the end of the hdrLen which is the length of the header.
        
        #unpacks in two columns one with x & and the other with polarity and time
        dataIN = np.fromfile(binary_file, 
                             dtype=np.uint32, count=(eventCount*2)).reshape(eventCount,2)
        binary_file.close()
        
        #Unpacks into 4 variables x, y, t and polarity
        evTime = (dataIN[:,1] >> 1)
        evY = (dataIN[:,0] >> 16)
        evX = (dataIN[:,0] & 0xFFFF)
        evPol = (dataIN[:,1] & 0x1)
        
        print('Time: '+ str(evTime))
        
        return {'t':evTime, 'x':evX, 'y':evY, 'p':evPol, \
                'time_stamp':timeStamp, 'image_size':[h,w] }
        



#This is the main where you specify which functions to run
if __name__ == '__main__':
    
    pathRAW = 'wallflow4_dense_3.raw'
    pathOUT = 'wallflow4_dense_3.evt'
    pathEVT = 'wallflow4_dense_3.evt'
    
    loadEvMv = LoadEventsMetativision(pathRAW, duration = 0)
    EBISaveEvents(pathOUT, loadEvMv, duration = 0)
    LoadEventsEVT(pathEVT, dbg = True)