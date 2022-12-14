# Imports
from skimage import io

def create_avgimage(files):
    average_image=np.zeros((320, 479, 320))
    count=0
    for file in files:
        average_image+=io.imread(file)[:,:,:320,0]
        print("%d/%d"%(count, len(files)),end="\r")
        count+=1
    return average_image