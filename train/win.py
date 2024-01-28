import numpy

def win (structure1, structure2):
    sum1=numpy.sum(structure1)
    sum2=numpy.sum(structure2)
    if(sum1>sum2):
        return 1,0
    elif(sum1<sum2):
        return 0,1
    else:
        return 1,1