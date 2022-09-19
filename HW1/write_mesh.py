import numpy


def write_paraboloid_obj(a, b, x1min, x1max, x2min, x2max, nx1, nx2, filename):
    vnbr = nx1*nx2
    fnbr = (nx1-1)*(nx2-1)*2
    V = numpy.zeros((vnbr, 3))
    F = numpy.zeros((fnbr, 3))
    verline = 0
    x1itv = (x1max-x1min)/(nx1-1)
    x2itv = (x2max-x2min)/(nx2-1)
    for i in range(nx1):
        for j in range(nx2):
            x1 = x1min+i*x1itv
            x2 = x2min+j*x2itv
            x3 = (a*x1*x1+b*x2*x2)/2
            V[verline][0] = x1
            V[verline][1] = x2
            V[verline][2] = x3
            verline = verline+1
    fline = 0
    for i in range(nx1-1):
        for j in range(nx2-1):
            id0 = nx2*i+j+1
            id1 = nx2 * (i + 1) + j+1
            id2 = nx2 * (i + 1) + j + 2
            id3 = nx2 * i + j + 2

            F[fline] = numpy.array([id0, id1, id2])
            F[fline+1] = numpy.array([id0, id2, id3])
            fline = fline+2
    f = open(filename, "a")
    for i in range(vnbr):
        vinfo = "v "+str(V[i][0])+" "+str(V[i][1])+" "+str(V[i][2])+"\n"
        f.write(vinfo)
    for i in range(fnbr):
        finfo = "f "+str(F[i][0])+" " + str(F[i][1])+" " + str(F[i][2])+"\n"
        f.write(finfo)

    f.close()