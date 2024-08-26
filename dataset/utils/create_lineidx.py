import os

def generate_lineidx_file(filein, idxout):
    idxout_tmp = idxout + '.tmp'
    with open(filein, 'r') as tsvin, open(idxout_tmp,'w') as tsvout:
        fsize = os.fstat(tsvin.fileno()).st_size
        fpos = 0
        while fpos!=fsize:
            tsvout.write(str(fpos)+"\n")
            tsvin.readline()
            fpos = tsvin.tell()
    os.rename(idxout_tmp, idxout)
    
for file in os.listdir("/data6/myt/SummaryCaption/data/caption"):
    if(file.split('.')[-1]=='tsv'):
        tmp = '.'
        idxout = tmp.join(file.split('.')[:-1])+'.lineidx'
        # print(idxout)
        generate_lineidx_file(os.path.join("/data6/myt/SummaryCaption/data/caption", file), 
                               os.path.join("/data6/myt/SummaryCaption/data/caption", idxout))