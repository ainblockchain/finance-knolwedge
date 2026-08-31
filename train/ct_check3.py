import compressed_tensors, os
root=os.path.dirname(compressed_tensors.__file__)
for dp,_,fs in os.walk(root):
    for f in fs:
        p=os.path.join(dp,f)
        if f.endswith('.py') and 'def unpack_from_int32' in open(p).read():
            s=open(p).read(); i=s.find('def unpack_from_int32'); j=s.find('\ndef ', i+10)
            print(s[i+1800:j])
            k=s.find('def pack_to_int32'); print("---- pack_to_int32 ----"); print(s[k:k+1600])
