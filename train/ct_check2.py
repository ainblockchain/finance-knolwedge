import compressed_tensors, os, re, inspect
root=os.path.dirname(compressed_tensors.__file__)
hits=[]
for dp,_,fs in os.walk(root):
    for f in fs:
        if f.endswith('.py'):
            p=os.path.join(dp,f); s=open(p).read()
            if 'def unpack_from_int32' in s or 'def pack_to_int32' in s: hits.append(p)
print("정의 위치:", hits)
for p in hits:
    s=open(p).read()
    i=s.find('def unpack_from_int32'); print(s[i:i+1800])
