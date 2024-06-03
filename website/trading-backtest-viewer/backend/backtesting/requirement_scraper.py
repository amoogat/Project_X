import os
with open ('services.py', 'r') as f:
    i = 0
    lister=[]
    for line in f.readlines():
        i +=1
        if i == 21:
            break
        if  'from' in line:
            print(line)
        if ',' in line:
            for package in line.split(','):
                for p in package.split():
                    lister.append(p.replace('import ','').replace(' as ','').replace('from ','').strip())
        else:
            for p in line.split():
                lister.append(p.replace('import ','').replace(' as ','').replace('from ','').strip())
    f.close()

        
with open ('views.py', 'r') as f1:
    i = 0
    listy=[]
    for line in f1.readlines():
        i +=1
        if i == 20:
            break
        if  'from' in line:
            print(line)
        if ',' in line:
            for package in line.split(','):
                for p in package.split():
                    listy.append(p.replace('import ','').replace(' as ','').replace('from ','').strip())
        else:
            for p in line.split():
                listy.append(p.replace('import ','').replace(' as ','').replace('from ','').strip())
        
    f1.close()
with open('requirements.txt','r')  as f2:
    for line in f2.readlines():
        if line.split('==')[0] in lister or line.split('==')[0] in listy:
            print(line)
            