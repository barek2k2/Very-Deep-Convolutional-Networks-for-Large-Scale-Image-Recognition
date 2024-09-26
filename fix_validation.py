
import os
import shutil

path = './data/val'
new_path = './data/validation'

with open(f'{path}/val_annotations.txt') as file:
  count = 0

  for line in file:
    imgName, type, box1, box2, box3, box4 = line.split()

    typeDir = f'{new_path}/{type}'

    if (not os.path.isdir(typeDir)):
      os.makedirs(f'{typeDir}/images')
    
    _, _, files = next(os.walk(typeDir))
    count = len(files)

    oldPath = f'{path}/images/{imgName}'
    newPath = f'{typeDir}/images/{type}_{count}.JPEG'

    shutil.copyfile(oldPath, newPath)

    with open(f'{typeDir}/{type}_boxes.txt', 'a+') as boxesFile:
      boxesFile.write(f'{type}_{count}.JPEG {box1} {box2} {box3} {box4}\n')

    




