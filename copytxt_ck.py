import os
from shutil import copyfile


os.chdir('/media/felipe/UbuntuHD/recfac/dataset/Emotion_labels/Emotion')
dir_list = os.listdir()
for folder in dir_list:
	os.chdir(folder)
	dir_list2 = os.listdir()
	# print(folder)
	# print(dir_list2)

	for subfolder in dir_list2:
		os.chdir(subfolder)
		dir_list3 = os.listdir()

		for txt in dir_list3:
			if(".txt" in txt):
				print(txt)
				a = "/media/felipe/UbuntuHD/recfac/dataset/ck-redimensionado/cohn-kanade-images/"+folder+"/"+subfolder+"/"+txt
				copyfile(txt, a)

		os.chdir('../')


	os.chdir('../')

	# print(dir_list)
	# for x in dir_list:
	# 	if(".txt" in x):
	# 		print(x)

	# copyfile()
