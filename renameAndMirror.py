import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from shutil import copyfile
import imghdr



def flip_image(image_path):
    """
    Flip or mirror the image

    @param image_path: The path to the image to edit
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    rotated_image = image_obj.transpose(Image.FLIP_LEFT_RIGHT)
    # rotated_image.save(image_path+"_mirrored.jpg")
    rotated_image.save(image_path+"_mirrored.png")
    # rotated_image.show()

def trav_dir(dirpath):
	os.chdir(dirpath)
	dir_list = os.listdir()

	#travers current directory and if directoy found call itself
	for x in dir_list:
		if(os.path.isdir(x)):
			trav_dir(x)
		#imghdr.what return mime type of the image
		elif(imghdr.what(x) in ['jpeg', 'jpg', 'png']):
			flip_image(x)

	#reached directory with no directory
	os.chdir('./..')

# trav_dir('/media/felipe/02F2FC09F2FBFF2B/IMAGENSSSSSS/alinhado/4')
trav_dir('./dataset/mug-pronto-sem-mirror')














